#!/usr/bin/env python3
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
"""Offline ASR dispatcher for Qwen3-ASR (Intel XPU/CPU), NeMo Canary,
faster-whisper, local Transformers ASR models, and whisper.cpp.

Inference uses local model caches only. Run install.sh while online to prefetch models.

The dispatcher imports only the Python standard library. It re-executes this
same file inside the selected model's venv, so the incompatible ML dependency
stacks never share an interpreter or site-packages directory.
"""
from __future__ import annotations

import argparse
import array
from datetime import datetime
import importlib
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
VENVS_DIR = APP_DIR / "venvs"
MODELS_DIR = APP_DIR / "models"
# Lower bounds are intentionally close to the official ggml file sizes so an
# old/incomplete download is not mistaken for a usable model. install.sh uses
# the same thresholds before promoting a download into its final filename.
WHISPER_CPP_MAX_DECODERS = 8  # whisper.cpp builds reject beam widths above this hard limit.
WHISPER_CPP_MIN_BYTES = {
    "large-v3-turbo": 1_620_000_000,
    "large-v3": 3_080_000_000,
    "base.en": 147_000_000,
}

MODEL_TO_ENGINE = {
    "qwen3": "qwen",
    "qwen3-1.7b": "qwen",
    "canary": "canary",                 # compatibility alias for Canary-1B-Flash
    "canary-1b": "canary",
    "canary-1b-flash": "canary",
    "canary-1b-v2": "canary",
    "faster-whisper": "faster-whisper",
    "seamless-m4t-v2": "seamless-m4t",
    "granite-speech-3.3-2b": "granite-speech",
    "granite-speech-3.3-8b": "granite-speech",
    "whisper-cpp": "whisper-cpp",
    "granite-speech-5.0-turboctc": "granite5-ctc",
    "cohere-transcribe-2b": "cohere-asr",
    "moss-transcribe-diarize": "moss-diarize",
    "moss-transcribe-preview-2b": "moss-preview",
    "ark-asr-3b": "ark-asr",
    # MIT-licensed CTranslate2 conversion; uses the faster-whisper worker.
    "distil-whisper-large-v3.5": "faster-whisper",
}
ENGINE_TO_VENV = {
    "qwen": "qwen",
    "canary": "canary",
    "faster-whisper": "faster-whisper",
    "seamless-m4t": "transformers-asr",
    "granite-speech": "transformers-asr",
    "granite5-ctc": "granite5-ctc",
    "cohere-asr": "cohere-asr",
    "moss-diarize": "moss-diarize",
    "moss-preview": "moss-preview",
    "ark-asr": "ark-asr",
}
# These engines share the local-only worker wrapper below, but keep separate
# dependency environments. Keeping the registry here prevents a new model from
# being accepted by the CLI and routed to the "Unknown worker engine" fallback.
OPEN_ASR_ENGINES = frozenset({
    "granite5-ctc", "cohere-asr", "moss-diarize", "moss-preview", "ark-asr",
})
CANARY_MODEL_IDS = {
    "canary": "nvidia/canary-1b-flash",
    "canary-1b": "nvidia/canary-1b",
    "canary-1b-flash": "nvidia/canary-1b-flash",
    "canary-1b-v2": "nvidia/canary-1b-v2",
}
TRANSFORMERS_MODEL_IDS = {
    "seamless-m4t-v2": "facebook/seamless-m4t-v2-large",
    "granite-speech-3.3-2b": "ibm-granite/granite-speech-3.3-2b",
    "granite-speech-3.3-8b": "ibm-granite/granite-speech-3.3-8b",
    "granite-speech-5.0-turboctc": "ibm-granite/granite-speech-5.0-470m-turboctc",
    "cohere-transcribe-2b": "CohereLabs/cohere-transcribe-03-2026",
    "moss-transcribe-diarize": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "moss-transcribe-preview-2b": "OpenMOSS-Team/MOSS-Transcribe-preview-2B",
    "ark-asr-3b": "Edge0/ARK-ASR-3B",
}
DISTIL_WHISPER_REPO_ID = "distil-whisper/distil-large-v3.5-ct2"
# Keep this mapping aligned with faster-whisper's own size-name aliases. In
# particular, large-v3-turbo is hosted by MobiusLab, not under Systran.
FASTER_WHISPER_REPO_IDS = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
    "base.en": "Systran/faster-whisper-base.en",
}
# Read-only compatibility for a model cache created by an older/custom build
# that stored Turbo under the stale Systran cache key. Do not download from it.
FASTER_WHISPER_LEGACY_REPO_IDS = {
    "large-v3-turbo": ("Systran/faster-whisper-large-v3-turbo",),
}


def faster_whisper_repo_id(model_name: str) -> str:
    """Return the canonical Hub repo for a supported faster-whisper size alias."""
    try:
        return FASTER_WHISPER_REPO_IDS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported faster-whisper model size: {model_name}") from exc


def resolve_faster_whisper_model_source(model_name: str) -> str:
    """Resolve canonical cached weights, then accept a complete legacy Turbo cache.

    If neither snapshot is found, return the size alias: faster-whisper handles
    its own alias mapping, and the offline policy prevents it from downloading.
    """
    repo_id = faster_whisper_repo_id(model_name)
    cached = resolve_hf_snapshot(repo_id)
    if cached != repo_id:
        return cached
    for legacy_repo_id in FASTER_WHISPER_LEGACY_REPO_IDS.get(model_name, ()):
        legacy_cached = resolve_hf_snapshot(legacy_repo_id)
        if legacy_cached != legacy_repo_id:
            return legacy_cached
    return model_name


# The published CTranslate2 checkpoint is approximately 1.52 GB. Checking only
# for the presence of model.bin allowed a stale/truncated file to look cached.
DISTIL_WHISPER_MIN_MODEL_BYTES = 1_400_000_000


def is_valid_distil_whisper_model_dir(path: Path) -> bool:
    """Return true only for a complete-looking Distil-Whisper CT2 snapshot."""
    try:
        return (
            path.is_dir()
            and (path / "config.json").is_file()
            and (path / "model.bin").is_file()
            and (path / "model.bin").stat().st_size >= DISTIL_WHISPER_MIN_MODEL_BYTES
        )
    except OSError:
        return False

GRANITE5_LANGUAGES = {"en", "fr", "de", "es", "pt", "it", "ja"}
COHERE_LANGUAGES = {"en", "de", "fr", "it", "es", "pt", "el", "nl", "pl", "ar", "vi", "zh", "ja", "ko"}
ARK_ASR_LANGUAGES = {
    "zh", "en", "de", "ja", "fr", "ko", "es", "pl", "it", "ro",
    "hu", "cs", "nl", "fi", "hr", "sk", "sl", "et", "lt",
}
MOSS_DIARIZE_LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "bul": "Bulgarian", "zh": "Chinese",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "en": "English", "et": "Estonian", "fi": "Finnish", "fr": "French",
    "de": "German", "el": "Greek", "he": "Hebrew", "hi": "Hindi",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian", "mk": "Macedonian",
    "ms": "Malay", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sr": "Serbian", "sk": "Slovak",
    "sl": "Slovenian", "es": "Spanish", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese", "eng": "English",
}


def moss_diarize_prompt(language: str) -> str | None:
    """Return a language-specific prompt for MOSS-Diarize, or None for auto-detect."""
    code = str(language or "auto").strip().lower().replace("_", "-")
    if code == "auto":
        return None
    short_code = code.split("-", 1)[0]
    name = MOSS_DIARIZE_LANGUAGE_NAMES.get(code) or MOSS_DIARIZE_LANGUAGE_NAMES.get(short_code)
    if not name:
        name = f"the language identified by code {code}"
    return (
        f"Transcribe the audio in {name}. For each segment, start with the timestamp "
        "and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end "
        "with the segment timestamp. Preserve the language spoken in the audio; "
        "do not translate it."
    )

# Inference is deliberately offline-only. `install.sh` performs intentional
# prefetches while connectivity is available; running `transcribe` must never
# probe the Hub or attempt a background model download.
OFFLINE_MODEL_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}


def enforce_offline_model_loading(env: dict[str, str] | None = None) -> None:
    """Force Hugging Face/Transformers model resolution to use local files only."""
    target = os.environ if env is None else env
    target.update(OFFLINE_MODEL_ENV)

# SeamlessM4T uses ISO-639-3 / model-specific language tags instead of the
# two-letter CLI convention used by Whisper, Qwen, and Canary. These are the
# likely languages used through this toolkit; unknown tags fail with a useful
# message instead of quietly being interpreted as translations into English.
SEAMLESS_LANGUAGE_CODES = {
    "af": "afr", "ar": "arb", "bg": "bul", "bn": "ben", "cs": "ces",
    "da": "dan", "de": "deu", "el": "ell", "en": "eng", "es": "spa",
    "fa": "pes", "fi": "fin", "fil": "tgl", "fr": "fra", "he": "heb",
    "hi": "hin", "hr": "hrv", "hu": "hun", "id": "ind", "it": "ita",
    "ja": "jpn", "ko": "kor", "mk": "mkd", "ms": "zsm", "nl": "nld",
    "no": "nob", "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus",
    "sk": "slk", "sl": "slv", "sr": "srp", "sv": "swe", "th": "tha",
    "tr": "tur", "uk": "ukr", "ur": "urd", "vi": "vie", "zh": "cmn",
}
GRANITE_SPEECH_LANGUAGES = {"en", "de", "es", "fr", "pt"}

LANGUAGE_NAMES = {
    "en": "English", "bg": "Bulgarian", "de": "German", "es": "Spanish",
    "fr": "French", "it": "Italian", "ru": "Russian", "zh": "Chinese",
    "yue": "Cantonese", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
    "id": "Indonesian", "ar": "Arabic", "tr": "Turkish", "hi": "Hindi",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "fil": "Filipino", "fa": "Persian",
    "el": "Greek", "hu": "Hungarian", "ro": "Romanian", "th": "Thai",
    "vi": "Vietnamese", "ms": "Malay", "uk": "Ukrainian", "hr": "Croatian",
    "sr": "Serbian", "sl": "Slovenian", "no": "Norwegian", "he": "Hebrew",
    "mk": "Macedonian",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}
# Canary-1B and Canary-1B-Flash are EU4 models; only Canary-1B-v2 supports
# the expanded 25-language EU25 ASR set, including Bulgarian.
CANARY_EU4_LANGUAGES = {"en", "de", "es", "fr"}
CANARY_V2_LANGUAGES = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
}
# Qwen3-ASR's documented/current runtime-supported language set. In particular,
# Bulgarian is intentionally absent: reject it before loading XPU/CPU models so
# an unsupported language is never misreported as an XPU failure.
QWEN_SUPPORTED_LANGUAGES = {
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
    "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
    "da", "fi", "pl", "cs", "fil", "fa", "el", "ro", "hu", "mk",
}


# Languages whose orthography uses spaces between words/phrases. The upstream
# qwen-asr long-audio merger currently joins stripped chunk texts with ""; add
# a separator between native chunks for these languages to avoid e.g.
# "thelecture" at a chunk boundary. CJK text keeps the upstream no-space join.
QWEN_SPACE_SEPARATED_LANGUAGES = {
    "English", "Arabic", "German", "French", "Spanish", "Portuguese",
    "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese",
    "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish",
    "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian",
    "Hungarian", "Macedonian",
}


def print_transcript_block(text: str, *, label: str) -> None:
    """Print a complete chunk/result transcript without arbitrary log truncation."""
    value = str(text or "").strip()
    print(f"{label} ({len(value)} characters; complete text):", flush=True)
    print(value if value else "[no speech detected]", flush=True)


def print_transcript_preview(text: str, *, label: str, output_path: Path,
                             limit: int = 220) -> None:
    """Print an explicitly labelled preview when repeating a huge final transcript is unhelpful."""
    value = str(text or "").strip()
    if len(value) <= limit:
        print_transcript_block(value, label=label)
        return
    print(
        f"{label}: showing first {limit} of {len(value)} characters; "
        f"complete transcript saved to {output_path}",
        flush=True,
    )
    print(value[:limit] + " … [preview ends here]", flush=True)


@contextmanager
def qwen_native_chunk_progress(model):
    """Print Qwen native-chunk transcripts as each chunk finishes.

    Qwen3-ASR's public ``transcribe()`` currently calls ``_infer_asr`` for all
    chunks and only returns the merged result after every chunk has completed.
    Wrap that internal inference method for this one model call, dispatch each
    chunk separately, and print its decoded text immediately. The original raw
    outputs are returned unchanged to Qwen, so its existing parsing/merging
    remains authoritative. This is diagnostic output only; it does not create
    audio files, change chunk boundaries, or add overlap.
    """
    module_name = "qwen_asr.inference.qwen3_asr"
    implementation = sys.modules.get(module_name)
    parser = getattr(implementation, "parse_asr_output", None) if implementation else None
    original_infer = getattr(model, "_infer_asr", None)
    captured_chunks: list[dict[str, Any]] = []
    if not callable(parser) or not callable(original_infer):
        print(
            "WARN: this qwen-asr version does not expose the chunk inference hooks needed "
            "for incremental chunk text; continuing without per-chunk previews.",
            file=sys.stderr, flush=True,
        )
        yield captured_chunks
        return

    had_instance_override = "_infer_asr" in getattr(model, "__dict__", {})
    old_instance_override = getattr(model, "__dict__", {}).get("_infer_asr")

    def infer_with_progress(contexts, wavs, languages):
        if not wavs:
            return original_infer(contexts, wavs, languages)
        if not (len(contexts) == len(wavs) == len(languages)):
            # Preserve upstream validation/errors for unexpected call shapes.
            return original_infer(contexts, wavs, languages)

        raw_outputs = []
        cursor_samples = 0
        total = len(wavs)
        sample_rate = 16000  # Qwen's normalized waveform sample rate.
        for index, (context, wav, language) in enumerate(zip(contexts, wavs, languages), 1):
            sample_count = len(wav)
            start = cursor_samples / sample_rate
            end = (cursor_samples + sample_count) / sample_rate
            print(f"[{index}/{total}] {start:.1f}s–{end:.1f}s", flush=True)

            chunk_outputs = original_infer([context], [wav], [language])
            raw_outputs.extend(chunk_outputs)
            raw = chunk_outputs[0] if chunk_outputs else ""
            # This is diagnostic output. Qwen's own transcribe() will parse the
            # returned raw outputs again and construct the authoritative final text.
            try:
                detected_language, chunk_text = parser(raw, user_language=language)
                chunk_text = str(chunk_text or "").strip()
            except Exception as exc:
                # A diagnostic parser failure must never discard the raw model
                # output or turn an otherwise successful inference into failure.
                detected_language, chunk_text = language, ""
                print(
                    f"WARN: could not decode preview for Qwen chunk {index}/{total} "
                    f"({type(exc).__name__}: {exc}); raw output is preserved.",
                    file=sys.stderr, flush=True,
                )
            print_transcript_block(chunk_text, label=f"Chunk transcript [{index}/{total}]")
            captured_chunks.append({
                "index": index, "start": start, "end": end,
                "language": detected_language, "text": chunk_text,
            })
            cursor_samples += sample_count
        return raw_outputs

    try:
        model._infer_asr = infer_with_progress
    except Exception as exc:
        print(
            f"WARN: cannot attach Qwen chunk-progress hook ({type(exc).__name__}: {exc}); "
            "continuing without per-chunk previews.",
            file=sys.stderr, flush=True,
        )
        yield captured_chunks
        return
    try:
        yield captured_chunks
    finally:
        if had_instance_override:
            model._infer_asr = old_instance_override
        else:
            try:
                delattr(model, "_infer_asr")
            except AttributeError:
                pass


@contextmanager
def qwen_native_chunk_settings(max_chunk_seconds: int):
    """Bound Qwen's own long-audio chunks without creating FFmpeg audio shards.

    qwen-asr currently sets MAX_ASR_INPUT_SECONDS to 1200 seconds (20 minutes),
    which is too large for some XPU memory budgets. Its transcribe() method
    consults the module-level constant below when applying its native
    low-energy splitter. We temporarily lower that cap for this process call.

    The current upstream text merger strips each chunk then joins chunk strings
    without a separator. Patch the parser only during this call so languages
    that use spaces retain a boundary between native chunks. Restore both
    globals unconditionally after success or failure.
    """
    if isinstance(max_chunk_seconds, bool) or not 30 <= int(max_chunk_seconds) <= 1200:
        raise ValueError("--qwen-internal-chunk-seconds must be between 30 and 1200")
    module_name = "qwen_asr.inference.qwen3_asr"
    try:
        implementation = sys.modules.get(module_name) or importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            "This qwen-asr version does not expose the supported native chunk settings; "
            "update qwen-asr or omit --internal-chunking."
        ) from exc
    if not hasattr(implementation, "MAX_ASR_INPUT_SECONDS") or not callable(
        getattr(implementation, "parse_asr_output", None)
    ):
        raise RuntimeError(
            "This qwen-asr version lacks the native chunk controls required for safe long-audio inference; "
            "update qwen-asr or omit --internal-chunking."
        )

    old_max = implementation.MAX_ASR_INPUT_SECONDS
    old_parser = implementation.parse_asr_output

    def parse_with_chunk_separator(raw_output: Any, user_language: str | None = None):
        language, text = old_parser(raw_output, user_language=user_language)
        if text and str(language).strip().title() in QWEN_SPACE_SEPARATED_LANGUAGES:
            text = str(text).rstrip() + " "
        return language, text

    implementation.MAX_ASR_INPUT_SECONDS = int(max_chunk_seconds)
    implementation.parse_asr_output = parse_with_chunk_separator
    try:
        yield
    finally:
        implementation.MAX_ASR_INPUT_SECONDS = old_max
        implementation.parse_asr_output = old_parser


def canary_languages_for_args(args: argparse.Namespace) -> set[str]:
    """Return language codes supported by the selected Canary checkpoint."""
    selected_id = os.environ.get(
        "CANARY_MODEL", CANARY_MODEL_IDS.get(args.model, CANARY_MODEL_IDS["canary"])
    ).rstrip("/").rsplit("/", 1)[-1]
    if selected_id == "canary-1b-v2":
        return CANARY_V2_LANGUAGES
    return CANARY_EU4_LANGUAGES


def validate_qwen_language(language: str) -> None:
    """Fail early for explicitly requested languages unsupported by Qwen3-ASR."""
    if language == "auto":
        return
    code = language.lower()
    if code not in QWEN_SUPPORTED_LANGUAGES:
        name = LANGUAGE_NAMES.get(code, language)
        if code in CANARY_V2_LANGUAGES:
            recommendation = (
                "Use --model canary-1b-v2 for this European language, or "
                "--model faster-whisper --whisper-size large-v3."
            )
        else:
            recommendation = "Use --model faster-whisper --whisper-size large-v3 for this language."
        raise ValueError(
            f"Qwen3-ASR does not support {name} (language code {code!r}) in this runtime. "
            f"{recommendation} This is a language-support issue, not an XPU failure."
        )


def validate_canary_language(args: argparse.Namespace) -> None:
    """Fail before loading NeMo if the language is outside this Canary model's coverage."""
    if args.lang == "auto" or args.lang.lower() in canary_languages_for_args(args):
        return
    if "bg" in canary_languages_for_args(args):
        raise ValueError(
            "Canary-1B-v2 supports: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, "
            "it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk. "
            "Use --model faster-whisper for other languages."
        )
    raise ValueError(
        "Canary-1B/Flash support only en, de, es, fr for ASR. "
        "Use --model canary-1b-v2 or faster-whisper for other languages."
    )


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("RUN: " + shlex.join([str(part) for part in cmd]), flush=True)
    return subprocess.run(
        [str(x) for x in cmd], check=check, text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        env=env,
    )


def normalized_word(word: str) -> str:
    """Normalize token punctuation/case only for overlap comparison."""
    return re.sub(r"[^\w]+", "", word, flags=re.UNICODE).casefold()


def _bounded_boundary_window(words: list[str], *, from_end: bool,
                             max_words: int, max_chars: int) -> list[str]:
    """Return a bounded word window at one transcript boundary.

    Bounding both words and characters prevents very long tokens or pathological
    decoder output from turning overlap matching into an unbounded operation.
    """
    seq = words[-max_words:] if from_end else words[:max_words]
    it = reversed(seq) if from_end else iter(seq)
    kept: list[str] = []
    chars = 0
    for word in it:
        # Keep the character budget strict, including pathological single tokens.
        if len(word) > max_chars or chars + len(word) > max_chars:
            break
        kept.append(word)
        chars += len(word)
    return list(reversed(kept)) if from_end else kept


def _at_most_one_edit(a: str, b: str) -> bool:
    """Whether two strings differ by one insertion, deletion, or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) > len(b):
            i += 1
        elif len(b) > len(a):
            j += 1
        else:
            i += 1
            j += 1
    if i < len(a) or j < len(b):
        edits += 1
    return edits <= 1


def _word_match_cost(a: str, b: str) -> tuple[float, bool, int]:
    """Return edit cost, whether the words are credible matches, and evidence chars.

    Fuzzy word matching is intentionally stricter than whole-string similarity.
    A SequenceMatcher ratio alone wrongly considers pairs such as ``after`` and
    ``other`` similar because they share a long suffix. For ASR overlap we accept
    only a single-character edit in a word of at least five characters, with the
    first two letters preserved. Ambiguous or semantically distinct short words
    stay unmatched; boundary-level overlap still tolerates inserted/deleted words.
    """
    aa, bb = normalized_word(a), normalized_word(b)
    if aa and aa == bb:
        return 0.0, True, len(aa)
    if not aa or not bb:
        return 1.15, False, 0
    shortest = min(len(aa), len(bb))
    if (shortest >= 5 and aa[:2] == bb[:2] and _at_most_one_edit(aa, bb)):
        return 0.28, True, shortest
    return 1.15, False, 0


def _best_fuzzy_boundary_overlap(left_words: list[str], right_words: list[str], *,
                                 max_words: int = 45, max_chars: int = 360,
                                 min_overlap: int = 3) -> tuple[int, int] | None:
    """Find a credible suffix/prefix alignment using bounded word edit distance.

    Returns (number of left words represented, number of right-prefix words to
    replace). The comparison is anchored to the two chunk edges, and is capped
    by both word count and character count. A single DP table is built for each
    possible left-suffix start, then reused for every right-prefix endpoint; this
    keeps worst-case work around O(n^2*m), rather than rebuilding O(n*m) tables
    for every pair of candidate lengths.

    Low-confidence alignments return None so the caller keeps both strings.
    """
    if min_overlap < 2 or max_words < min_overlap or max_chars < 1:
        raise ValueError("invalid overlap matching bounds")
    left = _bounded_boundary_window(left_words, from_end=True,
                                    max_words=max_words, max_chars=max_chars)
    right = _bounded_boundary_window(right_words, from_end=False,
                                     max_words=max_words, max_chars=max_chars)
    n, m = len(left), len(right)
    if n < min_overlap or m < min_overlap:
        return None

    pair_cost: list[list[tuple[float, bool, int]]] = [
        [_word_match_cost(a, b) for b in right] for a in left
    ]
    gap_cost = 0.90
    best: tuple[float, int, int] | None = None  # score, left length, right length

    # Each candidate chooses a different suffix start on the left. Compute the
    # full right-prefix table once for that suffix and inspect all endpoints.
    for left_start in range(0, n - min_overlap + 1):
        left_len = n - left_start
        dp = [[0.0] * (m + 1) for _ in range(left_len + 1)]
        trace = [[""] * (m + 1) for _ in range(left_len + 1)]
        for i in range(1, left_len + 1):
            dp[i][0] = i * gap_cost
            trace[i][0] = "D"
        for j in range(1, m + 1):
            dp[0][j] = j * gap_cost
            trace[0][j] = "I"

        for i in range(1, left_len + 1):
            left_index = left_start + i - 1
            for j in range(1, m + 1):
                sub_cost = pair_cost[left_index][j - 1][0]
                choices = (
                    (dp[i - 1][j - 1] + sub_cost, "M"),
                    (dp[i - 1][j] + gap_cost, "D"),
                    (dp[i][j - 1] + gap_cost, "I"),
                )
                value, action = min(choices, key=lambda item: item[0])
                dp[i][j], trace[i][j] = value, action

        for right_len in range(min_overlap, m + 1):
            # Both ends of the proposed overlap must be aligned word matches.
            if trace[left_len][right_len] != "M":
                continue
            if (not pair_cost[left_start][0][1]
                    or not pair_cost[n - 1][right_len - 1][1]):
                continue

            i, j = left_len, right_len
            matched_words = 0
            matched_chars = 0
            starts_with_credible_match = False
            valid_path = True
            while i > 0 or j > 0:
                action = trace[i][j]
                if action == "M":
                    original_left_i = left_start + i - 1
                    _cost, credible, char_count = pair_cost[original_left_i][j - 1]
                    if credible:
                        matched_words += 1
                        matched_chars += char_count
                    if i == 1 and j == 1 and credible:
                        starts_with_credible_match = True
                    i -= 1
                    j -= 1
                elif action == "D" and i > 0:
                    i -= 1
                elif action == "I" and j > 0:
                    j -= 1
                else:
                    valid_path = False
                    break
            if not valid_path or i != 0 or j != 0 or not starts_with_credible_match:
                continue

            quality = 1.0 - dp[left_len][right_len] / max(left_len, right_len)
            coverage = matched_words / max(left_len, right_len)
            if (matched_words < min_overlap or matched_chars < 10
                    or quality < 0.62 or coverage < 0.55):
                continue
            score = (matched_chars + 2.0 * matched_words + 3.0 * quality
                     - 0.35 * abs(left_len - right_len))
            if best is None or score > best[0]:
                best = (score, left_len, right_len)

    if best is None:
        return None
    return best[1], best[2]

def merge_overlapping_transcripts(texts: list[str], max_words: int = 45,
                                  min_overlap: int = 3, max_chars: int = 360) -> str:
    """Merge ASR chunks with conservative, bounded fuzzy overlap detection.

    The comparison is limited to the previous suffix and next prefix (at most
    ``max_words`` and ``max_chars`` at either edge). It tolerates punctuation,
    case, small spelling differences, and a modest number of inserted/deleted
    words. If the edge alignment is ambiguous, it preserves both chunks rather
    than silently deleting possibly unique speech.
    """
    if not texts:
        return ""
    if max_words < 1 or max_chars < 1:
        raise ValueError("max_words and max_chars must be positive")
    merged = texts[0].split()
    for current_text in texts[1:]:
        current = current_text.split()
        if not current:
            continue
        if not merged:
            merged.extend(current)
            continue
        match = _best_fuzzy_boundary_overlap(
            merged, current, max_words=max_words, max_chars=max_chars,
            min_overlap=min_overlap,
        )
        if match:
            left_count, right_count = match
            # Treat the newer chunk as the canonical rendering of the shared
            # audio. Replace the aligned old suffix with the aligned new prefix
            # rather than merely deleting the new prefix. This retains spelling
            # corrections, punctuation, and credible inserted words that the
            # later decode heard in the overlap. The raw sidecar preserves both
            # original chunk hypotheses for audit. Alignment is boundary-anchored
            # and confidence-gated; when uncertain, the no-match branch preserves
            # both texts instead of deleting content.
            merged = merged[:-left_count] + current[:right_count]
            merged.extend(current[right_count:])
        else:
            merged.extend(current)
    out = " ".join(merged)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([([{])\s+", r"\1", out)
    return out.strip()


RUN_ARTIFACT_NAMES = {
    "run.json", "transcript.txt", "transcript.raw.txt", "transcript.chunks.json",
    "transcript.segments.txt", "transcript.srt", "transcript.vtt",
}


def effective_model_label(args: argparse.Namespace, engine: str | None = None) -> str:
    """Stable human-readable model label used for output folder names and run metadata."""
    engine = engine or MODEL_TO_ENGINE[args.model]
    if engine in {"faster-whisper", "whisper-cpp"}:
        size = args.whisper_size or ("large-v3" if str(args.lang).lower() == "bg" else "large-v3-turbo")
        return f"{args.model}-{size}"
    if engine == "canary":
        return CANARY_MODEL_IDS.get(args.model, args.model).rsplit("/", 1)[-1]
    return args.model


def _folder_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_")
    return cleaned or "run"


def output_directory_for(args: argparse.Namespace, engine: str | None = None) -> Path:
    """Resolve --output as a directory, or generate a collision-safe run directory."""
    if args.output:
        target = Path(args.output).expanduser().resolve()
        if target.exists() and not target.is_dir():
            raise ValueError(
                f"--output now requires a directory, but this path is a file: {target}. "
                "Choose a new directory path (transcripts are now stored as transcript.txt inside it)."
            )
        return target
    source = Path(args.audio).expanduser().resolve()
    label = _folder_component(effective_model_label(args, engine))
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    stem = _folder_component(source.stem)
    base = Path.cwd() / f"{stem}.{label}.{timestamp}"
    target = base
    suffix = 2
    while target.exists():
        target = Path(str(base) + f"-{suffix:02d}")
        suffix += 1
    return target.resolve()


def output_path_for(args: argparse.Namespace) -> Path:
    """Canonical transcript path; all other artifacts are siblings inside its run folder."""
    return output_directory_for(args) / "transcript.txt"


def _write_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run.json"
    tmp = output_dir / "run.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def prepare_run_output(args: argparse.Namespace, engine: str, output_dir: Path) -> dict[str, Any]:
    """Create a run folder safely and write its identifying manifest."""
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path must be a directory, not a file: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in RUN_ARTIFACT_NAMES if (output_dir / name).exists()]
    if collisions and not getattr(args, "overwrite_output", False):
        raise FileExistsError(
            f"Output directory already contains artifacts from a prior run: {output_dir} "
            f"({', '.join(sorted(collisions))}). Choose another directory or pass --overwrite-output."
        )
    if collisions:
        for name in RUN_ARTIFACT_NAMES:
            path = output_dir / name
            if path.is_file() or path.is_symlink():
                path.unlink()
    source = Path(args.audio).expanduser().resolve()
    label = effective_model_label(args, engine)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished_at": None,
        "input_audio": str(source),
        "audio_basename": source.name,
        "model": args.model,
        "model_label": label,
        "engine": engine,
        "language": args.lang,
        "whisper_size": args.whisper_size if engine in {"faster-whisper", "whisper-cpp"} else None,
        "primary_transcript": "transcript.txt",
        "artifacts": {
            "transcript": "transcript.txt",
            "raw_transcript": "transcript.raw.txt",
            "chunk_log": "transcript.chunks.json",
            "run_manifest": "run.json",
        },
        "options": {
            "threads": args.threads,
            "chunk_seconds_quiet": args.chunk_seconds_quiet,
            "chunk_seconds": args.chunk_seconds,
            "overlap_seconds": args.overlap_seconds,
            "internal_chunking": bool(args.internal_chunking),
        },
    }
    if engine == "faster-whisper":
        manifest["artifacts"].update({"segments": "transcript.segments.txt", "srt": "transcript.srt", "vtt": "transcript.vtt"})
    _write_run_manifest(output_dir, manifest)
    return manifest


def finish_run_manifest(output_dir: Path, status: str, exit_code: int = 0, error: str | None = None) -> None:
    path = output_dir / "run.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    manifest["status"] = status
    manifest["exit_code"] = int(exit_code)
    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if error:
        manifest["error"] = error
    _write_run_manifest(output_dir, manifest)


def _argv_with_output_directory(argv: list[str], output_dir: Path) -> list[str]:
    """Replace any caller output flag with the single absolute run directory."""
    kept: list[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item in {"--output", "-o"}:
            i += 2
            continue
        if item.startswith("--output=") or item.startswith("-o="):
            i += 1
            continue
        kept.append(item)
        i += 1
    kept.extend(["--output", str(output_dir)])
    return kept


def ffprobe_audio(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,codec_name", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    obj = json.loads(proc.stdout)
    streams = obj.get("streams", [])
    if not streams:
        raise RuntimeError(f"No audio stream found in {path}")
    stream = streams[0]
    return {
        "sample_rate": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "codec": stream.get("codec_name", ""),
        "duration": float(obj.get("format", {}).get("duration", 0) or 0),
    }


def prepare_audio(source: Path, temp_dir: Path) -> tuple[Path, float]:
    """Ensure a 16 kHz mono PCM WAV input and return its actual duration."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required; rerun install.sh or install FFmpeg.")
    info = ffprobe_audio(source)
    if info["duration"] <= 0:
        raise RuntimeError(f"Input has invalid duration: {source}")
    if info["sample_rate"] == 16000 and info["channels"] == 1 and info["codec"] == "pcm_s16le" and source.suffix.lower() == ".wav":
        return source, info["duration"]
    converted = temp_dir / "transcribe_input_16k_mono.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", converted])
    converted_info = ffprobe_audio(converted)
    if converted_info["sample_rate"] != 16000 or converted_info["channels"] != 1 or converted_info["duration"] <= 0:
        raise RuntimeError("FFmpeg normalization did not produce a valid 16 kHz mono audio file.")
    return converted, converted_info["duration"]


def effective_chunk_settings(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve a hard-capped per-call window and overlap, including direct worker use."""
    legacy_canary = getattr(args, "model", "") in {"canary", "canary-1b", "canary-1b-flash"}
    seamless = getattr(args, "model", "") == "seamless-m4t-v2"
    chunk_seconds = getattr(args, "chunk_seconds", None)
    if chunk_seconds is None:
        chunk_seconds = 35.0 if legacy_canary else (50.0 if seamless else 300.0)
    overlap_seconds = getattr(args, "overlap_seconds", None)
    if overlap_seconds is None:
        if legacy_canary:
            overlap_seconds = min(7.0, chunk_seconds * 0.2)
        elif seamless:
            overlap_seconds = min(8.0, chunk_seconds * 0.16)
        else:
            overlap_seconds = min(15.0, chunk_seconds * 0.1)
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0 or chunk_seconds > 300:
        raise ValueError("--chunk-seconds must be finite, greater than zero, and no more than 300 seconds")
    if legacy_canary and chunk_seconds >= 40:
        raise ValueError("Legacy Canary-1B/Flash chunks must be shorter than 40 seconds")
    if seamless and chunk_seconds > 60 and not getattr(args, "internal_chunking", False):
        raise ValueError("SeamlessM4T v2 external windows are capped at 60 seconds per model call; use --chunk-seconds 50 or less, or explicitly select --internal-chunking for the full-file encoder-attention path.")
    if not math.isfinite(overlap_seconds) or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("--overlap-seconds must be >= 0 and less than --chunk-seconds")
    return float(chunk_seconds), float(overlap_seconds)


def chunk_intervals(duration: float, chunk_seconds: float,
                    overlap_seconds: float) -> list[tuple[float, float]]:
    """Create balanced windows with a fixed total overlap and no tiny tail shard.

    The nominal, non-overlapping core is divided evenly. Each interior boundary
    adds half the requested overlap to each adjacent window. The chunk count is
    increased until *every actual window including overlap* fits in chunk_seconds.
    For example, 301s with a 300s cap and 15s overlap becomes two ~158s windows,
    not a 300s window followed by a 1s tail (or a 16s tail).
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and greater than zero")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0 or chunk_seconds > 300:
        raise ValueError("--chunk-seconds must be finite, greater than zero, and no more than 300 seconds")
    if not math.isfinite(overlap_seconds) or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("--overlap-seconds must be >= 0 and less than --chunk-seconds")
    if duration <= chunk_seconds:
        return [(0.0, duration)]

    count = max(2, math.ceil(duration / chunk_seconds))
    # Find the smallest count for which the balanced windows, including overlap,
    # obey the hard cap. Two chunks have only one outer edge apiece; three or
    # more have interior chunks with overlap at both ends.
    while True:
        core = duration / count
        max_window = core + (overlap_seconds / 2.0 if count == 2 else overlap_seconds)
        if max_window <= chunk_seconds + 1e-7:
            break
        count += 1

    intervals: list[tuple[float, float]] = []
    half_overlap = overlap_seconds / 2.0
    for i in range(count):
        start = 0.0 if i == 0 else i * core - half_overlap
        end = duration if i == count - 1 else (i + 1) * core + half_overlap
        start = max(0.0, start)
        end = min(duration, end)
        if end <= start:
            raise RuntimeError("Internal chunk planner produced an empty interval")
        if end - start > chunk_seconds + 1e-5:
            raise RuntimeError("Internal chunk planner exceeded the configured maximum window")
        intervals.append((start, end))
    return intervals


def make_chunks(audio: Path, duration: float, temp_dir: Path, chunk_seconds: float,
                overlap_seconds: float) -> list[tuple[Path, float, float]]:
    intervals = chunk_intervals(duration, chunk_seconds, overlap_seconds)
    if len(intervals) == 1 and intervals[0] == (0.0, duration):
        return [(audio, 0.0, duration)]
    result: list[tuple[Path, float, float]] = []
    for index, (start, end) in enumerate(intervals):
        chunk = temp_dir / f"chunk_{index:04d}_{start:010.2f}.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{start:.3f}", "-i", audio, "-t", f"{end-start:.3f}",
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", chunk])
        if not chunk.is_file() or chunk.stat().st_size < 2048:
            raise RuntimeError(f"Chunk {index + 1} failed validation: {chunk}")
        actual = ffprobe_audio(chunk)["duration"]
        if actual <= 0 or actual > chunk_seconds + 0.15:
            raise RuntimeError(
                f"Chunk {index + 1} violates the {chunk_seconds:g}s maximum: {actual:.2f}s ({chunk})"
            )
        result.append((chunk, start, end))
    return result


def find_low_energy_frame_intervals(
    audio: Path,
    max_chunk_seconds: float,
    *,
    search_seconds: float = 5.0,
    energy_window_seconds: float = 0.1,
    energy_hop_seconds: float | None = None,
    balance_target: bool = False,
) -> tuple[list[tuple[int, int]], int, float]:
    """Find low-energy, exact-sample intervals in a normalized PCM WAV.

    ``balance_target=False`` preserves Canary-v2's strict maximum-chunk-length
    behavior. For the external FFmpeg quiet mode, ``balance_target=True`` treats
    the requested duration as a target: it chooses a balanced chunk count first,
    then adjusts boundaries in the low-energy search window. With no supplied
    hop, the hop equals the energy window length (historic Canary-v2 behavior).
    """
    if not math.isfinite(max_chunk_seconds) or not 20 <= max_chunk_seconds <= 1200:
        raise ValueError("Quiet chunk duration must be between 20 and 1200 seconds")
    if not math.isfinite(search_seconds) or search_seconds < 0:
        raise ValueError("Quiet-boundary search radius must be finite and >= 0")
    if search_seconds * 2 >= max_chunk_seconds:
        raise ValueError("Quiet-boundary search radius must be less than half the target quiet chunk duration")
    if not math.isfinite(energy_window_seconds) or energy_window_seconds <= 0:
        raise ValueError("Quiet-boundary energy window must be finite and > 0")
    if energy_hop_seconds is None:
        energy_hop_seconds = energy_window_seconds
    if not math.isfinite(energy_hop_seconds) or energy_hop_seconds <= 0:
        raise ValueError("Quiet-boundary energy hop must be finite and > 0")

    audio = Path(audio)
    try:
        source = wave.open(str(audio), "rb")
    except (wave.Error, OSError) as exc:
        raise RuntimeError(f"Quiet chunking requires a readable PCM WAV: {audio}") from exc

    with source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        total_frames = source.getnframes()
        compression = source.getcomptype()
        if channels != 1 or sample_width != 2 or sample_rate != 16000 or compression != "NONE":
            raise RuntimeError(
                "Quiet chunking expects the normalized 16-kHz mono PCM16 WAV; "
                f"got channels={channels}, sample_width={sample_width}, sample_rate={sample_rate}, "
                f"compression={compression!r}."
            )
        if total_frames <= 0:
            raise RuntimeError(f"Quiet chunking received an empty WAV: {audio}")

        actual_duration = total_frames / sample_rate
        target_frames = int(round(max_chunk_seconds * sample_rate))
        search_frames = int(round(search_seconds * sample_rate))

        if balance_target:
            # Treat the CLI value as the desired *approximate* duration, not an
            # absolute cap. Choose a balanced count nearest to the target first;
            # increase it only when the nominal average would be much too long.
            # This avoids turning 250 s into three ~83 s chunks when two ~125 s
            # chunks are closer to the requested 120 s target.
            chunk_count = max(1, int(math.floor(total_frames / target_frames + 0.5)))
            max_nominal_frames = target_frames + 2 * search_frames
            while total_frames / chunk_count > max_nominal_frames:
                chunk_count += 1
            if chunk_count == 1:
                return [(0, total_frames)], sample_rate, actual_duration
            # An interior interval can lengthen at both ends when each adjacent
            # quiet boundary moves away from it.
            max_actual_frames = target_frames + 4 * search_frames
        else:
            # Canary-v2's in-process path retains a hard per-chunk maximum.
            if total_frames <= target_frames:
                return [(0, total_frames)], sample_rate, actual_duration
            nominal_span = max(1, target_frames - 2 * search_frames)
            chunk_count = max(2, math.ceil(total_frames / nominal_span))
            max_actual_frames = target_frames

        window_frames = max(1, int(round(energy_window_seconds * sample_rate)))
        hop_frames = max(1, int(round(energy_hop_seconds * sample_rate)))
        left_half = window_frames // 2
        right_half = window_frames - left_half
        min_spacing = min(int(15 * sample_rate), max(1, target_frames // 3))
        boundaries: list[int] = [0]

        for boundary_index in range(1, chunk_count):
            target = round(total_frames * boundary_index / chunk_count)
            candidate_low = max(boundaries[-1] + min_spacing, target - search_frames)
            candidate_high = min(
                total_frames - (chunk_count - boundary_index) * min_spacing,
                target + search_frames,
                total_frames - right_half,
            )
            candidate_low = max(candidate_low, left_half)
            if candidate_low > candidate_high:
                boundary = max(boundaries[-1] + 1, min(target, total_frames - 1))
            else:
                # Use a fine hop (10 ms by default) and include the exact target
                # so tied quiet windows resolve to the nearest boundary.
                centers = list(range(candidate_low, candidate_high + 1, hop_frames))
                centers.append(max(candidate_low, min(target, candidate_high)))
                centers = sorted(set(centers))
                centers = [c for c in centers if c - left_half >= 0 and c + right_half <= total_frames]
                if not centers:
                    boundary = max(candidate_low, min(target, candidate_high))
                else:
                    first_window_start = min(c - left_half for c in centers)
                    last_window_end = max(c + right_half for c in centers)
                    source.setpos(first_window_start)
                    raw = source.readframes(last_window_end - first_window_start)
                    samples = array.array("h")
                    samples.frombytes(raw)
                    if sys.byteorder != "little":
                        samples.byteswap()
                    if len(samples) != last_window_end - first_window_start:
                        raise RuntimeError(f"Unexpected short PCM read while analysing quiet boundary in {audio}")

                    # Prefix sums make each 150-ms candidate evaluation O(1),
                    # rather than rescanning every sample for every hop.
                    squared_prefix = array.array("Q", [0])
                    running_energy = 0
                    for value in samples:
                        running_energy += int(value) * int(value)
                        squared_prefix.append(running_energy)

                    best: tuple[int, int, int] | None = None
                    for center in centers:
                        window_start = center - left_half
                        offset = window_start - first_window_start
                        energy = squared_prefix[offset + window_frames] - squared_prefix[offset]
                        candidate = (energy, abs(center - target), center)
                        if best is None or candidate < best:
                            best = candidate
                    boundary = best[2] if best is not None else max(candidate_low, min(target, candidate_high))

            if boundary <= boundaries[-1] or boundary >= total_frames:
                raise RuntimeError("Quiet-boundary splitter produced an invalid boundary")
            boundaries.append(boundary)

        boundaries.append(total_frames)
        intervals = list(zip(boundaries, boundaries[1:]))
        if any(end <= start or end - start > max_actual_frames for start, end in intervals):
            durations = [round((end - start) / sample_rate, 3) for start, end in intervals]
            raise RuntimeError(
                "Quiet-boundary splitter produced a chunk longer than the balanced-target allowance "
                f"({max_actual_frames / sample_rate:.2f}s for target {max_chunk_seconds:g}s and "
                f"search radius {search_seconds:g}s): {durations}"
            )
        return intervals, sample_rate, actual_duration


def make_low_energy_nonoverlap_chunks(
    audio: Path,
    duration: float,
    temp_dir: Path,
    max_chunk_seconds: int = 120,
    *,
    prefix: str = "canary_v2_low_energy",
    search_seconds: float = 5.0,
    energy_window_seconds: float = 0.1,
    energy_hop_seconds: float | None = None,
) -> list[tuple[Path, float, float]]:
    """Split a normalized PCM WAV in Python at low-energy sample boundaries.

    Used by Canary-v2's optional in-process path. The chunks partition the
    source exactly: no samples are dropped or duplicated and there is no overlap.
    """
    if isinstance(max_chunk_seconds, bool) or not 30 <= int(max_chunk_seconds) <= 1200:
        raise ValueError("--canary-internal-chunk-seconds must be between 30 and 1200")
    audio = Path(audio)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    intervals, sample_rate, actual_duration = find_low_energy_frame_intervals(
        audio, float(max_chunk_seconds), search_seconds=search_seconds,
        energy_window_seconds=energy_window_seconds, energy_hop_seconds=energy_hop_seconds,
    )
    if len(intervals) == 1 and intervals[0] == (0, round(actual_duration * sample_rate)):
        return [(audio, 0.0, actual_duration)]

    results: list[tuple[Path, float, float]] = []
    with wave.open(str(audio), "rb") as source:
        for index, (start_frame, end_frame) in enumerate(intervals, 1):
            chunk_path = temp_dir / f"{prefix}_{index:04d}_{start_frame / sample_rate:010.2f}.wav"
            source.setpos(start_frame)
            remaining = end_frame - start_frame
            with wave.open(str(chunk_path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(sample_rate)
                while remaining > 0:
                    raw = source.readframes(min(sample_rate * 15, remaining))
                    if not raw:
                        raise RuntimeError(f"Unexpected end of source WAV while writing {chunk_path}")
                    written_frames = len(raw) // 2
                    writer.writeframesraw(raw)
                    remaining -= written_frames
                writer.writeframes(b"")
            if not chunk_path.is_file() or chunk_path.stat().st_size < 2048:
                raise RuntimeError(f"Low-energy chunk failed validation: {chunk_path}")
            results.append((chunk_path, start_frame / sample_rate, end_frame / sample_rate))
    return results


def make_low_energy_ffmpeg_chunks(
    audio: Path,
    duration: float,
    temp_dir: Path,
    max_chunk_seconds: float = 120.0,
    *,
    search_seconds: float = 5.0,
    energy_window_seconds: float = 0.150,
    energy_hop_seconds: float = 0.010,
) -> list[tuple[Path, float, float]]:
    """Use FFmpeg to create balanced, low-energy chunks with zero overlap.

    ``max_chunk_seconds`` is treated as the approximate target duration (not a
    hard cap) in this external quiet mode. The planner chooses a balanced chunk
    count first, then searches around each nominal boundary. The boundary search
    is performed on normalized PCM samples, then FFmpeg's
    ``atrim=start_sample/end_sample`` filter writes contiguous, non-overlapping
    chunks. A postcondition checks that every chunk has its expected frame count
    and that total output frames equal source frames, so no audio is lost or
    duplicated by timestamp rounding.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for --chunk-seconds-quiet; install FFmpeg first.")
    audio = Path(audio)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    intervals, sample_rate, actual_duration = find_low_energy_frame_intervals(
        audio, max_chunk_seconds, search_seconds=search_seconds,
        energy_window_seconds=energy_window_seconds, energy_hop_seconds=energy_hop_seconds,
        balance_target=True,
    )
    if len(intervals) == 1 and intervals[0] == (0, round(actual_duration * sample_rate)):
        return [(audio, 0.0, actual_duration)]

    results: list[tuple[Path, float, float]] = []
    total_output_frames = 0
    for index, (start_frame, end_frame) in enumerate(intervals, 1):
        chunk_path = temp_dir / f"quiet_chunk_{index:04d}_{start_frame / sample_rate:010.3f}.wav"
        filter_spec = f"atrim=start_sample={start_frame}:end_sample={end_frame},asetpts=PTS-STARTPTS"
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
            "-map", "0:a:0", "-af", filter_spec, "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", str(chunk_path),
        ])
        if not chunk_path.is_file() or chunk_path.stat().st_size < 2048:
            raise RuntimeError(f"FFmpeg quiet chunk {index} failed validation: {chunk_path}")
        try:
            with wave.open(str(chunk_path), "rb") as check:
                if (check.getframerate() != sample_rate or check.getnchannels() != 1
                        or check.getsampwidth() != 2 or check.getcomptype() != "NONE"):
                    raise RuntimeError(f"FFmpeg produced an unexpected WAV format: {chunk_path}")
                actual_frames = check.getnframes()
        except (wave.Error, OSError) as exc:
            raise RuntimeError(f"FFmpeg did not produce a valid PCM WAV chunk: {chunk_path}") from exc
        expected_frames = end_frame - start_frame
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"FFmpeg quiet chunk {index} has {actual_frames} samples; expected {expected_frames}. "
                "Refusing to continue because exact zero-overlap coverage cannot be guaranteed."
            )
        total_output_frames += actual_frames
        results.append((chunk_path, start_frame / sample_rate, end_frame / sample_rate))

    with wave.open(str(audio), "rb") as source:
        source_frames = source.getnframes()
    if total_output_frames != source_frames:
        raise RuntimeError(
            f"FFmpeg quiet chunks cover {total_output_frames} samples but source has {source_frames}; "
            "refusing to continue because audio coverage is not exact."
        )
    return results


def make_external_chunks(audio: Path, duration: float, temp_dir: Path,
                         args: argparse.Namespace) -> list[tuple[Path, float, float]]:
    """Choose standard overlapping FFmpeg windows or optional quiet no-overlap windows."""
    quiet_cap = getattr(args, "chunk_seconds_quiet", None)
    if quiet_cap is not None:
        search = getattr(args, "chunk_quiet_search_seconds", None)
        window_ms = getattr(args, "chunk_quiet_window_ms", None)
        hop_ms = getattr(args, "chunk_quiet_hop_ms", None)
        search = 8.0 if search is None else float(search)
        window_ms = 200.0 if window_ms is None else float(window_ms)
        hop_ms = 10.0 if hop_ms is None else float(hop_ms)
        chunks = make_low_energy_ffmpeg_chunks(
            audio, duration, temp_dir, float(quiet_cap),
            search_seconds=search,
            energy_window_seconds=window_ms / 1000.0,
            energy_hop_seconds=hop_ms / 1000.0,
        )
        chunk_durations = [end - start for _, start, end in chunks]
        print(
            f"FFmpeg quiet-boundary mode: {duration:.1f}s audio -> {len(chunks)} balanced chunks; "
            f"target {quiet_cap:g}s, actual {min(chunk_durations):.1f}–{max(chunk_durations):.1f}s; "
            f"quietest {window_ms:g}-ms RMS window within +/-{search:g}s "
            f"(scan hop {hop_ms:g}ms), zero overlap / exact sample coverage.",
            flush=True,
        )
        return chunks
    chunk_seconds, overlap_seconds = effective_chunk_settings(args)
    return make_chunks(audio, duration, temp_dir, chunk_seconds, overlap_seconds)


def transformers_internal_chunking_label(engine: str) -> str:
    """Describe the single whole-file call used when an internal path is requested."""
    if engine == "seamless-m4t":
        return "seamless-native-full-file-chunk-attention"
    if engine == "granite-speech":
        return "granite-native-full-file-block-attention"
    raise ValueError(f"No Transformers internal-attention label for {engine!r}")


def select_transformers_asr_chunks(
    audio: Path, duration: float, temp_dir: Path, args: argparse.Namespace, engine: str,
) -> list[tuple[Path, float, float]]:
    """Use the model's attention on one whole file only when explicitly requested.

    SeamlessM4T v2 uses chunked attention inside its speech encoder. Granite Speech
    uses four-second block attention in its speech encoder, but does not expose a
    Transformers-level textual chunk-and-stitch API. In either case this option
    means a single full-file call rather than toolkit-created outer chunks.
    """
    if getattr(args, "internal_chunking", False):
        if engine not in {"seamless-m4t", "granite-speech"}:
            raise ValueError(f"Unsupported Transformers internal-attention mode: {engine}")
        return [(Path(audio), 0.0, float(duration))]
    return make_external_chunks(audio, duration, temp_dir, args)

def join_nonoverlapping_transcripts(texts: list[str]) -> str:
    """Join zero-overlap segment transcripts without fuzzy deduplication."""
    result = ""
    for raw in texts:
        part = str(raw).strip()
        if not part:
            continue
        if not result:
            result = part
        elif part[0] in ",.;:!?%)]}»”’" or result[-1] in "([{«“‘\"":
            result += part
        else:
            result += " " + part
    return result.strip()


def format_subtitle_timestamp(seconds: float, separator: str) -> str:
    """Format nonnegative seconds as SRT (comma) or WebVTT (period) time."""
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


def write_whisper_subtitles(out_path: Path, segments: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write SRT and WebVTT cue sidecars from faster-whisper absolute segments."""
    cues: list[tuple[float, float, str]] = []
    for row in segments:
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(row["start"]))
            end = max(start, float(row["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        # Subtitle formats require a visible positive cue duration. Real
        # faster-whisper segments should already have positive lengths; this
        # protects against zero-length rounded/edge segments in unusual input.
        if end <= start:
            end = start + 0.001
        cues.append((start, end, text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")))
    cues.sort(key=lambda cue: (cue[0], cue[1]))

    srt_blocks: list[str] = []
    vtt_blocks: list[str] = ["WEBVTT", ""]
    for number, (start, end, text) in enumerate(cues, 1):
        srt_blocks.append(
            f"{number}\n{format_subtitle_timestamp(start, ',')} --> "
            f"{format_subtitle_timestamp(end, ',')}\n{text}"
        )
        vtt_blocks.append(
            f"{number}\n{format_subtitle_timestamp(start, '.')} --> "
            f"{format_subtitle_timestamp(end, '.')}\n{text}"
        )

    paths = (
        out_path.with_name(out_path.stem + ".srt"),
        out_path.with_name(out_path.stem + ".vtt"),
    )
    payloads = ("\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""),
                "\n".join(vtt_blocks) + ("\n" if vtt_blocks else ""))
    for path, payload in zip(paths, payloads):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    print(f"Whisper subtitles: {paths[0]} (SRT), {paths[1]} (WebVTT)", flush=True)
    return paths


def write_results(out_path: Path, final_text: str, raw_text: str,
                  chunk_results: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(final_text.strip() + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    raw_path = out_path.with_name(out_path.stem + ".raw.txt")
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    raw_tmp.write_text(raw_text.strip() + "\n", encoding="utf-8")
    os.replace(raw_tmp, raw_path)
    chunks_path = out_path.with_name(out_path.stem + ".chunks.json")
    chunks_tmp = chunks_path.with_suffix(chunks_path.suffix + ".tmp")
    chunks_tmp.write_text(json.dumps(chunk_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(chunks_tmp, chunks_path)
    print(f"\nTranscript: {out_path}", flush=True)
    print(f"Raw chunk/model text (unmerged where applicable): {raw_path}", flush=True)
    print(f"Chunk log: {chunks_path}", flush=True)
    print(f"Run directory: {out_path.parent}", flush=True)


def _is_local_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return (any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
            or any(path.glob("model*.safetensors")) or (path / "model.bin").is_file())


def configured_hf_cache_roots() -> list[Path]:
    """Return configured and toolkit-default Hugging Face cache roots.

    Hugging Face honors several environment variables. Include configured roots
    so complete offline snapshots remain discoverable with custom cache layouts.
    All roots here are local filesystem paths; this helper never contacts Hub.
    """
    candidates: list[Path] = []

    # HF_HUB_CACHE and HUGGINGFACE_HUB_CACHE point directly at the Hub cache
    # directory. TRANSFORMERS_CACHE is also a direct cache directory in legacy
    # Transformers installations. Keep these exact paths intact.
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable, "").strip()
        if value:
            candidates.append(Path(value).expanduser())

    # HF_HOME is the Hugging Face home, with the Hub cache normally under /hub.
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    # Preserve custom cache locations when the worker has a model-specific HF_HOME.
    for value in os.environ.get("SPEECH_TOOLKIT_HF_CACHE_ROOTS", "").split(os.pathsep):
        if value.strip():
            candidates.append(Path(value).expanduser())

    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        candidates.append(Path(xdg).expanduser() / "huggingface" / "hub")

    candidates.extend([
        MODELS_DIR / "hf-cache" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "transcribe-tools-qwen" / "hf-cache" / "hub",
    ])

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            candidate = candidate.expanduser()
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _cached_snapshot_dirs(cache_root: Path, repo_id: str) -> list[Path]:
    """List concrete Hub snapshot directories even if refs/main is missing.

    ``snapshot_download(..., local_files_only=True)`` normally resolves a cache
    through its refs metadata. Some copied, resumed, or older Hub caches contain
    complete snapshots but lack a usable refs/main pointer. The actual snapshot
    directories still have every file needed for offline inference, so inspect
    them directly as a fallback.
    """
    repo_folder = cache_root / ("models--" + repo_id.replace("/", "--"))
    snapshots = repo_folder / "snapshots"
    if not snapshots.is_dir():
        return []
    candidates: list[Path] = []
    ref_main = repo_folder / "refs" / "main"
    if ref_main.is_file():
        try:
            revision = ref_main.read_text(encoding="utf-8").strip()
            preferred = snapshots / revision
            if preferred.is_dir():
                candidates.append(preferred)
        except OSError:
            pass
    try:
        remaining = [p for p in snapshots.iterdir() if p.is_dir() and p not in candidates]
        remaining.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(remaining)
    except OSError:
        pass
    return candidates


def _valid_hf_model_snapshot(path: Path, repo_id: str) -> bool:
    if repo_id == DISTIL_WHISPER_REPO_ID:
        return is_valid_distil_whisper_model_dir(path)
    return _is_local_model_dir(path)


def resolve_hf_snapshot(repo_id: str) -> str:
    """Resolve a complete local snapshot without relying only on Hub refs metadata.

    Inspect concrete snapshot directories before using the Hub helper, so a
    complete cache remains usable even if its refs/main pointer is missing or
    stale. No path in this routine can trigger a network request.
    """
    direct_candidates = [
        MODELS_DIR / repo_id.rsplit("/", 1)[-1],
        Path.home() / "transcribe-tools-qwen" / "models" / repo_id.rsplit("/", 1)[-1],
    ]
    for candidate in direct_candidates:
        if _valid_hf_model_snapshot(candidate, repo_id):
            print(f"Using existing local model: {candidate}", flush=True)
            return str(candidate)

    unique_roots = configured_hf_cache_roots()

    # Most reliable for complete snapshots whose refs/main metadata is absent.
    for cache in unique_roots:
        for snapshot in _cached_snapshot_dirs(cache, repo_id):
            if _valid_hf_model_snapshot(snapshot, repo_id):
                print(f"Using cached Hugging Face snapshot directly: {snapshot}", flush=True)
                return str(snapshot)

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return repo_id
    for cache in unique_roots:
        if not cache.is_dir():
            continue
        try:
            found = Path(snapshot_download(
                repo_id=repo_id, cache_dir=str(cache), local_files_only=True,
            ))
            if _valid_hf_model_snapshot(found, repo_id):
                print(f"Using cached Hugging Face snapshot: {found}", flush=True)
                return str(found)
            if repo_id == DISTIL_WHISPER_REPO_ID:
                try:
                    observed = (found / "model.bin").stat().st_size
                except OSError:
                    observed = 0
                print(
                    f"Ignoring incomplete Distil-Whisper cache at {found}; model.bin is "
                    f"{observed:,} bytes, expected at least {DISTIL_WHISPER_MIN_MODEL_BYTES:,}.",
                    flush=True,
                )
        except Exception as exc:
            # Do not let a bad cache/ref in one root block checking later roots.
            print(f"No resolvable cached snapshot for {repo_id} under {cache}: {exc}", flush=True)
    return repo_id


def _qwen_checkpoint(args: argparse.Namespace) -> str:
    if args.model == "qwen3-1.7b":
        override = args.qwen_checkpoint_17
        repo_id = "Qwen/Qwen3-ASR-1.7B"
    else:
        override = args.qwen_checkpoint_06
        repo_id = "Qwen/Qwen3-ASR-0.6B"
    if override:
        explicit_path = Path(override).expanduser()
        return str(explicit_path) if explicit_path.exists() else resolve_hf_snapshot(override)
    return resolve_hf_snapshot(repo_id)


def effective_max_new_tokens(args: argparse.Namespace, engine: str) -> int:
    """Return an explicit cap or a model-family-specific safe default.

    Qwen uses a 512-token cap by default; other model families use a larger cap
    suitable for their longer output chunks.
    """
    configured = getattr(args, "max_new_tokens", None)
    if configured is not None:
        return int(configured)
    return 512 if engine == "qwen" else 4096


def worker_qwen(args: argparse.Namespace, audio: Path, duration: float,
                temp_dir: Path, out_path: Path) -> None:
    enforce_offline_model_loading()
    import gc
    import torch
    from qwen_asr import Qwen3ASRModel

    validate_qwen_language(args.lang)
    torch.set_num_threads(args.threads)
    checkpoint = _qwen_checkpoint(args)
    max_new_tokens = effective_max_new_tokens(args, "qwen")
    requested_device = args.qwen_device
    try:
        xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception as xpu_probe_error:
        xpu_available = False
        if requested_device == "auto":
            print(f"WARN: torch.xpu.is_available() failed ({type(xpu_probe_error).__name__}: {xpu_probe_error}); using CPU.",
                  file=sys.stderr, flush=True)
        elif requested_device == "xpu":
            raise RuntimeError(f"Intel XPU runtime probe failed: {xpu_probe_error}") from xpu_probe_error
    if requested_device == "xpu" and not xpu_available:
        raise RuntimeError(
            "Qwen was explicitly requested on XPU, but torch.xpu.is_available() is false. "
            "Check the Intel GPU runtime/Level Zero and /dev/dri/renderD*. "
            "Use --qwen-device auto for automatic CPU fallback or --qwen-device cpu to force CPU inference."
        )

    initial_device = "xpu:0" if (xpu_available and requested_device != "cpu") else "cpu"

    def decode_on(device: str) -> tuple[list[str], list[dict[str, Any]]]:
        dtype = torch.bfloat16 if device.startswith("xpu") else torch.float32
        print(
            f"Loading {checkpoint} on {device} (threads={args.threads}, dtype={dtype}, "
            f"max_new_tokens={max_new_tokens})", flush=True,
        )
        model = Qwen3ASRModel.from_pretrained(
            checkpoint, dtype=dtype, device_map=device,
            max_inference_batch_size=1, max_new_tokens=max_new_tokens,
        )
        texts: list[str] = []
        details: list[dict[str, Any]] = []
        native_chunk_details: list[dict[str, Any]] = []
        language = None if args.lang == "auto" else LANGUAGE_NAMES.get(args.lang.lower(), args.lang)
        try:
            chunk_seconds, overlap_seconds = effective_chunk_settings(args)
            quiet_mode = getattr(args, "chunk_seconds_quiet", None) is not None
            if getattr(args, "internal_chunking", False) or (duration <= chunk_seconds and not quiet_mode):
                # Qwen's default native splitter permits 1200-second inputs,
                # which can allocate tens of GiB on XPU. In explicit internal
                # mode, lower that cap in qwen-asr itself; no FFmpeg windows or
                # toolkit-level chunk transcript stitching are introduced.
                if getattr(args, "internal_chunking", False):
                    native_cap = getattr(args, "qwen_internal_chunk_seconds", 120)
                    print(
                        f"[{device}] passing the full {duration:.1f}s file to Qwen3-ASR's internal "
                        f"low-energy chunker (max {native_cap}s/chunk, batch size 1)",
                        flush=True,
                    )
                    native_started = time.perf_counter()
                    with qwen_native_chunk_settings(native_cap):
                        with qwen_native_chunk_progress(model) as native_chunk_details:
                            results = model.transcribe(audio=str(audio), language=language)
                    native_elapsed = time.perf_counter() - native_started
                    print(
                        f"Qwen full-recording inference finished in {native_elapsed:.1f}s "
                        f"({duration / max(native_elapsed, 1e-9):.2f}x audio real-time; "
                        f"token cap {max_new_tokens}).", flush=True,
                    )
                else:
                    print(f"[{device}] using Qwen native long-audio transcription within the {chunk_seconds:g}s cap (batch size 1)", flush=True)
                    native_started = time.perf_counter()
                    results = model.transcribe(audio=str(audio), language=language)
                    native_elapsed = time.perf_counter() - native_started
                    print(
                        f"Qwen full-recording inference finished in {native_elapsed:.1f}s "
                        f"({duration / max(native_elapsed, 1e-9):.2f}x audio real-time; "
                        f"token cap {max_new_tokens}).", flush=True,
                    )
                item = results[0] if isinstance(results, (list, tuple)) and results else results
                text = str(getattr(item, "text", item)).strip() if item is not None else ""
                detected = getattr(item, "language", None) if item is not None else None
                if getattr(args, "internal_chunking", False) and native_chunk_details:
                    print(
                        f"Qwen final merged transcript: {len(text)} characters; "
                        f"full text saved to {out_path} (individual chunk transcripts shown above)",
                        flush=True,
                    )
                else:
                    print_transcript_preview(
                        text, label="Qwen transcript", output_path=out_path
                    )
                texts.append(text)
                record = {"index": 1, "start": 0.0, "end": duration,
                          "language": detected, "text": text, "device": device,
                          "max_new_tokens": max_new_tokens,
                          "inference_seconds": native_elapsed,
                          "chunking": ("qwen-native-internal-chunking" if getattr(args, "internal_chunking", False)
                                       else "qwen-native-long-form")}
                if getattr(args, "internal_chunking", False) and native_chunk_details:
                    record["native_chunks"] = native_chunk_details
                details.append(record)
            else:
                # Even long-form models receive bounded, balanced outer windows.
                chunks = make_external_chunks(audio, duration, temp_dir, args)
                quiet_mode = getattr(args, "chunk_seconds_quiet", None) is not None
                for idx, (chunk, start, end) in enumerate(chunks, 1):
                    print(
                        f"[{idx}/{len(chunks)}] {start:.1f}s–{end:.1f}s "
                        f"(Qwen max_new_tokens={max_new_tokens})", flush=True,
                    )
                    chunk_started = time.perf_counter()
                    results = model.transcribe(audio=str(chunk), language=language)
                    chunk_elapsed = time.perf_counter() - chunk_started
                    audio_seconds = max(0.0, end - start)
                    print(
                        f"Qwen chunk inference took {chunk_elapsed:.1f}s "
                        f"for {audio_seconds:.1f}s audio "
                        f"({audio_seconds / max(chunk_elapsed, 1e-9):.2f}x audio real-time).",
                        flush=True,
                    )
                    if not results:
                        text = ""
                        detected = None
                    else:
                        item = results[0] if isinstance(results, (list, tuple)) else results
                        text = str(getattr(item, "text", item)).strip()
                        detected = getattr(item, "language", None)
                    print_transcript_block(text, label=f"Chunk transcript [{idx}/{len(chunks)}]")
                    texts.append(text)
                    details.append({"index": idx, "start": start, "end": end,
                                    "language": detected, "text": text, "device": device,
                                    "max_new_tokens": max_new_tokens,
                                    "inference_seconds": chunk_elapsed,
                                    "chunking": ("ffmpeg-low-energy-no-overlap" if quiet_mode else "manual-overlap")})
        finally:
            del model
        return texts, details

    try:
        texts, details = decode_on(initial_device)
    except Exception as exc:
        # Intel XPU support is best-effort on Linux distributions outside the
        # platforms PyTorch validates directly. A failure can occur on the
        # first generated chunk, not only at model load, so retry the complete
        # job on CPU and discard any partial XPU transcript before merging.
        if not initial_device.startswith("xpu") or requested_device != "auto":
            raise
        error_summary = f"{type(exc).__name__}: {exc}"
        # Clear traceback references before the CPU reload; otherwise the failed
        # XPU model can remain live while the CPU copy is being allocated.
        try:
            exc.__traceback__ = None
        except Exception:
            pass
        print(f"WARN: Qwen XPU run failed ({error_summary}); restarting the complete transcription on CPU.",
              file=sys.stderr, flush=True)
        del exc
        try:
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
        except Exception:
            pass
        gc.collect()
        texts, details = decode_on("cpu")

    if getattr(args, "chunk_seconds_quiet", None) is not None:
        final = join_nonoverlapping_transcripts(texts)
    else:
        final = merge_overlapping_transcripts(texts, max_words=args.merge_max_words, max_chars=args.merge_max_chars)
    native_raw_chunks = [
        str(chunk.get("text", ""))
        for detail in details
        for chunk in detail.get("native_chunks", [])
    ]
    raw_text = "\n\n".join(native_raw_chunks) if native_raw_chunks else "\n\n".join(texts)
    write_results(out_path, final, raw_text, details)


def _canary_transcribe(model, audio_or_manifest, batch_size=1):
    """Run ordinary Canary transcription without optional confidence aggregation.

    Some NeMo/Canary combinations activate word-confidence aggregation but return
    ``None`` for token confidence, causing NeMo to crash inside decoding. We do
    not enable confidence capture and do not request hypothesis objects: normal
    decoding behavior and the configured beam size are left unchanged.
    """
    return model.transcribe(audio_or_manifest, batch_size=batch_size)


def estimate_nemo_unpacked_bytes(checkpoint_path: str | Path) -> int | None:
    """Estimate disk space required to extract a .nemo archive, without extracting it."""
    try:
        with tarfile.open(checkpoint_path, mode="r:*") as archive:
            return sum(member.size for member in archive.getmembers() if member.isfile())
    except (OSError, tarfile.TarError, EOFError):
        return None


def check_canary_v2_temp_space(checkpoint_path: str | Path,
                               temp_root: str | Path | None = None) -> tuple[int, int] | None:
    """Fail early with a useful message if the temp filesystem cannot unpack Canary-v2.

    NeMo restores the primary .nemo archive into a temporary directory. The
    archive's total uncompressed member sizes give a much better estimate than
    the downloaded file size. A little extra headroom is reserved for metadata,
    temporary audio chunks, and framework overhead.
    """
    unpacked = estimate_nemo_unpacked_bytes(checkpoint_path)
    if unpacked is None or unpacked <= 0:
        print(
            f"WARN: Could not estimate uncompressed size of Canary checkpoint {checkpoint_path}; "
            "NeMo will attempt loading without a disk-space preflight.",
            file=sys.stderr, flush=True,
        )
        return None
    temp_path = Path(temp_root or tempfile.gettempdir()).expanduser().resolve()
    try:
        free = shutil.disk_usage(temp_path).free
    except OSError as exc:
        print(f"WARN: Could not check free space on temporary filesystem {temp_path}: {exc}",
              file=sys.stderr, flush=True)
        return None
    reserve = max(512 * 1024**2, int(unpacked * 0.05))
    required = unpacked + reserve
    if free < required:
        gib = 1024**3
        raise RuntimeError(
            "Not enough temporary disk space to load Canary-1B-v2 without risking a failed extraction. "
            f"Temporary directory: {temp_path}; free: {free / gib:.2f} GiB; "
            f"estimated main-checkpoint extraction: {unpacked / gib:.2f} GiB; "
            f"recommended free space including headroom: {required / gib:.2f} GiB. "
            "The auxiliary timestamp model is disabled for plain transcription, but NeMo still must unpack "
            "the main checkpoint. Free disk space or set TMPDIR to a directory on a filesystem with enough "
            "space, e.g. `mkdir -p /path/with-space/tmp && TMPDIR=/path/with-space/tmp transcribe ...`. "
            "The downloaded model cache has not been deleted."
        )
    return unpacked, free


def find_cached_canary_checkpoint(model_id: str) -> Path | None:
    """Find a downloaded Canary .nemo archive without consulting any registry."""
    filename = model_id.rstrip("/").rsplit("/", 1)[-1] + ".nemo"
    roots: list[Path] = []
    env_cache = os.environ.get("NEMO_CACHE_DIR")
    if env_cache:
        roots.append(Path(env_cache).expanduser())
    roots.extend([
        MODELS_DIR / "nemo-cache",
        Path.home() / ".cache" / "torch" / "NeMo",
        MODELS_DIR / "hf-cache" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "transcribe-tools-qwen" / "hf-cache" / "hub",
    ])
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen or not root.exists():
            continue
        seen.add(key)
        if root.is_file() and root.name == filename:
            return root if root.is_file() else None
        try:
            for candidate in root.rglob(filename):
                # is_file follows Hugging Face's snapshot symlinks.
                if candidate.is_file() and not candidate.name.endswith((".download", ".partial")):
                    return candidate
        except OSError:
            continue
    return None


def load_canary_model(nemo_asr, model_id: str):
    """Load Canary from a cached .nemo path only; never contact model registries.

    Canary-v2's auxiliary timestamp CTC model is skipped because this toolkit
    emits transcripts, not forced-alignment timestamps. Direct local restore
    avoids NeMo's online `from_pretrained` registry lookup on inference runs.
    """
    checkpoint_path = find_cached_canary_checkpoint(model_id)
    if checkpoint_path is None:
        filename = model_id.rstrip("/").rsplit("/", 1)[-1] + ".nemo"
        raise FileNotFoundError(
            f"No cached Canary checkpoint {filename} was found. Inference is offline-only and will not "
            "download models. Connect to the network and rerun install.sh without --skip-model-download "
            "to prefetch this checkpoint; existing caches are preserved."
        )

    # The model archive identifies the concrete model class. Calling the
    # concrete class's restore_from(path) bypasses the online model catalog.
    model_cls = getattr(nemo_asr.models, "EncDecMultiTaskModel", None)
    if model_cls is None:
        model_cls = getattr(nemo_asr.models, "ASRModel", None)
    if model_cls is None or not callable(getattr(model_cls, "restore_from", None)):
        raise RuntimeError(
            "This NeMo installation does not expose a compatible local restore_from() loader; "
            "refusing to fall back to an online model-registry lookup."
        )

    is_v2 = checkpoint_path.name == "canary-1b-v2.nemo"
    if is_v2:
        check_canary_v2_temp_space(checkpoint_path)
        method_name = "_EncDecMultiTaskModel__restore_timestamps_asr_model"
        original_method = getattr(model_cls, method_name, None)
        if not callable(original_method):
            raise RuntimeError(
                "This NeMo version does not expose the expected Canary timestamp-restore hook; "
                "refusing to load Canary-1B-v2 because it may unpack a large auxiliary model and fill the disk. "
                "Keep the existing model cache; inspect the installed NeMo version before changing packages."
            )

        def _skip_optional_timestamp_model(_self):
            return None

        print(
            "Canary-1B-v2: loading cached .nemo file directly; skipping auxiliary timestamp-alignment model.",
            flush=True,
        )
        setattr(model_cls, method_name, _skip_optional_timestamp_model)
        try:
            return model_cls.restore_from(str(checkpoint_path), map_location="cpu")
        finally:
            setattr(model_cls, method_name, original_method)

    print(f"Loading cached Canary checkpoint directly from {checkpoint_path}; no network access.", flush=True)
    return model_cls.restore_from(str(checkpoint_path), map_location="cpu")


def canary_model_id_for_args(args: argparse.Namespace) -> str:
    """Resolve a selected Canary alias, honoring an explicit CANARY_MODEL override."""
    default_id = CANARY_MODEL_IDS.get(args.model, CANARY_MODEL_IDS["canary"])
    return os.environ.get("CANARY_MODEL") or default_id

def configure_canary_decoder(model, beam_size: int = 5, model_name: str = "Canary") -> bool:
    """Explicitly set NeMo Canary decoder beam width for reproducible defaults.

    NeMo model cards commonly ship with greedy beam_size=1, while reported
    accuracy-oriented Canary-1B results use width 5. We use 5 across the three
    Canary checkpoints by default and permit a CLI override. This affects decoder
    search/latency, not the outer audio-window memory cap.
    """
    try:
        decoding_cfg = model.cfg.decoding
        decoding_cfg.beam.beam_size = int(beam_size)
        model.change_decoding_strategy(decoding_cfg)
        print(
            f"{model_name} decoder beam size: {int(beam_size)} "
            f"({'greedy decoding' if int(beam_size) == 1 else 'beam search; slower than greedy'}).",
            flush=True,
        )
        return True
    except Exception as exc:
        # Fail clearly rather than silently claiming the requested accuracy
        # default was used when NeMo could have kept the checkpoint's greedy=1.
        raise RuntimeError(
            f"Could not set {model_name} decoder beam size to {beam_size}. "
            "The installed NeMo/checkpoint decoding API may differ; refusing to run with an unknown beam setting."
        ) from exc


def configure_canary_v2_decoder(model, beam_size: int = 5) -> bool:
    """Backward-compatible alias for the generic Canary decoder configurator."""
    return configure_canary_decoder(model, beam_size, model_name="Canary-1B-v2")


def _canary_v2_full_audio_manifest(audio: Path, duration: float, language: str,
                                  temp_dir: Path) -> Path:
    """Create a one-entry manifest for a Canary-v2 input that already fits the configured cap."""
    manifest = temp_dir / "canary_v2_longform.jsonl"
    entry = {
        "audio_filepath": str(audio),
        "duration": float(duration),
        "source_lang": language,
        "target_lang": language,
        "pnc": "yes",
    }
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return manifest


def worker_canary(args: argparse.Namespace, audio: Path, duration: float,
                  temp_dir: Path, out_path: Path) -> None:
    enforce_offline_model_loading()
    validate_canary_language(args)

    # `canary` remains an alias for Flash; explicit model names select checkpoints.
    model_id = canary_model_id_for_args(args)
    is_v2 = model_id.rstrip("/").rsplit("/", 1)[-1] == "canary-1b-v2"
    if getattr(args, "internal_chunking", False) and not is_v2:
        raise ValueError(
            "--internal-chunking is not supported for legacy Canary-1B/Flash. "
            "Use --model canary-1b-v2, or omit --internal-chunking to use safe short external windows."
        )
    import nemo.collections.asr as nemo_asr

    print(f"Loading {model_id} from the local NeMo cache only; inference never downloads models.", flush=True)
    model = load_canary_model(nemo_asr, model_id)
    model = model.cpu()
    model.eval()
    chunk_seconds, overlap_seconds = effective_chunk_settings(args)
    canary_beam_size = int(getattr(args, "canary_beam_size", 5) or 5)
    configure_canary_decoder(
        model, canary_beam_size,
        model_name="Canary-1B-v2" if is_v2 else model_id.rstrip("/").rsplit("/", 1)[-1],
    )
    texts: list[str] = []
    details: list[dict[str, Any]] = []
    lang = "en" if args.lang == "auto" else args.lang.lower()
    internal_v2 = is_v2 and bool(getattr(args, "internal_chunking", False))
    try:
        if internal_v2:
            cap_seconds = getattr(args, "canary_internal_chunk_seconds", 120)
            # NeMo's native overlapping merge is gated on timestamps_asr_model.
            # This toolkit deliberately skips that auxiliary checkpoint, so
            # bypassing the guard would feed the complete hour to one encoder
            # call (the user's log showed "Chunking is disabled"). Instead,
            # split the normalized WAV in-process at low-energy boundaries,
            # with exact no-overlap coverage, then concatenate text in order.
            chunks = make_low_energy_nonoverlap_chunks(
                audio, duration, temp_dir, cap_seconds, prefix="canary_v2_low_energy"
            )
            print(
                f"Canary-v2 low-energy internal mode: {duration:.1f}s split into {len(chunks)} "
                f"non-overlapping chunks (max {cap_seconds}s, quietest 100-ms window within +/-5s; "
                "no FFmpeg chunking or fuzzy overlap merge)",
                flush=True,
            )
            for idx, (chunk, start, end) in enumerate(chunks, 1):
                print(f"[{idx}/{len(chunks)}] {start:.1f}s–{end:.1f}s", flush=True)
                if lang == "en":
                    results = _canary_transcribe(model, [str(chunk)], batch_size=1)
                else:
                    manifest = temp_dir / f"canary_v2_low_energy_{idx:04d}.jsonl"
                    entry = {
                        "audio_filepath": str(chunk), "duration": max(0.01, end - start),
                        "source_lang": lang, "target_lang": lang, "pnc": "yes",
                    }
                    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
                    results = _canary_transcribe(model, str(manifest), batch_size=1)
                item = results[0] if isinstance(results, (list, tuple)) and results else results
                text = str(getattr(item, "text", item)).strip() if item is not None else ""
                print_transcript_block(text, label=f"Chunk transcript [{idx}/{len(chunks)}]")
                texts.append(text)
                detail = {
                    "index": idx, "start": start, "end": end, "language": lang, "text": text,
                    "chunking": "canary-v2-low-energy-no-overlap", "beam_size": canary_beam_size,
                    "max_chunk_seconds": cap_seconds,
                }
                details.append(detail)
        elif is_v2 and duration <= chunk_seconds and getattr(args, "chunk_seconds_quiet", None) is None:
            # Preserve the old small-file path. Since the optional timestamp
            # model is intentionally absent, do not claim this invokes NeMo's
            # dynamic long-form chunker; the input is simply under the cap.
            manifest = _canary_v2_full_audio_manifest(audio, duration, lang, temp_dir)
            print(f"Using a single Canary-v2 input within the {chunk_seconds:g}s cap ({duration:.1f}s audio)", flush=True)
            results = _canary_transcribe(model, str(manifest), batch_size=1)
            item = results[0] if isinstance(results, (list, tuple)) and results else results
            text = str(getattr(item, "text", item)).strip() if item is not None else ""
            print_transcript_block(text, label="Canary-v2 transcript")
            texts.append(text)
            detail = {"index": 1, "start": 0.0, "end": duration, "language": lang,
                      "text": text, "chunking": "single-input-under-cap",
                      "batch_size": 1, "beam_size": canary_beam_size}
            details.append(detail)
        else:
            # Legacy Canary remains below 40s; normal mode for longer recordings
            # retains its existing balanced FFmpeg windows and overlap merger.
            chunks = make_external_chunks(audio, duration, temp_dir, args)
            quiet_mode = getattr(args, "chunk_seconds_quiet", None) is not None
            for idx, (chunk, start, end) in enumerate(chunks, 1):
                print(f"[{idx}/{len(chunks)}] {start:.1f}s–{end:.1f}s", flush=True)
                if lang == "en":
                    results = _canary_transcribe(model, [str(chunk)], batch_size=1)
                else:
                    manifest = temp_dir / f"canary_{idx:04d}.jsonl"
                    entry = {
                        "audio_filepath": str(chunk), "duration": max(0.01, end-start),
                        "source_lang": lang, "target_lang": lang, "pnc": "yes",
                    }
                    if not is_v2:
                        entry["taskname"] = "asr"
                        entry["answer"] = "na"
                    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
                    results = _canary_transcribe(model, str(manifest), batch_size=1)
                item = results[0] if isinstance(results, (list, tuple)) and results else results
                text = str(getattr(item, "text", item)).strip() if item is not None else ""
                print_transcript_block(text, label=f"Chunk transcript [{idx}/{len(chunks)}]")
                texts.append(text)
                detail = {"index": idx, "start": start, "end": end, "language": lang,
                          "text": text,
                          "chunking": ("ffmpeg-low-energy-no-overlap" if quiet_mode else "manual-overlap"),
                          "beam_size": canary_beam_size}
                details.append(detail)
    finally:
        del model
    if internal_v2 or getattr(args, "chunk_seconds_quiet", None) is not None:
        final = join_nonoverlapping_transcripts(texts)
    elif len(texts) == 1:
        final = texts[0]
    else:
        final = merge_overlapping_transcripts(texts, max_words=args.merge_max_words, max_chars=args.merge_max_chars)
    write_results(out_path, final, "\n\n".join(texts), details)


def worker_faster_whisper(args: argparse.Namespace, audio: Path, duration: float,
                          temp_dir: Path, out_path: Path) -> None:
    enforce_offline_model_loading()
    from faster_whisper import WhisperModel

    lang = None if args.lang == "auto" else args.lang.lower()
    # Match the public CLI's language-aware default when called without a size.
    model_name = args.whisper_size or ("large-v3" if str(args.lang).lower() == "bg" else "large-v3-turbo")
    if args.model == "distil-whisper-large-v3.5":
        repo_id = DISTIL_WHISPER_REPO_ID
        cached_or_repo = resolve_hf_snapshot(repo_id)
        model_source = cached_or_repo if cached_or_repo != repo_id else model_name
    else:
        repo_id = faster_whisper_repo_id(model_name)
        model_source = resolve_faster_whisper_model_source(model_name)
        cached_or_repo = model_source if model_source != model_name else repo_id
    display_model = "Distil-Whisper Large-v3.5 CT2" if args.model == "distil-whisper-large-v3.5" else model_name
    if args.model == "distil-whisper-large-v3.5":
        if cached_or_repo == repo_id:
            raise FileNotFoundError(
                "Distil-Whisper Large-v3.5 weights are not cached locally or the cached snapshot is incomplete. "
                "Run ./install.sh while online without --skip-model-download, then check "
                "~/.local/share/speech-toolkit/state/optional-asr-install-status.txt. "
                "Inference itself remains offline; the CTranslate2 model.bin should be at least 1.4 GB."
            )
        distil_path = Path(cached_or_repo)
        if not is_valid_distil_whisper_model_dir(distil_path):
            try:
                observed = (distil_path / "model.bin").stat().st_size
            except OSError:
                observed = 0
            raise FileNotFoundError(
                f"Distil-Whisper cache is incomplete at {distil_path}: model.bin is {observed:,} bytes; "
                f"expected at least {DISTIL_WHISPER_MIN_MODEL_BYTES:,}. Run ./install.sh while online to resume the download."
            )
    print(f"Loading faster-whisper {display_model} on CPU/int8 (threads={args.threads})", flush=True)
    model = WhisperModel(model_source, device="cpu", compute_type="int8", cpu_threads=args.threads)
    beam_size = args.beam_size if args.beam_size is not None else default_beam_size_for("faster-whisper", args.lang)
    quiet_mode = getattr(args, "chunk_seconds_quiet", None) is not None
    # Quiet-boundary chunks are independently decoded and deliberately have no
    # overlap. For these chunks, VAD should remove long non-speech spans before
    # Whisper can hallucinate boilerplate over them, and previous-window context
    # is more likely to create repetition loops than to help continuity. Both
    # behaviors remain overrideable from the CLI.
    use_vad = bool(args.vad) or (quiet_mode and not bool(getattr(args, "no_vad", False)))
    use_previous_text = (
        not args.no_previous_text_context
        and (not quiet_mode or bool(getattr(args, "previous_text_context", False)))
    )
    kwargs: dict[str, Any] = {
        "language": lang,
        "beam_size": beam_size,
        "best_of": 5,
        "vad_filter": use_vad,
        "condition_on_previous_text": use_previous_text,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.4,
        # Add post-decode word alignment/probabilities for the review UI. This
        # does not alter beam search or the generated transcript tokens; it
        # enriches the JSON sidecar with timing/confidence metadata.
        "word_timestamps": True,
    }
    if use_vad:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 1800, "speech_pad_ms": 250, "threshold": 0.35}
    if quiet_mode:
        print(
            "faster-whisper quiet-chunk safeguards: VAD "
            f"{'enabled' if use_vad else 'disabled'}; previous-window text context "
            f"{'enabled' if use_previous_text else 'disabled'}.",
            flush=True,
        )

    chunk_seconds, overlap_seconds = effective_chunk_settings(args)
    if getattr(args, "internal_chunking", False):
        # faster-whisper's Whisper encoder/decoder already processes a full
        # audio file via its internal sliding windows and yields segments lazily.
        chunks = [(audio, 0.0, duration)]
        print(f"Using faster-whisper internal windowing for the full {duration:.1f}s file", flush=True)
    else:
        chunks = make_external_chunks(audio, duration, temp_dir, args)
    chunk_texts: list[str] = []
    raw_chunks: list[str] = []
    details: list[dict[str, Any]] = []
    # A readable, timestamped companion log for faster-whisper. This mirrors
    # the per-chunk and per-segment lines shown on stdout, while intentionally
    # omitting language-detection lines as requested. It is separate from the
    # plain final transcript and JSON metadata sidecar.
    timestamp_log_lines: list[str] = []
    for index, (chunk, chunk_start, chunk_end) in enumerate(chunks, 1):
        chunk_header = f"[{index}/{len(chunks)}] {chunk_start:.1f}s–{chunk_end:.1f}s"
        print(chunk_header, flush=True)
        timestamp_log_lines.append(chunk_header)
        segments, info = model.transcribe(str(chunk), **kwargs)
        print(f"Detected language: {info.language} (p={info.language_probability:.2f})", flush=True)
        lines: list[str] = []
        chunk_duration = max(0.0, float(chunk_end) - float(chunk_start))
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue
            # Whisper decodes in 30-second windows and can emit timestamps that
            # extend into padded time after the end of a short final window.
            # Never report times beyond the actual FFmpeg chunk or accept a
            # segment whose start lies wholly outside that chunk.
            raw_start = max(0.0, float(segment.start))
            raw_end = max(raw_start, float(segment.end))
            if raw_start >= chunk_duration:
                warning_line = (
                    f"Ignoring out-of-range Whisper segment at {raw_start:.1f}s "
                    f"(chunk duration {chunk_duration:.1f}s): {text}"
                )
                print(warning_line, flush=True)
                timestamp_log_lines.append(warning_line)
                continue
            local_start = min(raw_start, chunk_duration)
            local_end = min(max(raw_end, local_start), chunk_duration)
            abs_start = chunk_start + local_start
            abs_end = chunk_start + local_end
            lines.append(text)
            word_rows: list[dict[str, Any]] = []
            for word in (getattr(segment, "words", None) or []):
                word_text = str(getattr(word, "word", "") or "").strip()
                if not word_text:
                    continue
                try:
                    word_start = max(0.0, float(word.start))
                    word_end = max(word_start, float(word.end))
                except (AttributeError, TypeError, ValueError):
                    continue
                if word_start >= chunk_duration:
                    continue
                word_start = min(word_start, chunk_duration)
                word_end = min(max(word_end, word_start), chunk_duration)
                word_rows.append({
                    "word": word_text,
                    "start": chunk_start + word_start,
                    "end": chunk_start + word_end,
                    "probability": getattr(word, "probability", None),
                })
            details.append({"chunk": index, "start": abs_start,
                            "end": abs_end, "text": text,
                            "words": word_rows,
                            "language": info.language, "language_probability": info.language_probability,
                            "no_speech_probability": getattr(segment, "no_speech_prob", None),
                            "average_log_probability": getattr(segment, "avg_logprob", None),
                            "compression_ratio": getattr(segment, "compression_ratio", None),
                            "chunking": ("ffmpeg-low-energy-no-overlap" if quiet_mode else
                                         ("faster-whisper-internal-windowing" if getattr(args, "internal_chunking", False) else "manual-overlap"))})
            segment_line = f"[{abs_start:.1f}s–{abs_end:.1f}s] {text}"
            print(segment_line, flush=True)
            timestamp_log_lines.append(segment_line)
        # A fresh list is built for every chunk. Empty output intentionally stays
        # empty; it must never inherit text from the preceding chunk.
        chunk_text = " ".join(lines).strip()
        if not chunk_text:
            empty_line = "No speech segments emitted for this chunk."
            print(empty_line, flush=True)
            timestamp_log_lines.append(empty_line)
        chunk_texts.append(chunk_text)
        raw_chunks.append(chunk_text)
    if len(chunk_texts) == 1:
        final_text = chunk_texts[0]
    elif quiet_mode:
        final_text = join_nonoverlapping_transcripts(chunk_texts)
    else:
        final_text = merge_overlapping_transcripts(
            chunk_texts, max_words=args.merge_max_words, max_chars=args.merge_max_chars
        )
    write_results(out_path, final_text, "\n\n".join(raw_chunks), details)
    write_whisper_subtitles(out_path, details)
    timestamp_log_path = out_path.with_name(out_path.stem + ".segments.txt")
    timestamp_log_tmp = timestamp_log_path.with_suffix(timestamp_log_path.suffix + ".tmp")
    timestamp_log_tmp.write_text("\n".join(timestamp_log_lines).rstrip() + "\n", encoding="utf-8")
    os.replace(timestamp_log_tmp, timestamp_log_path)
    print(f"Timestamped faster-whisper segment log: {timestamp_log_path}", flush=True)


def seamless_lang_code(language: str) -> str:
    """Resolve the CLI's two-letter language code to SeamlessM4T's tag."""
    code = language.strip().lower()
    if code == "auto":
        raise ValueError(
            "SeamlessM4T v2 ASR needs an explicit output language. Pass --lang bg for Bulgarian "
            "or --lang en for English; automatic source-language detection is not enabled by this wrapper."
        )
    if code in SEAMLESS_LANGUAGE_CODES:
        return SEAMLESS_LANGUAGE_CODES[code]
    if code in SEAMLESS_LANGUAGE_CODES.values():
        return code
    raise ValueError(
        f"Language {language!r} has no configured SeamlessM4T language tag. "
        "Use a supported two-letter code such as bg, en, de, fr, es, pt, ru, or zh."
    )


def worker_transformers_asr(args: argparse.Namespace, audio: Path, duration: float,
                             temp_dir: Path, out_path: Path, engine: str) -> None:
    """Run SeamlessM4T v2 or IBM Granite Speech in the CPU-only Transformers venv.

    Both paths share the same outer chunk planner and write the standard text,
    raw-hypothesis, and chunk-metadata artifacts. They do not use the Qwen XPU
    environment and intentionally do not probe or install CUDA/NVIDIA software.
    """
    enforce_offline_model_loading()
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor

    model_id = TRANSFORMERS_MODEL_IDS[args.model]
    # Resolve a cached snapshot across the toolkit and legacy cache roots first.
    # If resolution falls back to the repo ID, local_files_only=True plus the
    # offline environment prevents any Hub metadata probes or download retries.
    model_source = resolve_hf_snapshot(model_id)
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch may already have initialized its inter-op pool in an imported
        # dependency; the configured intra-op limit remains in force.
        pass

    tokenizer = None
    max_new_tokens = effective_max_new_tokens(args, engine)
    if engine == "seamless-m4t":
        lang_tag = seamless_lang_code(args.lang)
        print(f"Loading {model_id} on CPU for ASR output language {lang_tag}; "
              f"max_new_tokens={max_new_tokens}; this model is CC-BY-NC-4.0.", flush=True)
        from transformers import SeamlessM4Tv2ForSpeechToText

        processor = AutoProcessor.from_pretrained(model_source, local_files_only=True)
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_source, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        ).to("cpu")
        model.eval()
    elif engine == "granite-speech":
        if args.lang != "auto" and args.lang.lower() not in GRANITE_SPEECH_LANGUAGES:
            raise ValueError(
                "IBM Granite Speech 3.3 supports ASR in English, French, German, Spanish, and Portuguese; "
                f"it does not document support for {args.lang!r}. Use Whisper or Canary-v2 for Bulgarian."
            )
        from transformers import AutoModelForSpeechSeq2Seq

        print(
            f"Loading {model_id} on CPU (bfloat16; beam_size={args.granite_beam_size}; "
            f"max_new_tokens={max_new_tokens}). "
            "The 8B checkpoint can use substantial physical RAM.", flush=True,
        )
        processor = AutoProcessor.from_pretrained(model_source, local_files_only=True)
        tokenizer = processor.tokenizer
        # IBM's official Transformers examples use bfloat16 for both sizes.
        # Keep it on CPU: device_map='cpu' does not activate CUDA or XPU.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_source, local_files_only=True, device_map="cpu", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()
    else:
        raise ValueError(f"Unsupported Transformers ASR engine: {engine}")

    internal_mode = bool(getattr(args, "internal_chunking", False))
    if internal_mode:
        chunks = select_transformers_asr_chunks(audio, duration, temp_dir, args, engine)
        strategy = transformers_internal_chunking_label(engine)
        if engine == "seamless-m4t":
            print(
                f"Internal mode: passing the full {duration:.1f}s recording in one call; "
                "SeamlessM4T v2 will apply its built-in speech-encoder chunk-attention mask. "
                "This does not create separately decoded text segments.",
                flush=True,
            )
        else:
            print(
                f"WARN: internal mode passes the full {duration:.1f}s recording to Granite in one call. "
                "Its speech encoder uses four-second block attention, but Transformers does not expose "
                "a native transcript-level chunk-and-stitch API; long recordings can require substantial RAM "
                "and IBM recommends short (about one-minute) units for consistent accuracy.",
                file=sys.stderr, flush=True,
            )
    else:
        effective_chunk_settings(args)  # validate effective manual-window settings before splitting
        chunks = select_transformers_asr_chunks(audio, duration, temp_dir, args, engine)
        strategy = None
    quiet_mode = getattr(args, "chunk_seconds_quiet", None) is not None
    if quiet_mode and args.model == "seamless-m4t-v2":
        actual_long_chunks = [end - start for _, start, end in chunks if end - start > 60.0]
        if actual_long_chunks:
            print(
                "WARN: quiet-boundary chunking produced SeamlessM4T v2 segment(s) longer than 60 seconds "
                f"(longest {max(actual_long_chunks):.1f}s). Long segments are experimental and may use more RAM; "
                "use --chunk-seconds 50 for the standard 50-second overlapping mode.",
                file=sys.stderr, flush=True,
            )
    chunk_texts: list[str] = []
    details: list[dict[str, Any]] = []
    for index, (chunk, chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"[{index}/{len(chunks)}] {chunk_start:.1f}s–{chunk_end:.1f}s", flush=True)
        waveform, sample_rate = sf.read(str(chunk), dtype="float32", always_2d=False)
        if sample_rate != 16000:
            raise RuntimeError(f"Prepared audio must be 16 kHz; got {sample_rate} Hz for {chunk}")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        if waveform.size == 0:
            raise RuntimeError(f"Prepared audio chunk is empty: {chunk}")
        with torch.inference_mode():
            if engine == "seamless-m4t":
                # Match the documented SeamlessM4Tv2Processor call: pass a
                # mono float waveform through `audio=` (the processor creates
                # its own feature tensors). Avoid assuming the processor accepts
                # the plural `audios=` keyword across Transformers versions.
                model_inputs = processor(
                    audio=waveform, sampling_rate=16000, return_tensors="pt",
                )
                model_inputs = model_inputs.to("cpu")
                output_ids = model.generate(
                    **model_inputs, tgt_lang=lang_tag,
                    max_new_tokens=max_new_tokens,
                )
                text = processor.decode(output_ids[0].tolist(), skip_special_tokens=True).strip()
            else:
                wav_tensor = torch.from_numpy(waveform).unsqueeze(0)
                user_prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
                chat = [{
                    "role": "system",
                    "content": "Transcribe spoken audio faithfully. Preserve the words as spoken and use appropriate punctuation and capitalization.",
                }, {"role": "user", "content": user_prompt}]
                prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True,
                )
                model_inputs = processor(
                    prompt, wav_tensor, device="cpu", return_tensors="pt",
                ).to("cpu")
                model_outputs = model.generate(
                    **model_inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, num_beams=args.granite_beam_size,
                )
                input_tokens = model_inputs["input_ids"].shape[-1]
                new_tokens = model_outputs[0, input_tokens:].unsqueeze(0)
                text = tokenizer.batch_decode(
                    new_tokens, add_special_tokens=False, skip_special_tokens=True,
                )[0].strip()

        chunk_texts.append(text)
        detail = {
            "engine": engine, "model": args.model, "chunk": index,
            "start": chunk_start, "end": chunk_end, "text": text,
            "language": args.lang,
            "chunking": (strategy if internal_mode else
                         ("ffmpeg-low-energy-no-overlap" if quiet_mode else "manual-overlap")),
        }
        details.append(detail)
        print_transcript_block(text, label=f"{args.model} transcript [{index}/{len(chunks)}]")

    if len(chunk_texts) == 1:
        final_text = chunk_texts[0]
    elif quiet_mode:
        final_text = join_nonoverlapping_transcripts(chunk_texts)
    else:
        final_text = merge_overlapping_transcripts(
            chunk_texts, max_words=args.merge_max_words, max_chars=args.merge_max_chars,
        )
    write_results(out_path, final_text, "\n\n".join(chunk_texts), details)
    del model



def worker_open_asr(args: argparse.Namespace, audio: Path, duration: float,
                    temp_dir: Path, out_path: Path, engine: str) -> None:
    """Run optional open-weight ASR models using dedicated CPU-only environments.

    All model and tokenizer loading is local-files-only. Hugging Face model code
    is loaded only from the cached snapshot prepared by install.sh; no audio or
    transcript is sent to an API. Each model has its own venv so incompatibility
    in a new Transformers release cannot break the existing backends.
    """
    enforce_offline_model_loading()
    import gc
    import numpy as np
    import soundfile as sf
    import torch

    model_id = TRANSFORMERS_MODEL_IDS[args.model]
    model_source = resolve_hf_snapshot(model_id)
    if model_source == model_id:
        repo_cache = "models--" + model_id.replace("/", "--")
        cache_roots = (
            MODELS_DIR / "hf-cache" / "hub",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "transcribe-tools-qwen" / "hf-cache" / "hub",
        )
        expected = "; ".join(str(root / repo_cache / "snapshots") for root in cache_roots)
        raise FileNotFoundError(
            f"{args.model} weights could not be resolved from the offline Hugging Face caches. "
            f"Expected repository snapshot folder: {repo_cache}. Checked: {expected}. "
            "If install.sh reported WEIGHTS READY, rerun `download-optional-asr-models` after this "
            "toolkit update to reconcile the cache, and retain its printed snapshot path. "
            "Transcription itself remains offline-only and never downloads weights or contacts an inference service."
        )
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("This ASR environment must use CPU-only PyTorch; refusing to load a CUDA runtime.")
    if "+xpu" in str(torch.__version__).lower() or (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("This ASR environment must use CPU-only PyTorch; refusing to activate Intel XPU.")

    native_longform_models = {"cohere-transcribe-2b", "moss-transcribe-diarize"}
    if args.internal_chunking and args.model not in native_longform_models:
        raise ValueError(
            f"--internal-chunking is only supported for {', '.join(sorted(native_longform_models))}; "
            f"{args.model} should use external chunks."
        )
    if args.internal_chunking:
        chunks = [(audio, 0.0, float(duration))]
        print(
            f"Native long-form path: passing the entire {duration:.1f}s recording to {args.model}; "
            "this may use more RAM than quiet-boundary chunking.", flush=True,
        )
    else:
        chunks = make_external_chunks(audio, duration, temp_dir, args)

    # Most of these models use autoregressive text decoding, but their model
    # implementations and audio frontends differ. Keep the calls explicit.
    model = None
    processor = None
    tokenizer = None
    helper_funcs: dict[str, Any] = {}
    if engine == "granite5-ctc":
        from transformers import AutoModelForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_source, local_files_only=True)
        model = AutoModelForCTC.from_pretrained(
            model_source, local_files_only=True, torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cpu").eval()
        print(f"Loaded Granite Speech 5.0 TurboCTC on CPU ({model_id}).", flush=True)
    elif engine == "cohere-asr":
        try:
            from transformers import CohereAsrForConditionalGeneration
            model_cls = CohereAsrForConditionalGeneration
        except ImportError:
            # Some published Transformers versions expose the architecture via
            # the generic auto class; the repository's local code supplies its
            # implementation when trust_remote_code is enabled.
            from transformers import AutoModelForSpeechSeq2Seq
            model_cls = AutoModelForSpeechSeq2Seq
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
        )
        model = model_cls.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True,
        ).to("cpu").eval()
        print(f"Loaded Cohere Transcribe 2B on CPU ({model_id}); language={args.lang}.", flush=True)
    elif engine == "moss-diarize":
        # The official MOSS-Transcribe-Diarize quickstart uses
        # AutoModelForCausalLM. A generic multimodal auto-class can select an
        # incompatible attention configuration in recent Transformers builds
        # with an incompatible attention-head count (1024 / 6).
        from transformers import AutoModelForCausalLM as MossModelClass, AutoProcessor
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages, generate_transcription, resolve_device,
        )
        helper_funcs = {
            "build_transcription_messages": build_transcription_messages,
            "generate_transcription": generate_transcription,
            "resolve_device": resolve_device,
        }
        model = MossModelClass.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).to("cpu").eval()
        processor = AutoProcessor.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
        )
        print(f"Loaded MOSS-Transcribe-Diarize 0.9B on CPU ({model_id}).", flush=True)
    elif engine == "moss-preview":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        tokenizer = AutoTokenizer.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
        )
        processor_cls = get_class_from_dynamic_module(
            "processing_Moss.MossProcessor", model_source, local_files_only=True,
        )
        mel_config_cls = get_class_from_dynamic_module(
            "processing_Moss.MelConfig", model_source, local_files_only=True,
        )
        mel_config = mel_config_cls(mel_sr=16000, mel_dim=128, mel_n_fft=400, mel_hop_length=160)
        processor = processor_cls(tokenizer, config=mel_config, enable_time_marker=False)
        template_path = Path(model_source) / "chat_template_default.py"
        if not template_path.is_file():
            raise FileNotFoundError(f"MOSS local chat template is missing: {template_path}; rerun install.sh to resume its model download.")
        processor.load_template(str(template_path))
        model = AutoModelForCausalLM.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).to("cpu").eval()
        print(f"Loaded MOSS-Transcribe-preview-2B on CPU ({model_id}); English-only.", flush=True)
    elif engine == "ark-asr":
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        processor = AutoProcessor.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).to("cpu").eval()
        print(f"Loaded ARK-ASR-3B on CPU ({model_id}); official audio prompt limit is 30s.", flush=True)
    else:
        raise ValueError(f"Unsupported open ASR engine: {engine}")

    if args.model == "cohere-transcribe-2b":
        max_new_tokens = int(args.max_new_tokens or 512)
    elif args.model == "moss-transcribe-diarize":
        max_new_tokens = int(args.max_new_tokens or (2048 if args.internal_chunking else 1024))
    elif args.model == "moss-transcribe-preview-2b":
        max_new_tokens = int(args.max_new_tokens or 512)
    elif args.model == "ark-asr-3b":
        max_new_tokens = int(args.max_new_tokens or 256)
    else:
        max_new_tokens = int(args.max_new_tokens or 4096)

    chunk_texts: list[str] = []
    details: list[dict[str, Any]] = []
    try:
        for index, (chunk, chunk_start, chunk_end) in enumerate(chunks, 1):
            print(f"[{index}/{len(chunks)}] {chunk_start:.1f}s–{chunk_end:.1f}s", flush=True)
            if engine == "moss-diarize":
                # Official helpers build the model's chat/audio messages from a
                # local file path. The audio never leaves this process/machine.
                moss_prompt = moss_diarize_prompt(args.lang)
                if moss_prompt is None:
                    messages = helper_funcs["build_transcription_messages"](str(chunk))
                else:
                    messages = helper_funcs["build_transcription_messages"](str(chunk), prompt=moss_prompt)
                device = helper_funcs["resolve_device"]("cpu")
                with torch.inference_mode():
                    result = helper_funcs["generate_transcription"](
                        model, processor, messages, max_new_tokens=max_new_tokens,
                        do_sample=False, device=device, dtype=torch.float32,
                    )
                text = str(result.get("text", "") if isinstance(result, dict) else result).strip()
            else:
                waveform, sample_rate = sf.read(str(chunk), dtype="float32", always_2d=False)
                if sample_rate != 16000:
                    raise RuntimeError(f"Prepared audio must be 16 kHz; got {sample_rate} Hz for {chunk}")
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)
                waveform = np.ascontiguousarray(waveform, dtype=np.float32)
                if waveform.size == 0:
                    raise RuntimeError(f"Audio chunk is empty: {chunk}")

                with torch.inference_mode():
                    if engine == "granite5-ctc":
                        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
                        inputs = inputs.to("cpu")
                        output_ids = model.generate(**inputs)
                        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    elif engine == "cohere-asr":
                        lang = "en" if args.lang == "auto" else args.lang.lower()
                        inputs = processor(
                            waveform, sampling_rate=16000, return_tensors="pt", language=lang,
                            punctuation=True,
                        )
                        audio_chunk_index = inputs.get("audio_chunk_index")
                        inputs = inputs.to("cpu", dtype=model.dtype)
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                        try:
                            decoded = processor.decode(
                                outputs, skip_special_tokens=True,
                                audio_chunk_index=audio_chunk_index, language=lang,
                            )
                        except TypeError:
                            decoded = processor.decode(outputs, skip_special_tokens=True)
                        text = str(decoded[0] if isinstance(decoded, (list, tuple)) else decoded).strip()
                    elif engine == "moss-preview":
                        inputs = processor(audio=waveform, return_tensors="pt").to("cpu")
                        if "audio_data" in inputs:
                            inputs["audio_data"] = inputs["audio_data"].to(dtype=model.dtype)
                        output_ids = model.generate(
                            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            num_beams=1, use_cache=True,
                            eos_token_id=[processor.end_token_id],
                        )
                        new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                        text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
                    elif engine == "ark-asr":
                        if len(waveform) > 30 * 16000:
                            raise ValueError(
                                f"ARK-ASR audio window is {len(waveform) / 16000:.1f}s; its official processor caps input at 30s. "
                                "Use --chunk-seconds-quiet 25 or a shorter --chunk-seconds window."
                            )
                        conversation = [{
                            "role": "user", "content": [
                                {"type": "audio", "path": str(chunk)},
                                {"type": "text", "text": "Please transcribe this audio."},
                            ],
                        }]
                        inputs = processor.apply_chat_template(
                            conversation, add_generation_prompt=True, return_tensors="pt",
                            sampling_rate=16000, audio_padding="longest", text_kwargs={"padding": "longest"},
                            audio_max_length=30 * 16000,
                        ).to("cpu")
                        if "audios" in inputs:
                            inputs["audios"] = inputs["audios"].to(dtype=model.dtype)
                        eos_ids = tokenizer.eos_token_id
                        keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
                        bad_ids = set(tokenizer.all_special_ids) - keep_ids
                        bad_ids.update(
                            token_id for token, token_id in tokenizer.get_added_vocab().items()
                            if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
                        )
                        outputs = model.generate(
                            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                            bad_words_ids=[[token_id] for token_id in sorted(bad_ids)],
                        )
                        new_ids = outputs[:, inputs.input_ids.shape[1]:]
                        text = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
                    else:
                        raise AssertionError(engine)

            print_transcript_block(text, label=f"Chunk transcript [{index}/{len(chunks)}]")
            chunk_texts.append(text)
            details.append({
                "engine": engine, "model": args.model, "chunk": index,
                "start": chunk_start, "end": chunk_end, "text": text,
                "language": args.lang,
                "chunking": ("native-full-file" if args.internal_chunking else
                             ("ffmpeg-low-energy-no-overlap" if getattr(args, "chunk_seconds_quiet", None) is not None else "manual-overlap")),
            })
    finally:
        del model
        gc.collect()

    if len(chunk_texts) == 1:
        final_text = chunk_texts[0]
    elif getattr(args, "chunk_seconds_quiet", None) is not None:
        final_text = join_nonoverlapping_transcripts(chunk_texts)
    else:
        final_text = merge_overlapping_transcripts(
            chunk_texts, max_words=args.merge_max_words, max_chars=args.merge_max_chars,
        )
    write_results(out_path, final_text, "\n\n".join(chunk_texts), details)


def locate_whisper_cpp_model(args: argparse.Namespace) -> Path:
    names = {
        "large-v3-turbo": "ggml-large-v3-turbo.bin",
        "large-v3": "ggml-large-v3.bin",
        "base.en": "ggml-base.en.bin",
    }
    filename = names[args.whisper_size]
    minimum = WHISPER_CPP_MIN_BYTES[args.whisper_size]
    candidates = [
        MODELS_DIR / filename,
        Path.home() / "transcribe-tools" / "models" / filename,
        Path.home() / "transcribe-tools-qwen" / "models" / filename,
    ]
    incomplete: list[str] = []
    for target in candidates:
        try:
            size = target.stat().st_size if target.is_file() else 0
        except OSError:
            size = 0
        if size >= minimum:
            if target != candidates[0]:
                print(f"Using existing whisper.cpp model outside the toolkit directory: {target}", flush=True)
            return target
        if size:
            incomplete.append(f"{target} ({size} bytes)")
    target = candidates[0]
    message = (
        f"whisper.cpp model is missing or incomplete: {target}\n"
        f"Expected at least {minimum:,} bytes for {args.whisper_size}. "
        "Run install.sh (without --skip-model-download) to resume the download."
    )
    if incomplete:
        message += "\nIncomplete file(s) found: " + "; ".join(incomplete)
    raise FileNotFoundError(message)


def run_whisper_cpp(args: argparse.Namespace, audio: Path, duration: float,
                    temp_dir: Path, out_path: Path) -> None:
    candidates = [APP_DIR / "bin" / "whisper-cli", APP_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"]
    binary = next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
    if binary is None:
        binary = shutil.which("whisper-cli")
    if not binary:
        raise FileNotFoundError("whisper-cli not found; rerun install.sh to build whisper.cpp.")
    model = locate_whisper_cpp_model(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lang = args.lang.lower()
    beam_size = args.beam_size if args.beam_size is not None else default_beam_size_for("whisper-cpp", args.lang)
    beam_size = validate_whisper_cpp_beam_size(beam_size)
    chunk_seconds, overlap_seconds = effective_chunk_settings(args)
    if getattr(args, "internal_chunking", False):
        # whisper-cli performs its own windowed decoding on the entire input;
        # avoid the toolkit's FFmpeg-created outer windows in this mode.
        chunks = [(audio, 0.0, duration)]
        print(f"Using whisper.cpp internal windowing for the full {duration:.1f}s file", flush=True)
    else:
        chunks = make_external_chunks(audio, duration, temp_dir, args)
    quiet_mode = getattr(args, "chunk_seconds_quiet", None) is not None
    chunk_texts: list[str] = []
    details: list[dict[str, Any]] = []
    for index, (chunk, chunk_start, chunk_end) in enumerate(chunks, 1):
        output_base = temp_dir / f"whisper_cpp_transcript_{index:04d}"
        cmd = [str(binary), "-m", str(model), "-f", str(chunk), "-t", str(args.threads),
               "-bs", str(beam_size), "-l", lang, "-otxt", "-of", str(output_base)]
        print(f"[{index}/{len(chunks)}] {chunk_start:.1f}s–{chunk_end:.1f}s", flush=True)
        run(cmd)
        generated = Path(str(output_base) + ".txt")
        if not generated.is_file() or generated.stat().st_size == 0:
            raise RuntimeError(f"whisper.cpp did not create its expected transcript: {generated}")
        text = generated.read_text(encoding="utf-8", errors="replace").strip()
        print_transcript_block(text, label=f"whisper.cpp transcript [{index}/{len(chunks)}]")
        chunk_texts.append(text)
        details.append({"engine": "whisper-cpp", "model": args.whisper_size, "chunk": index,
                        "start": chunk_start, "end": chunk_end, "text": text,
                        "chunking": ("ffmpeg-low-energy-no-overlap" if quiet_mode else
                                     ("whisper-cpp-internal-windowing" if getattr(args, "internal_chunking", False) else "manual-overlap"))})
    if len(chunk_texts) == 1:
        final_text = chunk_texts[0]
    elif quiet_mode:
        final_text = join_nonoverlapping_transcripts(chunk_texts)
    else:
        final_text = merge_overlapping_transcripts(
            chunk_texts, max_words=args.merge_max_words, max_chars=args.merge_max_chars
        )
    write_results(out_path, final_text, "\n\n".join(chunk_texts), details)



def preferred_hf_home(repo_id: str | None = None) -> Path:
    """Select a useful HF_HOME without ignoring an explicitly configured cache.

    The resolver still checks every configured root, but the selected HF_HOME
    also matters when a backend falls back to Hugging Face's own local-only
    lookup. Prefer a root already containing the requested repository.
    """
    cache_roots = configured_hf_cache_roots()

    def as_hf_home(cache: Path) -> Path:
        return cache.parent if cache.name == "hub" else cache

    if repo_id:
        repo_folder = "models--" + repo_id.replace("/", "--")
        for cache in cache_roots:
            if (cache / repo_folder).is_dir():
                return as_hf_home(cache)

    repo_dirs = [
        "models--Qwen--Qwen3-ASR-0.6B", "models--Qwen--Qwen3-ASR-1.7B",
        "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
        "models--Systran--faster-whisper-large-v3-turbo",  # legacy cache only
        "models--Systran--faster-whisper-large-v3", "models--Systran--faster-whisper-base.en",
        "models--nvidia--canary-1b", "models--nvidia--canary-1b-flash",
        "models--nvidia--canary-1b-v2",
        "models--facebook--seamless-m4t-v2-large",
        "models--ibm-granite--granite-speech-3.3-2b",
        "models--ibm-granite--granite-speech-3.3-8b",
        "models--ibm-granite--granite-speech-5.0-470m-turboctc",
        "models--CohereLabs--cohere-transcribe-03-2026",
        "models--OpenMOSS-Team--MOSS-Transcribe-Diarize",
        "models--OpenMOSS-Team--MOSS-Transcribe-preview-2B",
        "models--Edge0--ARK-ASR-3B",
        "models--distil-whisper--distil-large-v3.5-ct2",
    ]
    for cache in cache_roots:
        if any((cache / name).is_dir() for name in repo_dirs):
            return as_hf_home(cache)

    return MODELS_DIR / "hf-cache"


def hf_repo_for_args(args: argparse.Namespace, engine: str) -> str | None:
    """Return the model repo used for cache selection, unless a local path is set."""
    if engine == "qwen":
        if args.model == "qwen3-1.7b":
            override = args.qwen_checkpoint_17
            default = "Qwen/Qwen3-ASR-1.7B"
        else:
            override = args.qwen_checkpoint_06
            default = "Qwen/Qwen3-ASR-0.6B"
        if override and Path(override).expanduser().exists():
            return None
        return override or default
    if engine == "faster-whisper":
        return DISTIL_WHISPER_REPO_ID if args.model == "distil-whisper-large-v3.5" else faster_whisper_repo_id(args.whisper_size)
    if args.model in TRANSFORMERS_MODEL_IDS:
        return TRANSFORMERS_MODEL_IDS[args.model]
    # NeMo checkpoints are stored in NEMO_CACHE_DIR, not in the HF repo cache.
    return None


def preferred_nemo_cache(model_id: str = "nvidia/canary-1b-flash") -> Path:
    """Prefer the cache containing the *selected* Canary checkpoint.

    Canary-1B and Canary-1B-Flash are independent weights. Do not select a cache
    merely because the other variant is present there.
    """
    filename = model_id.rstrip("/").rsplit("/", 1)[-1] + ".nemo"
    candidates = [
        Path.home() / ".cache" / "torch" / "NeMo",
        MODELS_DIR / "nemo-cache",
        MODELS_DIR / "hf-cache" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    # Keep order while removing duplicates.
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    for candidate in unique:
        if candidate.is_dir() and any(candidate.rglob(filename)):
            return candidate
    return MODELS_DIR / "nemo-cache"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe cleaned or raw speech with isolated ASR backends.")
    parser.add_argument("audio", help="Audio file; inputs are converted to 16 kHz mono PCM as needed")
    parser.add_argument("-o", "--output", help="Output DIRECTORY for this run (default: AUDIO-BASENAME.MODEL.YYYYMMDD-HHMMSS)")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace artifacts from a previous run in the chosen output directory")
    parser.add_argument("--model", choices=tuple(MODEL_TO_ENGINE), default="faster-whisper",
                        help="Offline local ASR: faster-whisper, distil-whisper-large-v3.5, qwen3, Canary variants, SeamlessM4T, Granite Speech 3.3/5.0 TurboCTC, Cohere Transcribe, MOSS Transcribe (Diarize/Preview), ARK-ASR-3B, or whisper.cpp. Model weights must be prefetched by install.sh while online.")
    parser.add_argument("--lang", default="en", help="Language code (default en; e.g. bg, de) or auto")
    parser.add_argument("--threads", type=int, default=16, help="CPU thread count (default 16)")
    parser.add_argument("--chunk-seconds", type=float, default=None,
                        help="Use standard overlapping windows instead of the default quiet-boundary mode; model-specific default cap is 300s (legacy Canary 35s; SeamlessM4T v2 50s). Long files are balanced into windows, never a tiny tail shard.")
    parser.add_argument("--overlap-seconds", type=float, default=None,
                        help="Overlap for standard windows selected with --chunk-seconds (default 15s; legacy Canary 7s). The default quiet-boundary mode has zero overlap.")
    parser.add_argument("--chunk-seconds-quiet", type=float, default=None,
                        help="FFmpeg quiet-boundary chunks with zero overlap; default target is 40s for most backends (30–1200; ARK-ASR uses a 25s target because its windows are capped at 30s). Chunks are balanced first, then boundaries move to nearby quiet windows. Use --chunk-seconds or --internal-chunking to select another mode.")
    parser.add_argument("--chunk-quiet-search-seconds", type=float, default=None,
                        help="Quiet mode only: search +/- this many seconds around each balanced boundary (default 8)")
    parser.add_argument("--chunk-quiet-window-ms", type=float, default=None,
                        help="Quiet mode only: choose the boundary using the lowest-energy window of this length (default 200 ms)")
    parser.add_argument("--chunk-quiet-hop-ms", type=float, default=None,
                        help="Quiet mode only: scan candidate window centers at this resolution (default 10 ms)")
    parser.add_argument("--internal-chunking", action="store_true",
                        help="Use backend-native full-file/internal handling instead of toolkit FFmpeg outer windows. Qwen uses its native low-energy splitter; Canary-1B-v2 uses in-process low-energy zero-overlap WAV segments; faster-whisper/whisper.cpp use their own windows; SeamlessM4T uses encoder chunk attention; Granite uses one full-file call with its encoder block attention (not a transcript-level chunk-and-stitch API and potentially RAM-intensive). Legacy Canary-1B/Flash are not supported.")
    parser.add_argument("--qwen-internal-chunk-seconds", type=int, default=120,
                        help="Qwen only, with --internal-chunking: maximum seconds per native low-energy chunk (default 120; range 30–1200). Lowers XPU/RAM peak without FFmpeg windows.")
    parser.add_argument("--canary-internal-chunk-seconds", type=int, default=120,
                        help="Canary-1B-v2 only, with --internal-chunking: maximum seconds per low-energy, zero-overlap WAV segment (default 120; range 30–1200). Segments are selected/copied in Python, not FFmpeg, because NeMo native chunking needs the optional timestamp model.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Maximum generated output tokens per Qwen/Granite/Seamless segment (default: Qwen 512; Granite/Seamless 4096). Qwen's 512 default follows its Transformers backend default and avoids excessive decode loops on short chunks; raise explicitly for unusually long transcripts.")
    parser.add_argument("--beam-size", type=int, default=None,
                        help="Whisper decoder beam size (default 5; Bulgarian defaults to 20 for faster-whisper and 8 for whisper.cpp; whisper.cpp supports at most 8)")
    parser.add_argument("--canary-beam-size", type=int, default=None,
                        help="Decoder beam width for all Canary models (default 5; 1 is faster, 5 is the accuracy-oriented default)")
    parser.add_argument("--granite-beam-size", type=int, default=None,
                        help="Beam width for IBM Granite Speech (default 1; higher values can improve decoding at additional CPU cost)")
    previous_context_group = parser.add_mutually_exclusive_group()
    previous_context_group.add_argument("--no-previous-text-context", action="store_true",
                        help="Disable previous-window text context in faster-whisper (helps avoid repetition loops)")
    previous_context_group.add_argument("--previous-text-context", action="store_true",
                        help="Force previous-window text context in faster-whisper, including quiet-boundary chunks")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0,
                        help="faster-whisper repetition n-gram suppression (default 0; set 3 to suppress repeated trigrams)")
    parser.add_argument("--whisper-size", choices=("large-v3-turbo", "large-v3", "base.en"), default=None,
                        help="Model used by faster-whisper / whisper-cpp (default large-v3 for Bulgarian; large-v3-turbo otherwise)")
    parser.add_argument("--qwen-checkpoint-06", help="Override Qwen 0.6B Hugging Face model id or local directory")
    parser.add_argument("--qwen-checkpoint-17", help="Override Qwen 1.7B Hugging Face model id or local directory")
    parser.add_argument("--qwen-device", choices=("auto", "xpu", "cpu"), default=os.environ.get("QWEN_DEVICE", "cpu"),
                        help="Qwen only: use CPU by default, or choose auto/XPU explicitly")
    parser.add_argument("--merge-max-words", type=int, default=45,
                        help="Maximum tokens considered at each chunk boundary for overlap merging (default 45)")
    parser.add_argument("--merge-max-chars", type=int, default=360,
                        help="Maximum characters considered at each chunk boundary for overlap merging (default 360)")
    vad_group = parser.add_mutually_exclusive_group()
    vad_group.add_argument("--vad", action="store_true",
                           help="Enable conservative VAD for faster-whisper, including standard-overlap chunks")
    vad_group.add_argument("--no-vad", action="store_true",
                           help="Disable VAD for faster-whisper (quiet-boundary mode enables it by default)")
    parser.add_argument("--_worker", choices=("qwen", "canary", "faster-whisper", "seamless-m4t", "granite-speech", "granite5-ctc", "cohere-asr", "moss-diarize", "moss-preview", "ark-asr"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Quiet-boundary, zero-overlap splitting is the default outer-chunking mode.
    # Explicit standard windows and backend-native/internal mode remain opt-ins
    # that naturally suppress this default. Explicit --chunk-seconds-quiet still
    # wins through argparse and is validated against conflicting options below.
    legacy_canary = args.model in {"canary", "canary-1b", "canary-1b-flash"}
    if (
        args.chunk_seconds_quiet is None
        and not args.internal_chunking
        and args.chunk_seconds is None
        and args.overlap_seconds is None
        and not legacy_canary
    ):
        # ARK's official processor limits each audio prompt to 30 seconds.
        # Use a 25-second default to leave room for quiet-boundary movement.
        args.chunk_seconds_quiet = 25.0 if args.model == "ark-asr-3b" else 40.0
        if args.chunk_quiet_search_seconds is None:
            args.chunk_quiet_search_seconds = 2.0 if args.model == "ark-asr-3b" else 8.0
        if args.chunk_quiet_window_ms is None:
            args.chunk_quiet_window_ms = 200.0
        if args.chunk_quiet_hop_ms is None:
            args.chunk_quiet_hop_ms = 10.0
    # ARK's processor hard-caps each audio prompt at 30 seconds. Keep an
    # explicitly selected quiet target under that cap, not just the default.
    if args.model == "ark-asr-3b" and args.chunk_seconds_quiet is not None:
        if args.chunk_quiet_search_seconds is None:
            args.chunk_quiet_search_seconds = 2.0
        if args.chunk_quiet_window_ms is None:
            args.chunk_quiet_window_ms = 200.0
        if args.chunk_quiet_hop_ms is None:
            args.chunk_quiet_hop_ms = 10.0
    return args


def default_beam_size_for(engine: str, language: str) -> int:
    """Choose decoder width without exceeding whisper.cpp's hard decoder limit."""
    if language.lower() == "bg":
        if engine == "faster-whisper":
            return 20
        if engine == "whisper-cpp":
            return WHISPER_CPP_MAX_DECODERS
    return 5


def validate_whisper_cpp_beam_size(beam_size: int) -> int:
    """Fail early on a whisper.cpp beam width that its compiled decoder cannot support."""
    if beam_size < 1:
        raise ValueError("--beam-size must be at least 1")
    if beam_size > WHISPER_CPP_MAX_DECODERS:
        raise ValueError(
            f"whisper.cpp supports --beam-size at most {WHISPER_CPP_MAX_DECODERS} "
            f"(requested {beam_size}); use --beam-size {WHISPER_CPP_MAX_DECODERS} or less. "
            "This limit does not apply to faster-whisper."
        )
    return beam_size


def apply_inference_defaults(args: argparse.Namespace, engine: str) -> None:
    """Resolve model-specific decoding and hard-bounded chunking defaults."""
    legacy_canary = args.model in {"canary", "canary-1b", "canary-1b-flash"}
    seamless = args.model == "seamless-m4t-v2"
    # The five-minute maximum is applied to all engines. Legacy Canary's
    # separate <40s input limit takes precedence. Overlap is 15s by default
    # at the 300s cap and scales down for shorter user-selected windows.
    if args.chunk_seconds is None:
        args.chunk_seconds = 35.0 if legacy_canary else (50.0 if seamless else 300.0)
    if args.overlap_seconds is None:
        if legacy_canary:
            args.overlap_seconds = min(7.0, args.chunk_seconds * 0.2)
        elif seamless:
            args.overlap_seconds = min(8.0, args.chunk_seconds * 0.16)
        else:
            args.overlap_seconds = min(15.0, args.chunk_seconds * 0.1)

    whisper_engines = {"faster-whisper", "whisper-cpp"}
    if engine in whisper_engines:
        if args.whisper_size is None:
            # Bulgarian uses the full multilingual checkpoint and wider search;
            # Turbo remains the speed-oriented default for other languages.
            args.whisper_size = (
                "large-v3" if args.lang.lower() == "bg" else "large-v3-turbo"
            )
    elif args.whisper_size is not None:
        raise ValueError("--whisper-size only applies to --model faster-whisper or --model whisper-cpp")

    requested_canary_beam_size = args.canary_beam_size
    if engine == "canary":
        if requested_canary_beam_size is None:
            # Use an explicit accuracy-oriented default instead of silently
            # inheriting checkpoint-specific greedy beam_size=1.
            args.canary_beam_size = 5
        elif requested_canary_beam_size < 1:
            raise ValueError("--canary-beam-size must be at least 1")
    elif requested_canary_beam_size is not None:
        raise ValueError("--canary-beam-size only applies to Canary models")

    requested_granite_beam_size = args.granite_beam_size
    if engine == "granite-speech":
        if requested_granite_beam_size is None:
            args.granite_beam_size = 1
        elif requested_granite_beam_size < 1:
            raise ValueError("--granite-beam-size must be at least 1")
    elif requested_granite_beam_size is not None:
        raise ValueError("--granite-beam-size only applies to IBM Granite Speech models")

    requested_beam_size = args.beam_size
    if engine in whisper_engines:
        if requested_beam_size is None:
            # Both Whisper implementations accept the same user-facing option.
            args.beam_size = default_beam_size_for(engine, args.lang)
        elif requested_beam_size < 1:
            raise ValueError("--beam-size must be at least 1")
        if engine == "whisper-cpp":
            validate_whisper_cpp_beam_size(args.beam_size)
    elif requested_beam_size is not None:
        raise ValueError("--beam-size only applies to --model faster-whisper or --model whisper-cpp; use --canary-beam-size for Canary and --granite-beam-size for IBM Granite Speech")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be >= 0")
    if args.lang.lower() == "bg" and engine in {"faster-whisper", "whisper-cpp"} and args.whisper_size == "base.en":
        raise ValueError("whisper-size base.en is English-only; use large-v3 or large-v3-turbo for Bulgarian.")
    if legacy_canary and args.chunk_seconds is not None and args.chunk_seconds >= 40:
        raise ValueError("Legacy Canary-1B/Flash chunks must be shorter than 40 seconds; use --chunk-seconds 35 or --model canary-1b-v2.")
    if seamless and args.chunk_seconds is not None and args.chunk_seconds > 60 and not args.internal_chunking:
        raise ValueError("SeamlessM4T v2 external windows are capped at 60 seconds per model call; use --chunk-seconds 50 or less, or explicitly select --internal-chunking for the full-file encoder-attention path.")
    if args.chunk_seconds > 300:
        raise ValueError("--chunk-seconds cannot exceed the hard 300-second maximum (legacy Canary must stay below 40 seconds).")


def run_selected(args: argparse.Namespace, argv: list[str]) -> int:
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.max_new_tokens is not None and args.max_new_tokens < 32:
        raise ValueError("--max-new-tokens must be at least 32")
    if not 30 <= getattr(args, "qwen_internal_chunk_seconds", 120) <= 1200:
        raise ValueError("--qwen-internal-chunk-seconds must be between 30 and 1200")
    if not 30 <= getattr(args, "canary_internal_chunk_seconds", 120) <= 1200:
        raise ValueError("--canary-internal-chunk-seconds must be between 30 and 1200")
    if args.merge_max_words < 3 or args.merge_max_chars < 10:
        raise ValueError("--merge-max-words must be >= 3 and --merge-max-chars must be >= 10")
    quiet_cap = getattr(args, "chunk_seconds_quiet", None)
    quiet_tuning = (
        getattr(args, "chunk_quiet_search_seconds", None),
        getattr(args, "chunk_quiet_window_ms", None),
        getattr(args, "chunk_quiet_hop_ms", None),
    )
    if quiet_cap is None:
        if any(value is not None for value in quiet_tuning):
            raise ValueError("--chunk-quiet-search-seconds, --chunk-quiet-window-ms, and --chunk-quiet-hop-ms require --chunk-seconds-quiet")
    else:
        min_quiet_cap = 20 if args.model == "ark-asr-3b" else 30
        if not math.isfinite(quiet_cap) or not min_quiet_cap <= quiet_cap <= 1200:
            raise ValueError(f"--chunk-seconds-quiet must be between {min_quiet_cap} and 1200 seconds")
        if args.internal_chunking:
            raise ValueError("--chunk-seconds-quiet selects external FFmpeg chunking and cannot be combined with --internal-chunking")
        if args.chunk_seconds is not None:
            raise ValueError("Use either --chunk-seconds or --chunk-seconds-quiet, not both")
        if args.overlap_seconds is not None and args.overlap_seconds != 0:
            raise ValueError("Quiet-boundary chunking always has zero overlap; omit --overlap-seconds or set it to 0")
        quiet_search = 8.0 if quiet_tuning[0] is None else quiet_tuning[0]
        quiet_window = 200.0 if quiet_tuning[1] is None else quiet_tuning[1]
        quiet_hop = 10.0 if quiet_tuning[2] is None else quiet_tuning[2]
        if not math.isfinite(quiet_search) or not 0 <= quiet_search <= 60:
            raise ValueError("--chunk-quiet-search-seconds must be between 0 and 60")
        if quiet_search * 2 >= quiet_cap:
            raise ValueError("--chunk-quiet-search-seconds must be less than half of --chunk-seconds-quiet")
        if args.model == "ark-asr-3b" and quiet_cap + 2 * quiet_search > 30:
            raise ValueError(
                "ARK-ASR-3B accepts at most 30 seconds per audio window; "
                "use a quiet-chunk target/search combination with target + 2×search <= 30 "
                "(the default is 25s with a 2s quiet search)."
            )
        if not math.isfinite(quiet_window) or not 10 <= quiet_window <= 5000:
            raise ValueError("--chunk-quiet-window-ms must be between 10 and 5000")
        if not math.isfinite(quiet_hop) or not 1 <= quiet_hop <= 1000:
            raise ValueError("--chunk-quiet-hop-ms must be between 1 and 1000")
        if args.model in {"canary", "canary-1b", "canary-1b-flash"}:
            if quiet_cap >= 40:
                raise ValueError("Legacy Canary-1B/Flash quiet chunks must be shorter than 40 seconds; use Canary-v2 or a shorter target")
            if quiet_cap + 2 * quiet_search >= 40:
                raise ValueError(
                    "Legacy Canary-1B/Flash quiet chunks must remain below 40 seconds even after boundary shifts; "
                    "reduce --chunk-seconds-quiet or --chunk-quiet-search-seconds"
                )
    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    engine = MODEL_TO_ENGINE[args.model]
    apply_inference_defaults(args, engine)
    # Validate explicit language/model combinations before creating an output
    # directory, so bad requests do not leave an apparently valid run folder.
    if engine == "qwen":
        validate_qwen_language(args.lang)
    elif engine == "canary":
        validate_canary_language(args)
    elif engine == "seamless-m4t":
        seamless_lang_code(args.lang)
    elif engine == "granite-speech" and args.lang != "auto" and args.lang.lower() not in GRANITE_SPEECH_LANGUAGES:
        raise ValueError(
            "IBM Granite Speech 3.3 supports ASR in English, French, German, Spanish, and Portuguese; "
            f"it does not document support for {args.lang!r}. Use Whisper or Canary-v2 for Bulgarian."
        )
    elif engine == "granite5-ctc" and args.lang != "auto" and args.lang.lower() not in GRANITE5_LANGUAGES:
        raise ValueError(f"Granite Speech 5.0 TurboCTC supports {', '.join(sorted(GRANITE5_LANGUAGES))}; got {args.lang!r}.")
    elif engine == "cohere-asr" and args.lang != "auto" and args.lang.lower() not in COHERE_LANGUAGES:
        raise ValueError(f"Cohere Transcribe supports {', '.join(sorted(COHERE_LANGUAGES))}; got {args.lang!r}.")
    elif engine == "moss-preview" and args.lang.lower() not in {"en", "auto"}:
        raise ValueError("MOSS-Transcribe-preview-2B is English-only; use --lang en.")
    elif engine == "ark-asr" and args.lang != "auto" and args.lang.lower() not in ARK_ASR_LANGUAGES:
        raise ValueError(
            "ARK-ASR-3B documents support for Chinese, English, German, Japanese, French, Korean, "
            "Spanish, Polish, Italian, Romanian, Hungarian, Czech, Dutch, Finnish, Croatian, Slovak, "
            "Slovene, Estonian, and Lithuanian; it does not document support for "
            f"{args.lang!r}. Use Whisper or Canary-1B-v2 for Bulgarian."
        )
    if getattr(args, "internal_chunking", False) and engine == "canary":
        selected_canary = canary_model_id_for_args(args).rstrip("/").rsplit("/", 1)[-1]
        if selected_canary != "canary-1b-v2":
            raise ValueError(
                "--internal-chunking requires Canary-1B-v2 for Canary models; "
                "legacy Canary-1B/Flash use short external windows."
            )
    if args._worker is None and engine != "whisper-cpp":
        required_python = VENVS_DIR / ENGINE_TO_VENV[engine] / "bin" / "python"
        if not required_python.is_file():
            raise FileNotFoundError(f"Environment is missing: {required_python.parent}. Run install.sh first.")
    output_dir = output_directory_for(args, engine)
    out_path = output_dir / "transcript.txt"
    if output_dir == source or out_path.resolve() == source.resolve():
        raise ValueError("Output directory must not overwrite the input audio.")
    # Allocate a run folder only once in the dispatcher. The absolute folder is
    # passed down to the isolated worker so it cannot invent a different timestamp.
    if args._worker is None:
        prepare_run_output(args, engine, output_dir)
        args.output = str(output_dir)
    elif not output_dir.is_dir():
        raise FileNotFoundError(f"Output run directory does not exist: {output_dir}")
    # Direct hidden worker-mode calls must also remain offline. In ordinary use,
    # the parent applies these flags to its child process before Python starts.
    if args._worker is not None:
        enforce_offline_model_loading()
    if args._worker is None and engine != "whisper-cpp":
        venv = VENVS_DIR / ENGINE_TO_VENV[engine]
        python = venv / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"Environment is missing: {venv}. Run install.sh first.")
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        enforce_offline_model_loading(env)
        # Capture explicit cache roots before choosing this run's preferred
        # HF_HOME. Otherwise setting HF_HOME below can hide a model cached under
        # a custom HF_HOME/HUGGINGFACE_HUB_CACHE in the invoking shell.
        env["SPEECH_TOOLKIT_HF_CACHE_ROOTS"] = os.pathsep.join(
            str(root) for root in configured_hf_cache_roots()
        )
        hf_home = preferred_hf_home(hf_repo_for_args(args, engine))
        if not env.get("HF_HOME"):
            env["HF_HOME"] = str(hf_home)
        if not env.get("HUGGINGFACE_HUB_CACHE") and not env.get("HF_HUB_CACHE"):
            env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        if engine == "canary":
            env["NEMO_CACHE_DIR"] = str(preferred_nemo_cache(canary_model_id_for_args(args)))
        env["NEMO_DISABLE_TELEMETRY"] = "1"
        env["ONELOGGER_DISABLE"] = "1"
        print("Inference is offline-only: using cached model files; Hub/API checks and downloads are disabled.", flush=True)
        # The same script handles dispatch and backend work; the worker flag
        # causes it to run inside exactly one model-specific interpreter.
        child_args = _argv_with_output_directory(argv, output_dir)
        cmd = [str(python), str(Path(__file__).resolve()), "--_worker", engine, *child_args]
        print(f"Output directory: {output_dir}", flush=True)
        try:
            result = run(cmd, env=env).returncode
            transcript_exists = out_path.is_file()
            if result == 0 and not transcript_exists:
                finish_run_manifest(output_dir, "failed", 2,
                                    "Worker exited successfully but did not create transcript.txt")
                print(f"transcribe: ERROR: worker exited successfully but did not create {out_path}", file=sys.stderr)
                return 2
            finish_run_manifest(output_dir, "complete" if result == 0 else "failed", result,
                                None if result == 0 else f"Worker exited with code {result}")
            return result
        except Exception as exc:
            finish_run_manifest(output_dir, "failed", 2, f"{type(exc).__name__}: {exc}")
            raise

    if args._worker is None and engine == "whisper-cpp":
        print(f"Output directory: {output_dir}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="transcribe-prepare-") as tmp:
                temp_dir = Path(tmp)
                prepared, duration = prepare_audio(source, temp_dir)
                run_whisper_cpp(args, prepared, duration, temp_dir, out_path)
            finish_run_manifest(output_dir, "complete", 0)
            return 0
        except Exception as exc:
            finish_run_manifest(output_dir, "failed", 2, f"{type(exc).__name__}: {exc}")
            raise

    # Worker mode: all optional/non-standard imports happen only after the
    # isolated interpreter has been selected by the dispatcher.
    with tempfile.TemporaryDirectory(prefix="transcribe-work-") as tmp:
        temp_dir = Path(tmp)
        audio, duration = prepare_audio(source, temp_dir)
        if args._worker in ("qwen", "canary"):
            if args._worker == "qwen":
                validate_qwen_language(args.lang)
            else:
                validate_canary_language(args)
            if args.chunk_seconds is not None and (
                args.chunk_seconds <= 0 or args.chunk_seconds > 300
                or args.overlap_seconds < 0 or args.overlap_seconds >= args.chunk_seconds
            ):
                raise ValueError("Chunk seconds must be >0 and <=300; overlap must be from 0 up to (but not including) chunk seconds.")
            if args._worker == "qwen":
                worker_qwen(args, audio, duration, temp_dir, out_path)
            else:
                worker_canary(args, audio, duration, temp_dir, out_path)
        elif args._worker == "faster-whisper":
            worker_faster_whisper(args, audio, duration, temp_dir, out_path)
        elif args._worker in ("seamless-m4t", "granite-speech"):
            worker_transformers_asr(args, audio, duration, temp_dir, out_path, args._worker)
        elif args._worker in OPEN_ASR_ENGINES:
            worker_open_asr(args, audio, duration, temp_dir, out_path, args._worker)
        else:
            raise ValueError(f"Unknown worker engine: {args._worker}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parse_args(args_list)
        return run_selected(args, args_list)
    except KeyboardInterrupt:
        print("transcribe: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ImportError) as exc:
        print(f"transcribe: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
