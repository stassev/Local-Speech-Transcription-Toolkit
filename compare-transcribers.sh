#!/usr/bin/env bash
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
# Run every distinct speech-toolkit backend against one audio file without
# overwriting results. Audio is used exactly as supplied.
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  compare-transcribers.sh [OPTIONS] AUDIO LANG_CODE

Examples:
  # Raw WAV; no preprocessing
  ./compare-transcribers.sh out/261014_0051.wav en

  # MP3 and WAV inputs are both used as supplied
  ./compare-transcribers.sh recording.mp3 bg

Options:
  --output-root DIR           Store comparisons under DIR
  --include-noncommercial-models  Include Canary-1B and SeamlessM4T v2
                                  (CC BY-NC 4.0; noncommercial use only)
  -h, --help                  Show this help

Results are written to a unique subdirectory, each model gets separate
transcript/sidecar/log names, and models run sequentially to limit RAM use.
Alternatively set TRANSCRIBE_COMPARISON_DIR to choose the output root.
USAGE
}

OUTPUT_ROOT=${TRANSCRIBE_COMPARISON_DIR:-"$PWD/transcription-comparisons"}
INCLUDE_NONCOMMERCIAL_MODELS=0
POSITIONAL=()

while (($#)); do
  case "$1" in
    --output-root)
      if (($# < 2)); then echo "ERROR: --output-root needs a directory." >&2; exit 2; fi
      OUTPUT_ROOT=$2; shift 2 ;;
    --include-noncommercial-models) INCLUDE_NONCOMMERCIAL_MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; while (($#)); do POSITIONAL+=("$1"); shift; done; break ;;
    -*) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if ((${#POSITIONAL[@]} != 2)); then usage >&2; exit 2; fi
AUDIO=${POSITIONAL[0]}
LANG=${POSITIONAL[1],,}
if [[ ! -f "$AUDIO" ]]; then
  echo "ERROR: audio file not found: $AUDIO" >&2
  exit 2
fi
if ! command -v transcribe >/dev/null 2>&1; then
  echo "ERROR: 'transcribe' not found in PATH. Run the speech-toolkit installer or activate its PATH." >&2
  exit 127
fi

mkdir -p "$OUTPUT_ROOT" || exit 1
BASE=$(basename -- "$AUDIO")
BASE=${BASE%.*}
OUT_DIR=$(mktemp -d "$OUTPUT_ROOT/${BASE}_${LANG}.XXXXXX") || exit 1
TRANSCRIBE_AUDIO=$AUDIO

printf 'Original input: %s\nLanguage:       %s\nResults:        %s\n' "$AUDIO" "$LANG" "$OUT_DIR"
printf 'Input audio is used as supplied: %s\n' "$TRANSCRIBE_AUDIO"

# One entry per distinct backend. Omit 'canary': transcribe documents it as
# an alias for canary-1b-flash, so running both would duplicate that checkpoint.
# The optional Granite 8B model is not swept by default due to its RAM footprint;
# invoke transcribe directly after opting into its weights with install.sh.
MODELS=(qwen3 qwen3-1.7b canary-1b-flash canary-1b-v2 granite-speech-3.3-2b faster-whisper)
if (( INCLUDE_NONCOMMERCIAL_MODELS == 1 )); then
  echo "NOTICE: Including Canary-1B and SeamlessM4T v2, whose model weights are licensed CC BY-NC 4.0. Use only as permitted by those terms; see THIRD_PARTY_NOTICES.md."
  MODELS+=(canary-1b seamless-m4t-v2)
fi
# whisper-cpp)
WHISPER_SIZES=(large-v3-turbo large-v3)
# base.en)
FAILED=0
SUCCEEDED=0

for MODEL in "${MODELS[@]}"; do
  # Sweep all three Whisper checkpoints for both Whisper backends; every
  # size gets its own transcript and log. Other backends run exactly once.
  if [[ "$MODEL" == "faster-whisper" || "$MODEL" == "whisper-cpp" ]]; then
    RUN_SIZES=("${WHISPER_SIZES[@]}")
  else
    RUN_SIZES=("")
  fi

  for WHISPER_SIZE in "${RUN_SIZES[@]}"; do
    if [[ -n "$WHISPER_SIZE" ]]; then
      RUN_NAME="${MODEL}-${WHISPER_SIZE}"
      printf '\n========== %s (whisper size: %s; %s) ==========\n' "$MODEL" "$WHISPER_SIZE" "$LANG"
    else
      RUN_NAME=$MODEL
      printf '\n========== %s (%s) ==========\n' "$MODEL" "$LANG"
    fi

    OUT_TXT="$OUT_DIR/${RUN_NAME}.txt"
    LOG="$OUT_DIR/${RUN_NAME}.log"
    printf 'Log: %s\n' "$LOG"

    # Toolkit defaults: Whisper beam 20 for Bulgarian and 5 otherwise; Canary
    # beam 5. Unsupported model/language/size combinations may fail independently.
    TRANSCRIBE_ARGS=("$TRANSCRIBE_AUDIO" --model "$MODEL" --lang "$LANG" --chunk-seconds 35 --overlap-seconds 7 --output "$OUT_TXT")
    if [[ -n "$WHISPER_SIZE" ]]; then
      TRANSCRIBE_ARGS+=(--whisper-size "$WHISPER_SIZE")
    fi

    if transcribe "${TRANSCRIBE_ARGS[@]}" >"$LOG" 2>&1; then
      if [[ -s "$OUT_TXT" ]]; then
        printf 'PASS — transcript: %s\n\n' "$OUT_TXT"
        cat "$OUT_TXT"
        printf '\n'
        SUCCEEDED=$((SUCCEEDED + 1))
      else
        printf 'WARN — command exited successfully but no transcript file was produced. Recent log:\n'
        tail -n 20 "$LOG" 2>/dev/null || true
        FAILED=$((FAILED + 1))
      fi
    else
      STATUS=$?
      printf 'FAIL (exit %s). Recent log:\n' "$STATUS"
      tail -n 24 "$LOG" 2>/dev/null || true
      FAILED=$((FAILED + 1))
    fi
  done
done

printf '\n========== SUMMARY ==========\n'
printf 'Succeeded: %d\nFailed/unavailable: %d\nResults directory: %s\n' "$SUCCEEDED" "$FAILED" "$OUT_DIR"
printf 'Each run has its own .txt, .raw.txt, .chunks.json, and .log files where applicable.\n'

# Some model/language/size combinations are intentionally unsupported.
if [[ $SUCCEEDED -eq 0 ]]; then exit 1; fi
exit 0
