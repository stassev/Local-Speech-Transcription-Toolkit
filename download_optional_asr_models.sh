#!/usr/bin/env bash
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
# Download/resume optional open-weight ASR models into the toolkit's local HF cache.
# This only prepares model files. Transcription remains offline-only.
set -uo pipefail

APP_DIR="${APP_DIR:-$HOME/.local/share/speech-toolkit}"
STATUS_FILE="$APP_DIR/state/optional-asr-install-status.txt"

usage() {
  cat <<'USAGE'
Usage: download-optional-asr-models [--models CSV]

Downloads/resumes the optional open-weight ASR model files only; does not reinstall
Python environments or run transcription. Internet is needed only for these explicit
weight downloads. Inference continues to use local cached files and never calls a
hosted inference service.

Default models:
  distil-whisper-large-v3.5
  granite-speech-5.0-turboctc
  moss-transcribe-diarize
  cohere-transcribe-2b
  ark-asr-3b
  moss-transcribe-preview-2b

Examples:
  download-optional-asr-models
  download-optional-asr-models --models distil-whisper-large-v3.5,ark-asr-3b
USAGE
}

DEFAULT_MODELS='distil-whisper-large-v3.5,granite-speech-5.0-turboctc,moss-transcribe-diarize,cohere-transcribe-2b,ark-asr-3b,moss-transcribe-preview-2b'
MODELS_CSV="$DEFAULT_MODELS"
while (($#)); do
  case "$1" in
    --models)
      [[ $# -ge 2 && -n "$2" ]] || { echo 'ERROR: --models needs a comma-separated list.' >&2; exit 2; }
      MODELS_CSV="$2"; shift 2
      while (($#)) && [[ "$1" != -* ]]; do MODELS_CSV+=",$1"; shift; done
      ;;
    --models=*) MODELS_CSV="${1#*=}"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$APP_DIR/state" "$APP_DIR/models/hf-cache/hub"
[[ -f "$STATUS_FILE" ]] || : > "$STATUS_FILE"

set_status() {
  local key="$1" value="$2" tmp
  tmp="$(mktemp "$APP_DIR/state/.optional-asr-status.XXXXXX")" || return 1
  awk -F '\t' -v key="$key" '$1 != key { print }' "$STATUS_FILE" > "$tmp" || { rm -f -- "$tmp"; return 1; }
  printf '%s\t%s\n' "$key" "$value" >> "$tmp"
  mv -- "$tmp" "$STATUS_FILE"
}

get_python() {
  case "$1" in
    distil-whisper-large-v3.5) printf '%s\n' "$APP_DIR/venvs/faster-whisper/bin/python" ;;
    granite-speech-5.0-turboctc) printf '%s\n' "$APP_DIR/venvs/granite5-ctc/bin/python" ;;
    moss-transcribe-diarize) printf '%s\n' "$APP_DIR/venvs/moss-diarize/bin/python" ;;
    cohere-transcribe-2b) printf '%s\n' "$APP_DIR/venvs/cohere-asr/bin/python" ;;
    ark-asr-3b) printf '%s\n' "$APP_DIR/venvs/ark-asr/bin/python" ;;
    moss-transcribe-preview-2b) printf '%s\n' "$APP_DIR/venvs/moss-preview/bin/python" ;;
    *) return 1 ;;
  esac
}

get_repo() {
  case "$1" in
    distil-whisper-large-v3.5) printf '%s\n' 'distil-whisper/distil-large-v3.5-ct2' ;;
    granite-speech-5.0-turboctc) printf '%s\n' 'ibm-granite/granite-speech-5.0-470m-turboctc' ;;
    moss-transcribe-diarize) printf '%s\n' 'OpenMOSS-Team/MOSS-Transcribe-Diarize' ;;
    cohere-transcribe-2b) printf '%s\n' 'CohereLabs/cohere-transcribe-03-2026' ;;
    ark-asr-3b) printf '%s\n' 'Edge0/ARK-ASR-3B' ;;
    moss-transcribe-preview-2b) printf '%s\n' 'OpenMOSS-Team/MOSS-Transcribe-preview-2B' ;;
    *) return 1 ;;
  esac
}

# Accept CSV (including spaces after commas) and preserve the user's selection.
IFS=', ' read -r -a REQUESTED <<< "$MODELS_CSV"
FAILED=0
for model in "${REQUESTED[@]}"; do
  [[ -n "$model" ]] || continue
  case "$model" in
    distil-whisper-large-v3.5|granite-speech-5.0-turboctc|moss-transcribe-diarize|cohere-transcribe-2b|ark-asr-3b|moss-transcribe-preview-2b) ;;
    *) echo "WARN: Unknown optional model '$model'; skipping it." >&2; set_status "$model" "WEIGHTS SKIPPED (unknown model name)" || true; FAILED=1; continue ;;
  esac
  py="$(get_python "$model")"
  repo="$(get_repo "$model")"
  if [[ ! -x "$py" ]]; then
    echo "WARN: $model environment is missing at $py; skipping this model and continuing." >&2
    set_status "$model" "WEIGHTS FAILED (environment unavailable; run install.sh first)" || true
    FAILED=1
    continue
  fi
  echo
  echo "========== Download/resume: $model ($repo) =========="
  echo "Only model/config/tokenizer files are downloaded. No audio or transcript is read or uploaded."
  model_log="$(mktemp "$APP_DIR/state/.optional-asr-download.XXXXXX")" || {
    echo "WARN: Could not create a diagnostic log for $model; continuing." >&2
    set_status "$model" "WEIGHTS FAILED (could not create diagnostic log)" || true
    FAILED=1
    continue
  }
  if OPTIONAL_MODEL_NAME="$model" OPTIONAL_MODEL_REPO="$repo" OPTIONAL_MODEL_CACHE="$APP_DIR/models/hf-cache/hub" "$py" - 2>&1 <<'PY' | tee "$model_log"
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model = os.environ["OPTIONAL_MODEL_NAME"]
repo = os.environ["OPTIONAL_MODEL_REPO"]
active = Path(os.environ["OPTIONAL_MODEL_CACHE"])
active.mkdir(parents=True, exist_ok=True)
roots = [active, Path.home()/".cache/huggingface/hub", Path.home()/"transcribe-tools-qwen/hf-cache/hub"]
roots = list(dict.fromkeys(roots))
patterns = [
    "*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.tiktoken",
    "*.jinja", "*.py", "*.yaml", "*.yml", "tokenizer*", "processor*",
    "configuration*", "modeling*", "processing*", "chat_template*", "spiece.model",
]

def complete(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    if model == "distil-whisper-large-v3.5":
        try:
            return (path / "model.bin").is_file() and (path / "model.bin").stat().st_size >= 1_400_000_000
        except OSError:
            return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("pytorch_model*.bin"))

repo_dir = "models--" + repo.replace("/", "--")
partial_roots = [root for root in roots if (root / repo_dir).is_dir()]
# First reuse any complete local cache. The app cache is preferred for new data.
for root in roots:
    if not (root / repo_dir).is_dir():
        continue
    try:
        path = snapshot_download(repo_id=repo, cache_dir=str(root), local_files_only=True, allow_patterns=patterns)
        if complete(path):
            print(f"WEIGHTS READY (reused): {path}", flush=True)
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass
# Resume the first existing partial cache if any, otherwise download to the toolkit cache.
target = partial_roots[0] if partial_roots else active
print(f"Downloading/resuming into: {target}", flush=True)
path = snapshot_download(repo_id=repo, cache_dir=str(target), allow_patterns=patterns)
if not complete(path):
    raise RuntimeError(f"Downloaded snapshot is incomplete for {repo}: {path}")
print(f"WEIGHTS READY: {path}", flush=True)
PY
  then
    resolved_path="$(sed -nE 's/^WEIGHTS READY( \(reused\))?: //p' "$model_log" | tail -n 1)"
    if [[ -n "$resolved_path" ]]; then
      set_status "$model" "WEIGHTS READY (snapshot: $resolved_path)" || true
    else
      set_status "$model" "WEIGHTS READY (snapshot path not reported; see retained downloader output)" || true
    fi
  else
    echo "WARN: $model download failed; any partial files are preserved. continuing with the next model." >&2
    set_status "$model" "WEIGHTS FAILED (see downloader output; partial files retained; rerun this command to resume)" || true
    FAILED=1
  fi
  rm -f -- "$model_log"
done

set_status "optional model weights" "DOWNLOAD ATTEMPT COMPLETED (see per-model outcomes above; inference remains offline-only)" || true
echo
echo "Optional model outcomes are recorded in: $STATUS_FILE"
if (( FAILED )); then
  echo "Some optional model downloads were skipped or failed; successful models remain usable." >&2
  exit 1
fi
echo "All selected optional model weights are ready."
