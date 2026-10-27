#!/usr/bin/env bash
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
# Compare backends' internal long-audio handling without toolkit-made FFmpeg chunks.
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  compare-transcribers-long.sh [OPTIONS] AUDIO LANG_CODE

This sweep enables each backend's memory-aware long-audio mode
(--internal-chunking). Qwen uses its native low-energy splitter; Canary-v2
uses in-process low-energy, zero-overlap WAV segments because its NeMo native
merger requires the optional timestamp model; faster-whisper and whisper.cpp
use their own windowing. A watchdog enforces a 25 GB default process-tree limit.

Default models:
  qwen3, qwen3-1.7b, canary-1b-v2,
  faster-whisper (large-v3-turbo and large-v3),
  whisper-cpp (large-v3-turbo and large-v3)

Legacy Canary-1B and Canary-1B-Flash are deliberately excluded because they do
not have Canary-v2's documented dynamic long-form chunking path.

Options:
  --models LIST             Comma-separated subset of: qwen3,qwen3-1.7b,
                            canary-1b-v2,faster-whisper,whisper-cpp
  --ram-limit-gb N          Per-run process-tree memory limit in GB (default 25)
  --monitor-interval SEC    Memory sampling interval, at least 0.05s (default 0.5)
  --threads N               Pass inference thread count (default: transcribe default)
  --qwen-device auto|xpu|cpu  Qwen device (default auto)
  --qwen-internal-chunk-seconds N  Qwen native low-energy chunk cap (default 120)
  --canary-internal-chunk-seconds N  Canary-v2 low-energy segment cap (default 120)
  --output-root DIR         Store comparisons under DIR
  -h, --help                Show this help

Audio is used as supplied; normalize it separately with norm_audio or
dynamic_norm_audio if desired. The RAM cap is a polling watchdog, not a kernel-enforced cgroup limit;
a very brief peak between samples can exceed the threshold before termination.
Results are sequential and each backend/model has separate transcript, sidecar,
and log files. TRANSCRIBE_COMPARISON_DIR sets the default output root.
USAGE
}

OUTPUT_ROOT=${TRANSCRIBE_COMPARISON_DIR:-"$PWD/transcription-comparisons-long"}
RAM_LIMIT_GB=25
MONITOR_INTERVAL=0.1
THREADS=""
QWEN_DEVICE=cpu
QWEN_INTERNAL_CHUNK_SECONDS=600
CANARY_INTERNAL_CHUNK_SECONDS=600
MODELS_CSV="qwen3,qwen3-1.7b,canary-1b-v2,faster-whisper,whisper-cpp"
POSITIONAL=()

while (($#)); do
  case "$1" in
    --output-root)
      (($# >= 2)) || { echo "ERROR: --output-root needs a directory." >&2; exit 2; }
      OUTPUT_ROOT=$2; shift 2 ;;
    --ram-limit-gb)
      (($# >= 2)) || { echo "ERROR: --ram-limit-gb needs a number." >&2; exit 2; }
      RAM_LIMIT_GB=$2; shift 2 ;;
    --monitor-interval)
      (($# >= 2)) || { echo "ERROR: --monitor-interval needs seconds." >&2; exit 2; }
      MONITOR_INTERVAL=$2; shift 2 ;;
    --threads)
      (($# >= 2)) || { echo "ERROR: --threads needs a count." >&2; exit 2; }
      THREADS=$2; shift 2 ;;
    --qwen-device)
      (($# >= 2)) || { echo "ERROR: --qwen-device needs auto, xpu, or cpu." >&2; exit 2; }
      QWEN_DEVICE=$2; shift 2 ;;
    --qwen-internal-chunk-seconds)
      (($# >= 2)) || { echo "ERROR: --qwen-internal-chunk-seconds needs seconds." >&2; exit 2; }
      QWEN_INTERNAL_CHUNK_SECONDS=$2; shift 2 ;;
    --canary-internal-chunk-seconds)
      (($# >= 2)) || { echo "ERROR: --canary-internal-chunk-seconds needs seconds." >&2; exit 2; }
      CANARY_INTERNAL_CHUNK_SECONDS=$2; shift 2 ;;
    --models)
      (($# >= 2)) || { echo "ERROR: --models needs a comma-separated list." >&2; exit 2; }
      MODELS_CSV=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; while (($#)); do POSITIONAL+=("$1"); shift; done; break ;;
    -*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if ((${#POSITIONAL[@]} != 2)); then usage >&2; exit 2; fi
AUDIO=${POSITIONAL[0]}
LANG=${POSITIONAL[1],,}
if [[ ! -f "$AUDIO" ]]; then echo "ERROR: audio file not found: $AUDIO" >&2; exit 2; fi
if ! [[ "$RAM_LIMIT_GB" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || ! awk -v n="$RAM_LIMIT_GB" 'BEGIN { exit !(n > 0) }'; then
  echo "ERROR: --ram-limit-gb must be a positive number." >&2; exit 2
fi
if ! [[ "$MONITOR_INTERVAL" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || ! awk -v n="$MONITOR_INTERVAL" 'BEGIN { exit !(n >= 0.05) }'; then
  echo "ERROR: --monitor-interval must be at least 0.05 seconds." >&2; exit 2
fi
if [[ "$QWEN_DEVICE" != auto && "$QWEN_DEVICE" != xpu && "$QWEN_DEVICE" != cpu ]]; then
  echo "ERROR: --qwen-device must be auto, xpu, or cpu." >&2; exit 2
fi
if ! [[ "$QWEN_INTERNAL_CHUNK_SECONDS" =~ ^[0-9]+$ ]] || ((QWEN_INTERNAL_CHUNK_SECONDS < 30 || QWEN_INTERNAL_CHUNK_SECONDS > 1200)); then
  echo "ERROR: --qwen-internal-chunk-seconds must be an integer from 30 to 1200." >&2; exit 2
fi
if ! [[ "$CANARY_INTERNAL_CHUNK_SECONDS" =~ ^[0-9]+$ ]] || ((CANARY_INTERNAL_CHUNK_SECONDS < 30 || CANARY_INTERNAL_CHUNK_SECONDS > 1200)); then
  echo "ERROR: --canary-internal-chunk-seconds must be an integer from 30 to 1200." >&2; exit 2
fi

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
if ((${#MODELS[@]} == 0)); then echo "ERROR: --models list is empty." >&2; exit 2; fi
for MODEL in "${MODELS[@]}"; do
  case "$MODEL" in
    qwen3|qwen3-1.7b|canary-1b-v2|faster-whisper|whisper-cpp) ;;
    *) echo "ERROR: unsupported model in --models: '$MODEL'. Legacy Canary is not a native-long-audio candidate." >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MEMORY_GUARD=${MEMORY_GUARD:-"$SCRIPT_DIR/memory_guard.py"}
if [[ ! -f "$MEMORY_GUARD" ]]; then
  echo "ERROR: memory guard not found: $MEMORY_GUARD" >&2; exit 2
fi
PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: python3 not found." >&2; exit 127; }
# If running from the installed toolkit directory, dispatch through that exact
# source and environment rather than a possibly stale transcribe on PATH.
TRANSCRIBE_PREFIX=()
if [[ -x "$SCRIPT_DIR/venvs/tools/bin/python" && -f "$SCRIPT_DIR/transcribe.py" ]]; then
  TRANSCRIBE_PREFIX=("$SCRIPT_DIR/venvs/tools/bin/python" "$SCRIPT_DIR/transcribe.py")
elif [[ -n "${TRANSCRIBE_BIN:-}" ]]; then
  TRANSCRIBE_PREFIX=("$TRANSCRIBE_BIN")
elif command -v transcribe >/dev/null 2>&1; then
  TRANSCRIBE_PREFIX=("$(command -v transcribe)")
else
  echo "ERROR: 'transcribe' not found in PATH. Run install.sh or activate the toolkit." >&2; exit 127
fi

mkdir -p "$OUTPUT_ROOT" || exit 1
BASE=$(basename -- "$AUDIO"); BASE=${BASE%.*}
OUT_DIR=$(mktemp -d "$OUTPUT_ROOT/${BASE}_${LANG}.XXXXXX") || exit 1
TRANSCRIBE_AUDIO=$AUDIO
printf 'Input audio is used as supplied: %s\nLanguage:       %s\nResults:        %s\nRAM limit:      %s GB per model\n' \
  "$AUDIO" "$LANG" "$OUT_DIR" "$RAM_LIMIT_GB"
printf 'Models:         %s\nTranscription input: %s\n' "$MODELS_CSV" "$TRANSCRIBE_AUDIO"

FAILED=0
SUCCEEDED=0
RAM_KILLED=0
for MODEL in "${MODELS[@]}"; do
  case "$MODEL" in
    faster-whisper|whisper-cpp) RUN_SIZES=(large-v3-turbo large-v3) ;;
    *) RUN_SIZES=("") ;;
  esac
  for WHISPER_SIZE in "${RUN_SIZES[@]}"; do
    if [[ -n "$WHISPER_SIZE" ]]; then RUN_NAME="${MODEL}-${WHISPER_SIZE}"; else RUN_NAME=$MODEL; fi
    printf '\n========== %s (%s; long-audio memory-aware mode) ==========\n' "$RUN_NAME" "$LANG"
    OUT_TXT="$OUT_DIR/${RUN_NAME}.txt"
    LOG="$OUT_DIR/${RUN_NAME}.log"
    TRANSCRIBE_ARGS=("$TRANSCRIBE_AUDIO" --model "$MODEL" --lang "$LANG" --internal-chunking --output "$OUT_TXT")
    [[ -z "$THREADS" ]] || TRANSCRIBE_ARGS+=(--threads "$THREADS")
    if [[ "$MODEL" == qwen3 || "$MODEL" == qwen3-1.7b ]]; then
      TRANSCRIBE_ARGS+=(--qwen-device "$QWEN_DEVICE")
      TRANSCRIBE_ARGS+=(--qwen-internal-chunk-seconds "${QWEN_INTERNAL_CHUNK_SECONDS:-120}")
    elif [[ "$MODEL" == canary-1b-v2 ]]; then
      TRANSCRIBE_ARGS+=(--canary-internal-chunk-seconds "$CANARY_INTERNAL_CHUNK_SECONDS")
    fi
    [[ -z "$WHISPER_SIZE" ]] || TRANSCRIBE_ARGS+=(--whisper-size "$WHISPER_SIZE")

    printf 'RAM monitor: %s GB; sample every %ss\nLog: %s\n' "$RAM_LIMIT_GB" "$MONITOR_INTERVAL" "$LOG"
    "$PYTHON" "$MEMORY_GUARD" --limit-gb "$RAM_LIMIT_GB" --interval "$MONITOR_INTERVAL" -- \
      "${TRANSCRIBE_PREFIX[@]}" "${TRANSCRIBE_ARGS[@]}" >"$LOG" 2>&1
    STATUS=$?
    if [[ "$STATUS" -eq 0 && -s "$OUT_TXT" ]]; then
      printf 'PASS — transcript: %s\n' "$OUT_TXT"
      SUCCEEDED=$((SUCCEEDED + 1))
    elif grep -q 'RAM_LIMIT_EXCEEDED' "$LOG" 2>/dev/null; then
      printf 'KILLED — memory watchdog terminated %s at/above %s GB. Log: %s\n' "$RUN_NAME" "$RAM_LIMIT_GB" "$LOG"
      RAM_KILLED=$((RAM_KILLED + 1)); FAILED=$((FAILED + 1))
      tail -n 12 "$LOG" 2>/dev/null || true
    else
      printf 'FAIL (exit %s). Recent log:\n' "$STATUS"
      tail -n 20 "$LOG" 2>/dev/null || true
      FAILED=$((FAILED + 1))
    fi
  done
done

printf '\n========== SUMMARY ==========\nSucceeded: %d\nFailed/unavailable: %d\nRAM-limit terminations: %d\nResults directory: %s\n' \
  "$SUCCEEDED" "$FAILED" "$RAM_KILLED" "$OUT_DIR"
printf 'Each run uses whole-file input and writes native-backend results; see per-model .log and .chunks.json files.\n'
(( SUCCEEDED > 0 )) || exit 1
exit 0
