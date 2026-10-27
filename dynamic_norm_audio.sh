#!/usr/bin/env bash
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
# Convert audio to 16 kHz mono PCM WAV using FFmpeg's dynamic audio normalization.
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  dynamic_norm_audio INPUT_AUDIO [-o OUTPUT.wav]

Runs the equivalent of:
  ffmpeg -hide_banner -loglevel error -y -i INPUT_AUDIO -vn \\
    -af dynaudnorm -ac 1 -ar 16000 -c:a pcm_s16le OUTPUT.wav

If -o/--out is omitted, writes INPUT_BASENAME_dynaudnorm.wav beside the input.
The original file is never modified. Any FFmpeg-supported audio format is accepted.
USAGE
}

if (($# == 0)); then usage >&2; exit 2; fi
INPUT=""
OUTPUT=""
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -o|--out|--output)
      (($# >= 2)) || { echo "ERROR: $1 requires an output path." >&2; exit 2; }
      OUTPUT=$2; shift 2 ;;
    --)
      shift
      if (($#)); then
        [[ -z "$INPUT" ]] || { echo "ERROR: only one input file is allowed." >&2; exit 2; }
        INPUT=$1; shift
        (($# == 0)) || { echo "ERROR: unexpected extra arguments." >&2; exit 2; }
      fi
      break ;;
    -* ) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      [[ -z "$INPUT" ]] || { echo "ERROR: only one input file is allowed." >&2; exit 2; }
      INPUT=$1; shift ;;
  esac
done

[[ -n "$INPUT" ]] || { echo "ERROR: an input file is required." >&2; usage >&2; exit 2; }
[[ -f "$INPUT" ]] || { echo "ERROR: input file not found: $INPUT" >&2; exit 2; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found in PATH." >&2; exit 127; }
command -v ffprobe >/dev/null 2>&1 || { echo "ERROR: ffprobe not found in PATH." >&2; exit 127; }

INPUT_ABS=$(realpath -- "$INPUT")
if [[ -z "$OUTPUT" ]]; then
  INPUT_DIR=$(dirname -- "$INPUT_ABS")
  INPUT_BASE=$(basename -- "$INPUT_ABS")
  INPUT_BASE=${INPUT_BASE%.*}
  OUTPUT="$INPUT_DIR/${INPUT_BASE}_dynaudnorm.wav"
fi
OUTPUT_DIR=$(dirname -- "$OUTPUT")
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_ABS=$(realpath -m -- "$OUTPUT")
[[ "$OUTPUT_ABS" != "$INPUT_ABS" ]] || { echo "ERROR: output path must differ from input path." >&2; exit 2; }

TMP_OUTPUT=$(mktemp --tmpdir="$OUTPUT_DIR" ".dynamic-norm.XXXXXX.wav")
cleanup() { rm -f -- "$TMP_OUTPUT"; }
trap cleanup EXIT

printf 'Input:  %s\nOutput: %s\nFilter: dynaudnorm; output: 16 kHz mono PCM S16LE\n' "$INPUT_ABS" "$OUTPUT_ABS"
ffmpeg -hide_banner -loglevel error -y -i "$INPUT_ABS" -vn -af 'dynaudnorm' -ac 1 -ar 16000 -c:a pcm_s16le "$TMP_OUTPUT"
[[ -s "$TMP_OUTPUT" ]] || { echo "ERROR: FFmpeg did not produce a non-empty output WAV." >&2; exit 1; }
PROBE=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 "$TMP_OUTPUT")
ACTUAL_CODEC=$(sed -n 's/^codec_name=//p' <<<"$PROBE")
ACTUAL_RATE=$(sed -n 's/^sample_rate=//p' <<<"$PROBE")
ACTUAL_CHANNELS=$(sed -n 's/^channels=//p' <<<"$PROBE")
if [[ "$ACTUAL_CODEC" != pcm_s16le || "$ACTUAL_RATE" != 16000 || "$ACTUAL_CHANNELS" != 1 ]]; then
  echo "ERROR: dynaudnorm output contract failed: expected mono/16000 Hz/pcm_s16le; got codec=$ACTUAL_CODEC rate=$ACTUAL_RATE channels=$ACTUAL_CHANNELS" >&2
  exit 1
fi
mv -f -- "$TMP_OUTPUT" "$OUTPUT_ABS"
trap - EXIT
printf 'Done: %s\n' "$OUTPUT_ABS"
