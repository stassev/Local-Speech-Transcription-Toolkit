#!/usr/bin/env bash
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
# Run normalization variants through a selectable set of local transcribe backends.
# A failure in one preprocessing method/model is logged and does not stop later runs.
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  transcribe_audio [OPTIONS] INPUT_AUDIO

Default run matrix (four runs, sequentially; WARNING: uses non-commercial models):
  faster-whisper on norm_audio, with 40s quiet-boundary chunks
  faster-whisper on dynaudnorm, with 40s quiet-boundary chunks
  canary-1b-v2 on norm_audio, with 40s quiet-boundary chunks
  canary-1b-v2 on norm_audio, with 100s quiet-boundary chunks
  (faster-whisper uses large-v3-turbo by default for English.)
  To run a generic four-model sweep on both normalizers, set
  TRANSCRIBE_AUDIO_DEFAULT_MODELS to exactly four distinct comma-separated models.

Options:
  --models CSV [MODEL ...]    Replace the curated four-run default with each listed model
                              on both normalizers. Quote CSV values with spaces,
                              or leave a space after commas unquoted.
  --add-models CSV [MODEL ...] Add models to the default matrix (or to
                              --models), running each on both normalizers.
  --normalizers MODEL=SPEC    Repeatable per-model preprocessing setting. SPEC is
                              norm_audio, dynaudnorm, or both. Example:
                              --normalizers canary-1b-v2=norm_audio
                              --normalizers qwen3=dynaudnorm
  --quiet-models CSV          For custom/added models, explicitly pass quiet chunking
                              to these models; default: canary-1b-v2. Use 'none' or 'all'.
                              Curated defaults keep their explicit 40s/100s settings.
  --internal-models CSV       Models that receive --internal-chunking; default: none.
                              Currently supported: cohere-transcribe-2b,moss-transcribe-diarize.
  --output-root DIR           Results root (default: ./transcribe-audio-results)
  --chunk-seconds-quiet SEC   Forward to every model, overriding per-model quiet defaults.
  --lang CODE                 Passed to every `transcribe` invocation (default: en).
  OTHER `transcribe` OPTIONS  Passed through to every selected model, e.g. --threads 8,
                              --beam-size 6, --no-vad, --whisper-size large-v3.
                              Unrecognized options are also forwarded for future compatibility;
  --model NAME                Convenience alias for a single model; conflicts with --models/--add-models.
  --output DIR / -o DIR       Maps to this runner's results root (each run still gets its own folder).
  --help                      Show this help

Examples:
  transcribe_audio lecture.mp3
  transcribe_audio --lang bg lecture.mp3
  transcribe_audio --add-models qwen3,granite-speech-5.0-turboctc lecture.mp3
  transcribe_audio --normalizers canary-1b-v2=both --add-models qwen3 lecture.mp3
  transcribe_audio --models faster-whisper,canary-1b-v2 lecture.mp3
  transcribe_audio --models granite-speech-5.0-turboctc, cohere-transcribe-2b,moss-transcribe-diarize,distil-whisper-large-v3.5 lecture.mp3
  transcribe_audio --models ark-asr-3b,moss-transcribe-preview-2b lecture.wav
  transcribe_audio --models cohere-transcribe-2b,moss-transcribe-diarize \
    --internal-models cohere-transcribe-2b,moss-transcribe-diarize lecture.wav

Each run has a separate output directory and log. Models are run one at a time to
limit RAM use. Unknown/uninstalled models may fail, but the script continues to
the next model and normalization variant. Arguments recognized by `transcribe`
are forwarded to each selected-model run. The runner reserves --models for a
sweep; --model selects one model when no --models option is given. It maps
`transcribe --output`/`-o` to the runner's results root so output folders remain
separate. Only local models supported by the installed `transcribe` command can
be selected; proprietary leaderboard API IDs are not downloadable model weights
and are not installed by this toolkit.
USAGE
}

LANG_CODE=en
LANG_EXPLICIT=0
MODELS_CSV=''
MODELS_EXPLICIT=0
ADD_MODELS_CSV=''
ADD_MODELS_EXPLICIT=0
NORMALIZER_OVERRIDE_MODELS=()
NORMALIZER_OVERRIDE_SPECS=()
SINGLE_MODEL=''
SINGLE_MODEL_SEEN=0
QUIET_MODELS_CSV='canary-1b-v2'
QUIET_MODELS_EXPLICIT=0
INTERNAL_MODELS_CSV='none'
QUIET_SECONDS=40
QUIET_SECONDS_EXPLICIT=0
GLOBAL_INTERNAL=0
EXPLICIT_OTHER_CHUNK_MODE=0
OUTPUT_ROOT="${TRANSCRIBE_AUDIO_OUTPUT_ROOT:-$PWD/transcribe-audio-results}"
DEFAULT_MODELS_CSV="${TRANSCRIBE_AUDIO_DEFAULT_MODELS-}"
DEFAULT_MODELS_CONFIGURED=0
[[ ${TRANSCRIBE_AUDIO_DEFAULT_MODELS+x} ]] && DEFAULT_MODELS_CONFIGURED=1
INPUT=""
TRANSCRIBE_EXTRA_ARGS=()

need_value() {
  local opt="$1"
  if (($# < 2)) || [[ -z "$2" ]]; then
    echo "ERROR: $opt needs a value." >&2
    exit 2
  fi
}
is_audio_candidate() {
  local candidate="$1"
  [[ -f "$candidate" ]] && return 0
  [[ "$candidate" =~ \.(wav|wave|mp3|m4a|m4b|aac|flac|ogg|oga|opus|wma|aif|aiff|caf|webm|mp4|mkv|mov|mpga|mpeg|3gp)$ ]]
}

while (($#)); do
  token="$1"
  case "$token" in
    --models|--models=*)
      if [[ "$token" == *=* ]]; then MODELS_CSV="${token#*=}"; shift
      else need_value "$token" "${2-}"; MODELS_CSV="$2"; shift 2
      fi
      MODELS_EXPLICIT=1
      # Also accept a space after commas when the model list is unquoted.
      while (($#)) && [[ "$1" != -* ]] && ! is_audio_candidate "$1"; do
        if [[ "$MODELS_CSV" == *, ]]; then MODELS_CSV+="$1"; else MODELS_CSV+=",$1"; fi
        shift
      done
      ;;
    --add-models|--add-models=*)
      if [[ "$token" == *=* ]]; then ADD_MODELS_CSV="${token#*=}"; shift
      else need_value "$token" "${2-}"; ADD_MODELS_CSV="$2"; shift 2
      fi
      ADD_MODELS_EXPLICIT=1
      while (($#)) && [[ "$1" != -* ]] && ! is_audio_candidate "$1"; do
        if [[ "$ADD_MODELS_CSV" == *, ]]; then ADD_MODELS_CSV+="$1"; else ADD_MODELS_CSV+=",$1"; fi
        shift
      done
      ;;
    --normalizers|--normalizers=*)
      if [[ "$token" == *=* ]]; then setting="${token#*=}"; shift
      else need_value "$token" "${2-}"; setting="$2"; shift 2; fi
      [[ "$setting" == *=* ]] || { echo "ERROR: --normalizers expects MODEL=norm_audio|dynaudnorm|both, got: $setting" >&2; exit 2; }
      norm_model="${setting%%=*}"
      norm_spec="${setting#*=}"
      norm_model="${norm_model#${norm_model%%[![:space:]]*}}"; norm_model="${norm_model%${norm_model##*[![:space:]]}}"
      norm_spec="${norm_spec#${norm_spec%%[![:space:]]*}}"; norm_spec="${norm_spec%${norm_spec##*[![:space:]]}}"
      [[ -n "$norm_model" ]] || { echo "ERROR: --normalizers model name cannot be empty." >&2; exit 2; }
      case "$norm_spec" in norm_audio|dynaudnorm|both) ;; *) echo "ERROR: invalid normalizer '$norm_spec'; use norm_audio, dynaudnorm, or both." >&2; exit 2 ;; esac
      NORMALIZER_OVERRIDE_MODELS+=("$norm_model")
      NORMALIZER_OVERRIDE_SPECS+=("$norm_spec")
      ;;
    --quiet-models|--quiet-models=*)
      if [[ "$token" == *=* ]]; then QUIET_MODELS_CSV="${token#*=}"; shift
      else need_value "$token" "${2-}"; QUIET_MODELS_CSV="$2"; shift 2; fi
      QUIET_MODELS_EXPLICIT=1 ;;
    --internal-models|--internal-models=*)
      if [[ "$token" == *=* ]]; then INTERNAL_MODELS_CSV="${token#*=}"; shift
      else need_value "$token" "${2-}"; INTERNAL_MODELS_CSV="$2"; shift 2; fi ;;
    --output-root|--output-root=*)
      if [[ "$token" == *=* ]]; then OUTPUT_ROOT="${token#*=}"; shift
      else need_value "$token" "${2-}"; OUTPUT_ROOT="$2"; shift 2; fi ;;
    --output|--output=*|-o)
      # A single-run transcribe output directory would collide across this
      # comparison's models/variants, so reinterpret it as the results root.
      if [[ "$token" == *=* ]]; then OUTPUT_ROOT="${token#*=}"; shift
      else need_value "$token" "${2-}"; OUTPUT_ROOT="$2"; shift 2; fi ;;
    --lang|--lang=*)
      if [[ "$token" == *=* ]]; then LANG_CODE="${token#*=}"; shift
      else need_value "$token" "${2-}"; LANG_CODE="$2"; shift 2; fi
      LANG_EXPLICIT=1 ;;
    --model|--model=*)
      if [[ "$token" == *=* ]]; then SINGLE_MODEL="${token#*=}"; shift
      else need_value "$token" "${2-}"; SINGLE_MODEL="$2"; shift 2; fi
      SINGLE_MODEL_SEEN=1 ;;
    --chunk-seconds-quiet|--chunk-seconds-quiet=*)
      if [[ "$token" == *=* ]]; then QUIET_SECONDS="${token#*=}"; TRANSCRIBE_EXTRA_ARGS+=(--chunk-seconds-quiet "$QUIET_SECONDS"); shift
      else need_value "$token" "${2-}"; QUIET_SECONDS="$2"; TRANSCRIBE_EXTRA_ARGS+=(--chunk-seconds-quiet "$2"); shift 2; fi
      QUIET_SECONDS_EXPLICIT=1 ;;
    --chunk-seconds|--chunk-seconds=*|--overlap-seconds|--overlap-seconds=*|--chunk-quiet-search-seconds|--chunk-quiet-search-seconds=*|--chunk-quiet-window-ms|--chunk-quiet-window-ms=*|--chunk-quiet-hop-ms|--chunk-quiet-hop-ms=*|--threads|--threads=*|--qwen-internal-chunk-seconds|--qwen-internal-chunk-seconds=*|--canary-internal-chunk-seconds|--canary-internal-chunk-seconds=*|--max-new-tokens|--max-new-tokens=*|--beam-size|--beam-size=*|--canary-beam-size|--canary-beam-size=*|--granite-beam-size|--granite-beam-size=*|--no-repeat-ngram-size|--no-repeat-ngram-size=*|--whisper-size|--whisper-size=*|--qwen-checkpoint-06|--qwen-checkpoint-06=*|--qwen-checkpoint-17|--qwen-checkpoint-17=*|--qwen-device|--qwen-device=*|--merge-max-words|--merge-max-words=*|--merge-max-chars|--merge-max-chars=*)
      opt="$token"
      if [[ "$token" == *=* ]]; then
        TRANSCRIBE_EXTRA_ARGS+=("${token%%=*}" "${token#*=}")
        [[ "${token%%=*}" == --chunk-seconds || "${token%%=*}" == --overlap-seconds ]] && EXPLICIT_OTHER_CHUNK_MODE=1
        shift
      else
        need_value "$token" "${2-}"
        TRANSCRIBE_EXTRA_ARGS+=("$token" "$2")
        [[ "$token" == --chunk-seconds || "$token" == --overlap-seconds ]] && EXPLICIT_OTHER_CHUNK_MODE=1
        shift 2
      fi ;;
    --overwrite-output|--internal-chunking|--no-previous-text-context|--previous-text-context|--vad|--no-vad)
      TRANSCRIBE_EXTRA_ARGS+=("$token")
      [[ "$token" == --internal-chunking ]] && GLOBAL_INTERNAL=1
      shift ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      while (($#)); do
        if [[ -z "$INPUT" ]]; then INPUT="$1"; else echo "ERROR: unexpected extra argument: $1" >&2; exit 2; fi
        shift
      done
      break ;;
    -* )
      # Keep the wrapper forward-compatible with any current or future public
      # `transcribe` option not otherwise reserved by this runner. Values joined
      # with '=' are split into the conventional two-argument form. Otherwise,
      # consume a following non-option token as the value unless it looks like
      # the input audio path; the child `transcribe` command remains the source
      # of truth for validating whether the option exists and its value.
      if [[ "$token" == *=* ]]; then
        TRANSCRIBE_EXTRA_ARGS+=("${token%%=*}" "${token#*=}")
        shift
      elif (($# > 1)) && [[ "$2" != -* ]] && ! is_audio_candidate "$2"; then
        TRANSCRIBE_EXTRA_ARGS+=("$token" "$2")
        shift 2
      else
        TRANSCRIBE_EXTRA_ARGS+=("$token")
        shift
      fi ;;
    *)
      if [[ -z "$INPUT" ]]; then INPUT="$token"
      else echo "ERROR: unexpected extra argument: $token" >&2; exit 2
      fi
      shift ;;
  esac
done

if (( SINGLE_MODEL_SEEN )); then
  if (( MODELS_EXPLICIT || ADD_MODELS_EXPLICIT )); then
    echo "ERROR: use --model NAME by itself, or combine --models/--add-models for a sweep." >&2
    exit 2
  fi
fi

[[ -n "$INPUT" ]] || { echo "ERROR: input audio is required." >&2; usage >&2; exit 2; }
[[ -f "$INPUT" ]] || { echo "ERROR: input audio not found: $INPUT" >&2; exit 2; }
[[ "$QUIET_SECONDS" =~ ^([2-9][0-9]|[1-9][0-9]{2,})([.][0-9]+)?$ ]] || {
  echo "ERROR: --chunk-seconds-quiet must be a number >= 20." >&2; exit 2;
}

# Parse CSVs while trimming whitespace around model names.
parse_csv() {
  local csv="$1" item
  local -a raw=()
  PARSED_CSV=()
  IFS=',' read -r -a raw <<< "$csv"
  for item in "${raw[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "$item" ]] && PARSED_CSV+=("$item")
  done
}

CLEAN_MODELS=()
ADDED_MODELS=()
if (( MODELS_EXPLICIT )); then
  parse_csv "$MODELS_CSV"; CLEAN_MODELS=("${PARSED_CSV[@]}")
  ((${#CLEAN_MODELS[@]})) || { echo "ERROR: --models did not contain any model names." >&2; exit 2; }
fi
if (( ADD_MODELS_EXPLICIT )); then
  parse_csv "$ADD_MODELS_CSV"; ADDED_MODELS=("${PARSED_CSV[@]}")
  ((${#ADDED_MODELS[@]})) || { echo "ERROR: --add-models did not contain any model names." >&2; exit 2; }
fi

if [[ "$QUIET_MODELS_CSV" == all ]]; then
  QUIET_MODE=all
elif [[ "$QUIET_MODELS_CSV" == none || -z "$QUIET_MODELS_CSV" ]]; then
  QUIET_MODE=none
else
  QUIET_MODE=list
  parse_csv "$QUIET_MODELS_CSV"; QUIET_MODELS=("${PARSED_CSV[@]}")
fi
is_quiet_model() {
  local model="$1" q
  [[ "$QUIET_MODE" == all ]] && return 0
  [[ "$QUIET_MODE" == none ]] && return 1
  for q in "${QUIET_MODELS[@]-}"; do [[ "$model" == "$q" ]] && return 0; done
  return 1
}

if [[ "$INTERNAL_MODELS_CSV" == none || -z "$INTERNAL_MODELS_CSV" ]]; then
  INTERNAL_MODELS=()
else
  parse_csv "$INTERNAL_MODELS_CSV"; INTERNAL_MODELS=("${PARSED_CSV[@]}")
fi
is_internal_model() {
  local model="$1" q
  for q in "${INTERNAL_MODELS[@]-}"; do [[ "$model" == "$q" ]] && return 0; done
  return 1
}

# Planning rows are model, normalization variant, and an optional quiet-chunk override.
# "default" leaves the normal transcribe defaults intact; --chunk-seconds-quiet can
# override the model defaults for every run.
PLAN_MODELS=()
PLAN_VARIANTS=()
PLAN_QUIETS=()
add_plan() {
  PLAN_MODELS+=("$1")
  PLAN_VARIANTS+=("$2")
  PLAN_QUIETS+=("$3")
}
add_both_variants() {
  local model="$1" quiet="$2"
  add_plan "$model" norm_audio "$quiet"
  add_plan "$model" dynaudnorm "$quiet"
}

if (( ! MODELS_EXPLICIT && ! SINGLE_MODEL_SEEN )); then
  if (( DEFAULT_MODELS_CONFIGURED )); then
    # Backward-compatible opt-in: a custom four-model sweep on both variants.
    parse_csv "$DEFAULT_MODELS_CSV"; DEFAULT_MODELS=("${PARSED_CSV[@]}")
    ((${#DEFAULT_MODELS[@]} == 4)) || {
      echo "ERROR: TRANSCRIBE_AUDIO_DEFAULT_MODELS must name exactly four comma-separated models (got ${#DEFAULT_MODELS[@]})." >&2
      exit 2
    }
    for ((i = 0; i < ${#DEFAULT_MODELS[@]}; i++)); do
      for ((j = 0; j < i; j++)); do
        if [[ "${DEFAULT_MODELS[$i]}" == "${DEFAULT_MODELS[$j]}" ]]; then
          echo "ERROR: TRANSCRIBE_AUDIO_DEFAULT_MODELS must contain four distinct models; duplicate: ${DEFAULT_MODELS[$i]}" >&2
          exit 2
        fi
      done
    done
    for model in "${DEFAULT_MODELS[@]}"; do
      quiet=default
      is_quiet_model "$model" && quiet="$QUIET_SECONDS"
      add_both_variants "$model" "$quiet"
    done
  else
    # Keep the default model/normalizer/chunk matrix explicit so optional models
    # enter the comparison only when requested.
    add_plan faster-whisper norm_audio 40
    add_plan faster-whisper dynaudnorm 40
    add_plan canary-1b-v2 norm_audio 40
    add_plan canary-1b-v2 norm_audio 100
  fi
elif (( SINGLE_MODEL_SEEN )); then
  quiet=default; is_quiet_model "$SINGLE_MODEL" && quiet="$QUIET_SECONDS"
  add_both_variants "$SINGLE_MODEL" "$quiet"
else
  for model in "${CLEAN_MODELS[@]}"; do
    quiet=default; is_quiet_model "$model" && quiet="$QUIET_SECONDS"
    add_both_variants "$model" "$quiet"
  done
fi

# --add-models is additive by design: it retains the curated default configurations
# (or the list supplied with --models), then adds each requested model on both methods.
for model in "${ADDED_MODELS[@]}"; do
  quiet=default; is_quiet_model "$model" && quiet="$QUIET_SECONDS"
  add_both_variants "$model" "$quiet"
done

# Resolve each model's normalization preference. Without an override, each planned
# winning run keeps its own variant; custom/added models default to both variants.
normalizer_override_for() {
  local model="$1" i
  NORMALIZER_OVERRIDE_RESULT=""
  for i in "${!NORMALIZER_OVERRIDE_MODELS[@]}"; do
    if [[ "${NORMALIZER_OVERRIDE_MODELS[$i]}" == "$model" ]]; then
      NORMALIZER_OVERRIDE_RESULT="${NORMALIZER_OVERRIDE_SPECS[$i]}"
    fi
  done
}

# Create the final plan and deduplicate identical configurations (e.g. if --chunk-
# seconds-quiet 40 makes the default 40s and 100s Canary entries identical).
RUN_MODELS=()
RUN_VARIANTS=()
RUN_QUIETS=()
RUN_TAGS=()
SEEN_RUN_KEYS='|'
for i in "${!PLAN_MODELS[@]}"; do
  model="${PLAN_MODELS[$i]}"
  variant="${PLAN_VARIANTS[$i]}"
  planned_quiet="${PLAN_QUIETS[$i]}"
  normalizer_override_for "$model"
  variants=("$variant")
  case "$NORMALIZER_OVERRIDE_RESULT" in
    norm_audio) variants=(norm_audio) ;;
    dynaudnorm) variants=(dynaudnorm) ;;
    both) variants=(norm_audio dynaudnorm) ;;
    "") ;;
  esac

  quiet="$planned_quiet"
  if (( QUIET_SECONDS_EXPLICIT )); then
    quiet="$QUIET_SECONDS"
  elif (( GLOBAL_INTERNAL || EXPLICIT_OTHER_CHUNK_MODE )); then
    quiet=none
  elif [[ "$quiet" == default ]] && is_quiet_model "$model"; then
    quiet="$QUIET_SECONDS"
  fi
  # A model explicitly selected for native chunking shouldn't also inherit a
  # preset external quiet-chunk value unless the user explicitly requested both.
  if (( ! QUIET_SECONDS_EXPLICIT )) && is_internal_model "$model"; then quiet=none; fi

  # Normalize the dedupe key against transcribe's implicit quiet-mode defaults.
  # Most models default to 40s; ARK uses 25s; legacy Canary-1B/Flash uses its
  # historical mode. This avoids duplicate work when --add-models repeats a winner.
  if [[ "$quiet" == default ]]; then
    case "$model" in
      canary|canary-1b|canary-1b-flash) quiet_identity=legacy-default ;;
      ark-asr-3b) quiet_identity=25 ;;
      *) quiet_identity=40 ;;
    esac
  else
    quiet_identity="$quiet"
  fi

  for variant in "${variants[@]}"; do
    key="$model|$variant|$quiet_identity|$GLOBAL_INTERNAL|$EXPLICIT_OTHER_CHUNK_MODE"
    [[ "$SEEN_RUN_KEYS" == *"|$key|"* ]] && continue
    SEEN_RUN_KEYS+="$key|"
    RUN_MODELS+=("$model")
    RUN_VARIANTS+=("$variant")
    RUN_QUIETS+=("$quiet")
    if [[ "$quiet" != none && "$quiet" != default ]]; then
      qtag="${quiet//./p}"
      RUN_TAGS+=("quiet${qtag}")
    elif (( GLOBAL_INTERNAL )) || is_internal_model "$model"; then
      RUN_TAGS+=(internal)
    elif (( EXPLICIT_OTHER_CHUNK_MODE )); then
      RUN_TAGS+=(chunk)
    elif [[ "$quiet" == default ]]; then
      # Label a run as default when no per-run override is passed, even when
      # transcribe internally resolves its own default to a particular chunk size.
      RUN_TAGS+=(default)
    else
      RUN_TAGS+=(default)
    fi
  done
done
((${#RUN_MODELS[@]})) || { echo "ERROR: no runs planned." >&2; exit 2; }

NEED_NORM_AUDIO=0
NEED_DYNAUDNORM=0
for variant in "${RUN_VARIANTS[@]}"; do
  [[ "$variant" == norm_audio ]] && NEED_NORM_AUDIO=1
  [[ "$variant" == dynaudnorm ]] && NEED_DYNAUDNORM=1
done
PREPROCESS_TOTAL=$((NEED_NORM_AUDIO + NEED_DYNAUDNORM))

for cmd in transcribe; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: '$cmd' is not in PATH. Run install.sh and ensure ~/.local/bin is on PATH." >&2
    exit 127
  fi
done
if (( NEED_NORM_AUDIO )) && ! command -v norm_audio >/dev/null 2>&1; then
  echo "ERROR: 'norm_audio' is not in PATH. Run install.sh and ensure ~/.local/bin is on PATH." >&2; exit 127
fi
if (( NEED_DYNAUDNORM )) && ! command -v dynamic_norm_audio >/dev/null 2>&1; then
  echo "ERROR: 'dynamic_norm_audio' is not in PATH. Run install.sh and ensure ~/.local/bin is on PATH." >&2; exit 127
fi

INPUT_ABS=$(realpath -- "$INPUT")
BASE=$(basename -- "$INPUT_ABS")
BASE=${BASE%.*}
LANG_SAFE=${LANG_CODE//[^a-zA-Z0-9_-]/_}
mkdir -p -- "$OUTPUT_ROOT" || exit 1
OUT_ROOT=$(realpath -- "$OUTPUT_ROOT")
OUT_DIR=$(mktemp -d "$OUT_ROOT/${BASE}_${LANG_SAFE}.XXXXXX") || exit 1
AUDIO_DIR="$OUT_DIR/preprocessed"
mkdir -p -- "$AUDIO_DIR"
printf 'Input:   %s\nLanguage: %s\nResults: %s\n' "$INPUT_ABS" "$LANG_CODE" "$OUT_DIR"
printf 'Planned runs:\n'
for i in "${!RUN_MODELS[@]}"; do
  printf '  %d. %-34s %-12s chunk=%s\n' "$((i + 1))" "${RUN_MODELS[$i]}" "${RUN_VARIANTS[$i]}" "${RUN_QUIETS[$i]}"
done

VARIANT_NAMES=(norm_audio dynaudnorm)
VARIANT_PATHS=("$AUDIO_DIR/${BASE}_norm.wav" "$AUDIO_DIR/${BASE}_dynaudnorm.wav")
VARIANT_CMDS=(norm_audio dynamic_norm_audio)
VARIANT_NEEDED=($NEED_NORM_AUDIO $NEED_DYNAUDNORM)
PREPROCESS_OK=0
FAILED=0
SUCCEEDED=0
run_preprocess_variant() {
  local variant="$1" cmd="$2" wav="$3" native_wav status
  if [[ "$variant" == "norm_audio" ]]; then
    # Preserve standalone norm_audio's 44.1 kHz Audacity-style
    # normalization, then perform the same FFmpeg conversion transcribe used to
    # perform on this file at inference time. This gives both variants an
    # identical mono/16 kHz/PCM16 contract without changing the loudness meter's
    # sample rate or resampling the normalized signal twice.
    native_wav=$(mktemp --tmpdir="$AUDIO_DIR" --suffix=.wav ".${BASE}_norm_native.XXXXXX") || return 1
    if "$cmd" "$INPUT_ABS" -o "$native_wav"; then
      printf 'RUN: ffmpeg -hide_banner -loglevel error -y -i %q -vn -ac 1 -ar 16000 -c:a pcm_s16le %q\n' "$native_wav" "$wav"
      if ffmpeg -hide_banner -loglevel error -y -i "$native_wav" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$wav"; then
        status=0
      else
        status=$?
      fi
    else
      status=$?
    fi
    rm -f -- "$native_wav"
    if (( status != 0 )); then
      rm -f -- "$wav"
      return "$status"
    fi
    [[ -s "$wav" ]]
  else
    "$cmd" "$INPUT_ABS" -o "$wav" && [[ -s "$wav" ]]
  fi
}

for i in 0 1; do
  (( ${VARIANT_NEEDED[$i]} )) || continue
  variant=${VARIANT_NAMES[$i]}
  wav=${VARIANT_PATHS[$i]}
  cmd=${VARIANT_CMDS[$i]}
  plog="$OUT_DIR/${variant}.preprocess.log"
  printf '\n========== Preprocess: %s =========\n' "$variant"
  if run_preprocess_variant "$variant" "$cmd" "$wav" >"$plog" 2>&1; then
    printf 'PASS — %s\n' "$wav"
    PREPROCESS_OK=$((PREPROCESS_OK + 1))
  else
    status=$?
    printf 'FAIL — %s (exit %s). Log: %s\n' "$variant" "$status" "$plog" >&2
    tail -n 20 "$plog" >&2 2>/dev/null || true
    FAILED=$((FAILED + 1))
  fi
done

for i in "${!RUN_MODELS[@]}"; do
  model="${RUN_MODELS[$i]}"
  variant="${RUN_VARIANTS[$i]}"
  quiet="${RUN_QUIETS[$i]}"
  tag="${RUN_TAGS[$i]}"
  if [[ "$variant" == norm_audio ]]; then vindex=0; else vindex=1; fi
  wav="${VARIANT_PATHS[$vindex]}"
  if [[ ! -s "$wav" ]]; then
    printf '\n========== Transcribe: %s / %s =========\n' "$variant" "$model"
    printf 'SKIP — preprocessing output missing for %s; continuing.\n' "$variant" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  model_safe=${model//[^a-zA-Z0-9_.-]/_}
  run_dir="$OUT_DIR/$variant/${model_safe}--${tag}"
  log_file="$OUT_DIR/${variant}.${model_safe}--${tag}.log"
  mkdir -p -- "$(dirname -- "$run_dir")"
  # Keep underlying transcribe defaults intact unless the caller explicitly
  # changes language or model settings. faster-whisper remains the CLI default,
  # so its model flag is omitted; all other backends are selected explicitly.
  args=("$wav" --output "$run_dir" "${TRANSCRIBE_EXTRA_ARGS[@]}")
  if (( LANG_EXPLICIT )) || [[ "$LANG_CODE" != en ]]; then args+=(--lang "$LANG_CODE"); fi
  # The sweep owns model selection; faster-whisper is also transcribe's default.
  if [[ "$model" != faster-whisper ]]; then args+=(--model "$model"); fi
  if [[ "$quiet" != none && "$quiet" != default ]]; then args+=(--chunk-seconds-quiet "$quiet"); fi
  # The custom-model internal selector remains model-scoped.
  if (( ! GLOBAL_INTERNAL )) && is_internal_model "$model"; then
    case "$model" in
      cohere-transcribe-2b|moss-transcribe-diarize) args+=(--internal-chunking) ;;
      *) echo "WARN: --internal-models does not support '$model'; skipping native mode for this model." >&2 ;;
    esac
  fi

  printf '\n========== Transcribe: %s / %s (chunk=%s) =========\n' "$variant" "$model" "$quiet"
  printf 'Log: %s\n' "$log_file"
  if transcribe "${args[@]}" >"$log_file" 2>&1; then
    if [[ -d "$run_dir" ]] && find "$run_dir" -maxdepth 2 -type f \( -name '*.txt' -o -name 'run.json' \) -size +0c -print -quit | grep -q .; then
      printf 'PASS — output: %s\n' "$run_dir"
      SUCCEEDED=$((SUCCEEDED + 1))
    else
      printf 'WARN — command returned success but expected output was not found in %s. See %s\n' "$run_dir" "$log_file" >&2
      tail -n 20 "$log_file" >&2 2>/dev/null || true
      FAILED=$((FAILED + 1))
    fi
  else
    status=$?
    printf 'FAIL — %s / %s (exit %s). Continuing with the next run.\n' "$variant" "$model" "$status" >&2
    tail -n 20 "$log_file" >&2 2>/dev/null || true
    FAILED=$((FAILED + 1))
  fi
done

printf '\n========== SUMMARY =========\nPreprocessing variants succeeded: %d/%d\nTranscriptions succeeded: %d\nRuns failed/unavailable: %d\nResults directory: %s\n' \
  "$PREPROCESS_OK" "$PREPROCESS_TOTAL" "$SUCCEEDED" "$FAILED" "$OUT_DIR"
printf 'To compare transcript outputs visually, open transcript_railway_standalone.html and add the relevant run folders.\n'
if (( SUCCEEDED == 0 )); then exit 1; fi
exit 0
