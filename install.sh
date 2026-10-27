#!/usr/bin/env bash
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
# Install the local speech-transcription toolkit on Manjaro/Arch Linux.
# Isolated Python 3.12 environments; Intel XPU only for Qwen when available; all others CPU-only.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$HOME/.local/share/speech-toolkit}"
USER_BIN="$HOME/.local/bin"
JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
if (( JOBS > 8 )); then JOBS=8; fi
if (( JOBS < 1 )); then JOBS=1; fi
SKIP_MODEL_DOWNLOAD=0
INCLUDE_GRANITE_8B=0
INCLUDE_NONCOMMERCIAL_MODELS=0
QWEN_DEVICE_REQUEST=auto

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --skip-model-download   Install code/environments, but do not prefetch large model weights.
  --include-granite-8b    Also prefetch IBM Granite Speech 3.3 8B weights (large; opt-in).
  --include-noncommercial-models  Also prefetch Canary-1B and SeamlessM4T v2 weights
                          (CC BY-NC 4.0; noncommercial use only).
  --qwen-device MODE      Qwen device: auto (default), xpu, or cpu. XPU is Intel GPU only.
                          If automatic XPU setup falls back to CPU, use xpu to retry deliberately.
  --help                  Show this help.

Environment overrides:
  APP_DIR=/path           Installation directory (default ~/.local/share/speech-toolkit)
  BUILD_JOBS=N            whisper.cpp compilation jobs (default min(nproc, 8))

The installer reuses existing model caches, virtual environments, package wheels, and working whisper.cpp builds. Qwen uses native PyTorch Intel XPU when available (or CPU); all other backends are CPU-only. No CUDA/NVIDIA Python packages, IPEX, or oneCCL are installed. It also attempts Granite Speech 5.0 TurboCTC, MOSS-Transcribe-Diarize, Cohere Transcribe, Distil-Whisper Large-v3.5, ARK-ASR-3B, and MOSS-Transcribe-preview-2B in isolated CPU environments. Canary-1B and SeamlessM4T v2 weights are not downloaded by default because their model weights are licensed CC BY-NC 4.0; opt in explicitly with --include-noncommercial-models after reviewing THIRD_PARTY_NOTICES.md. Each optional package/model download failure is recorded in state/optional-asr-install-status.txt and does not block the next candidate. Model downloads occur during installation; transcription is local-only and never calls hosted inference APIs. The eight proprietary leaderboard services do not provide downloadable local weights and are deliberately excluded.
EOF
}
while (($#)); do
  case "$1" in
    --skip-model-download) SKIP_MODEL_DOWNLOAD=1; shift ;;
    --include-granite-8b) INCLUDE_GRANITE_8B=1; shift ;;
    --include-noncommercial-models) INCLUDE_NONCOMMERCIAL_MODELS=1; shift ;;
    --qwen-device)
      [[ $# -ge 2 ]] || { echo "--qwen-device needs auto, xpu, or cpu" >&2; exit 2; }
      QWEN_DEVICE_REQUEST="$2"; shift 2 ;;
    --qwen-device=*) QWEN_DEVICE_REQUEST="${1#*=}"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "ERROR: Run as your normal user, not root. The script uses sudo only for pacman." >&2
  exit 1
fi

[[ "$QWEN_DEVICE_REQUEST" == auto || "$QWEN_DEVICE_REQUEST" == xpu || "$QWEN_DEVICE_REQUEST" == cpu ]] || { echo "ERROR: --qwen-device must be auto, xpu, or cpu." >&2; exit 2; }
required_files=(
  norm_audio.py dynamic_norm_audio.sh transcribe_audio.sh download_optional_asr_models.sh
  transcribe.py memory_guard.py compare-transcribers.sh compare-transcribers-long.sh
  transcript_railway_standalone.html README_TRANSCRIPT_RAILWAY_STANDALONE.md
  THIRD_PARTY_NOTICES.md
)
for file in "${required_files[@]}"; do
  if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
    echo "ERROR: required file is missing beside install.sh: $file" >&2
    exit 1
  fi
done

if ! command -v pacman >/dev/null 2>&1; then
  echo "ERROR: This installer targets Manjaro/Arch Linux (pacman not found)." >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

log "Check system prerequisites (do not run a full upgrade on every rerun)"
BASE_PACKAGES=(base-devel cmake curl git wget ffmpeg libsndfile python python-pip rust)
INTEL_GPU=0
for vendor_file in /sys/class/drm/card*/device/vendor; do
  [[ -r "$vendor_file" ]] || continue
  if [[ "$(<"$vendor_file")" == "0x8086" ]]; then INTEL_GPU=1; break; fi
done
if [[ "$QWEN_DEVICE_REQUEST" != cpu && $INTEL_GPU -eq 1 ]]; then
  BASE_PACKAGES+=(intel-compute-runtime)
fi
MISSING_PACKAGES=()
for pkg in "${BASE_PACKAGES[@]}"; do
  if ! pacman -Q "$pkg" >/dev/null 2>&1; then MISSING_PACKAGES+=("$pkg"); fi
done
if ((${#MISSING_PACKAGES[@]})); then
  printf 'Missing system packages: %s\n' "${MISSING_PACKAGES[*]}"
  # -Syu avoids an unsupported partial-upgrade state; invoke it only if something is missing.
  sudo pacman -Syu --needed --noconfirm "${MISSING_PACKAGES[@]}"
else
  echo "All required system packages are already installed; skipping pacman."
fi

if ! command -v uv >/dev/null 2>&1; then
  log "Install uv with its official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv could not be installed or is not on PATH."

log "Install a managed Python 3.12 (do not modify Manjaro's system Python)"
if ! uv python install 3.12; then
  echo "WARN: uv did not accept the existing Python install; validating the interpreter and forcing a repair if needed." >&2
fi
# Venv bin/python is often a symlink to uv's managed interpreter. If a user
# accidentally pastes a logged command beginning with a shell redirection `>`
# and targets venv/bin/python, the shared interpreter may be truncated. Detect
# this before creating/reusing environments and repair the managed interpreter
# in place; existing model weights and package environments are otherwise kept.
UV_PYTHON_BIN="$(uv python find 3.12 2>/dev/null || true)"
if [[ -z "$UV_PYTHON_BIN" ]] || ! "$UV_PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12); assert sys.implementation.name == "cpython"' >/dev/null 2>&1; then
  echo "WARN: uv-managed Python 3.12 did not execute correctly; reinstalling its interpreter (existing model weights are untouched)." >&2
  uv python install --reinstall --force 3.12
  UV_PYTHON_BIN="$(uv python find 3.12 2>/dev/null || true)"
fi
[[ -n "$UV_PYTHON_BIN" ]] || die "uv could not resolve a usable managed Python 3.12 after repair."
"$UV_PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12); print("Managed Python OK:", sys.executable, sys.version.split()[0])' || die "uv-managed Python 3.12 remains unusable after repair."

mkdir -p "$APP_DIR" "$APP_DIR/venvs" "$APP_DIR/models" "$APP_DIR/bin" "$APP_DIR/state" "$USER_BIN"
QWEN_XPU_FAILURE_MARKER="$APP_DIR/state/qwen-xpu-unavailable.txt"
cp "$SCRIPT_DIR/norm_audio.py" "$APP_DIR/norm_audio.py"
cp "$SCRIPT_DIR/dynamic_norm_audio.sh" "$APP_DIR/dynamic_norm_audio.sh"
cp "$SCRIPT_DIR/transcribe_audio.sh" "$APP_DIR/transcribe_audio.sh"
cp "$SCRIPT_DIR/download_optional_asr_models.sh" "$APP_DIR/download_optional_asr_models.sh"
cp "$SCRIPT_DIR/transcribe.py" "$APP_DIR/transcribe.py"
cp "$SCRIPT_DIR/memory_guard.py" "$APP_DIR/memory_guard.py"
cp "$SCRIPT_DIR/compare-transcribers.sh" "$APP_DIR/compare-transcribers.sh"
cp "$SCRIPT_DIR/compare-transcribers-long.sh" "$APP_DIR/compare-transcribers-long.sh"
cp "$SCRIPT_DIR/transcript_railway_standalone.html" "$APP_DIR/transcript_railway_standalone.html"
cp "$SCRIPT_DIR/README.md" "$APP_DIR/README.md"
cp "$SCRIPT_DIR/README_TRANSCRIPT_RAILWAY_STANDALONE.md" "$APP_DIR/README_TRANSCRIPT_RAILWAY_STANDALONE.md"
cp "$SCRIPT_DIR/THIRD_PARTY_NOTICES.md" "$APP_DIR/THIRD_PARTY_NOTICES.md"
cp "$SCRIPT_DIR/LICENSE" "$APP_DIR/LICENSE"
chmod 0755 "$APP_DIR/memory_guard.py" "$APP_DIR/compare-transcribers.sh" "$APP_DIR/compare-transcribers-long.sh" "$APP_DIR/dynamic_norm_audio.sh" "$APP_DIR/transcribe_audio.sh" "$APP_DIR/download_optional_asr_models.sh"
cp "$SCRIPT_DIR/install.sh" "$APP_DIR/install.sh"

make_venv() {
  local name="$1"
  local path="$APP_DIR/venvs/$name"
  if [[ -x "$path/bin/python" ]] && "$path/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
    echo "Reusing existing Python 3.12 environment: $path" >&2
  else
    if [[ -e "$path" || -L "$path" ]]; then
      local backup="$path.backup.$(date +%Y%m%d-%H%M%S)"
      echo "Existing environment is invalid; preserving it as $backup" >&2
      mv -- "$path" "$backup"
    fi
    uv venv --python 3.12 --seed "$path" >&2
  fi
  printf '%s\n' "$path/bin/python"
}

backup_venv() {
  local name="$1"
  local path="$APP_DIR/venvs/$name"
  local backup stamp
  [[ -e "$path" || -L "$path" ]] || return 0
  stamp="$(date +%Y%m%d-%H%M%S)"
  backup="$path.backup.$stamp"
  if [[ -e "$backup" || -L "$backup" ]]; then backup="$backup.$RANDOM"; fi
  echo "Preserving incompatible $name environment as $backup (model/cache directories are untouched)." >&2
  mv -- "$path" "$backup"
}

forbidden_python_packages() {
  local py="$1"
  "$py" - <<'PY'
import importlib.metadata as md, re
bad=[]
for d in md.distributions():
    name=re.sub(r"[-_.]+", "-", d.metadata.get("Name", "")).lower()
    if name in {"cuda", "cuda-python", "cuda-bindings", "cuda-pathfinder", "pytorch-cuda", "numba-cuda"} or name.startswith(("nvidia-", "cuda-")):
        bad.append(name)
print("\n".join(sorted(set(bad))))
PY
}

remove_forbidden_python_packages() {
  local py="$1" names
  names="$(forbidden_python_packages "$py" || true)"
  if [[ -n "$names" ]]; then
    echo "Removing NVIDIA/CUDA package artifacts from this isolated env: ${names//$'\n'/, }" >&2
    while IFS= read -r name; do
      [[ -n "$name" ]] && "$py" -m pip uninstall -y "$name" >/dev/null || true
    done <<< "$names"
  fi
}

assert_no_forbidden_python_packages() {
  local py="$1" names
  names="$(forbidden_python_packages "$py")"
  [[ -z "$names" ]] || { echo "ERROR: NVIDIA/CUDA Python packages remain in $py: ${names//$'\n'/, }" >&2; return 1; }
}

log "Create shared tools environment for audio normalization"
TOOL_PY="$(make_venv tools)"
if ! "$TOOL_PY" -c 'import numpy, soundfile, pyloudnorm' >/dev/null 2>&1; then
  uv pip install --python "$TOOL_PY" numpy soundfile pyloudnorm
fi
"$TOOL_PY" -m pip check

install_cpu_torch() {
  local py="$1"
  if "$py" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md, torch, torchaudio, re
assert torch.version.cuda is None and not torch.cuda.is_available()
assert "+xpu" not in torch.__version__.lower(), f"XPU build in CPU-only environment: {torch.__version__}"
assert "+xpu" not in torchaudio.__version__.lower(), f"XPU TorchAudio in CPU-only environment: {torchaudio.__version__}"
bad=[]
for d in md.distributions():
    n=re.sub(r"[-_.]+", "-", d.metadata.get("Name", "")).lower()
    if n in {"cuda", "cuda-python", "cuda-bindings", "cuda-pathfinder", "pytorch-cuda", "numba-cuda"} or n.startswith(("nvidia-", "cuda-")): bad.append(n)
assert not bad, bad
PY
  then
    echo "Reusing existing CPU-only Torch/TorchAudio in $py"
  else
    echo "Installing CPU-only Torch/TorchAudio (this env only)"
    # --reinstall is important when converting a partially installed XPU environment
    # back to CPU after an Intel runtime probe fails.
    uv pip install --reinstall --python "$py" --index-url https://download.pytorch.org/whl/cpu torch torchaudio || return $?
  fi
  "$py" - <<'PY' || return $?
import torch, torchaudio
assert torch.version.cuda is None and not torch.cuda.is_available(), f"Not CPU-only: torch={torch.__version__}"
assert "+xpu" not in torch.__version__.lower(), f"Expected CPU Torch, got {torch.__version__}"
assert "+xpu" not in torchaudio.__version__.lower(), f"Expected CPU TorchAudio, got {torchaudio.__version__}"
print("CPU Torch/TorchAudio verified:", torch.__version__, torchaudio.__version__)
PY
  assert_no_forbidden_python_packages "$py" || return $?
  "$py" - <<'PY' > "${py%/bin/python}/torch-constraints.txt" || return $?
import torch, torchaudio
print(f"torch=={torch.__version__}")
print(f"torchaudio=={torchaudio.__version__}")
PY
}

install_xpu_torch() {
  local py="$1"
  if "$py" - <<'PY' >/dev/null 2>&1
import torch, torchaudio, re, importlib.metadata as md
assert "+xpu" in torch.__version__.lower() and "+xpu" in torchaudio.__version__.lower()
# Pin to the known-matched native XPU 2.8.0 Torch/TorchAudio wheel pair.
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.version.cuda is None and not torch.cuda.is_available()
assert hasattr(torch, "xpu") and torch.xpu.is_available()
bad=[]
for d in md.distributions():
    n=re.sub(r"[-_.]+", "-", d.metadata.get("Name", "")).lower()
    if n in {"cuda", "cuda-python", "cuda-bindings", "cuda-pathfinder", "pytorch-cuda", "numba-cuda"} or n.startswith(("nvidia-", "cuda-")): bad.append(n)
assert not bad, bad
PY
  then
    echo "Reusing existing Intel-XPU Torch/TorchAudio 2.8.0 in $py"
  else
    echo "Installing native Intel-XPU Torch/TorchAudio 2.8.0 for Qwen (no CUDA, no IPEX)"
    # Qwen only needs torch + torchaudio. Do not download torchvision, IPEX, or
    # oneCCL: they are not runtime requirements for this ASR path. Native XPU
    # support is the preferred route; IPEX was archived in March 2026.
    uv pip install --reinstall --python "$py" --index-url https://download.pytorch.org/whl/xpu \
      'torch==2.8.0' 'torchaudio==2.8.0' || return $?
  fi
  "$py" - <<'PY' || return $?
import torch, torchaudio
assert "+xpu" in torch.__version__.lower() and "+xpu" in torchaudio.__version__.lower(), (torch.__version__, torchaudio.__version__)
assert torch.version.cuda is None and not torch.cuda.is_available()
available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
print("Intel-XPU wheel pair verified:", torch.__version__, torchaudio.__version__)
print("torch.xpu.is_available():", available)
if not available:
    raise SystemExit("XPU wheels installed but Intel XPU runtime/device is unavailable")
print("XPU device:", torch.xpu.get_device_name(0))
PY
  assert_no_forbidden_python_packages "$py" || return $?
}

log "Install Qwen3-ASR in an isolated Python 3.12 environment"
QWEN_PY="$(make_venv qwen)"
remove_forbidden_python_packages "$QWEN_PY"
QWEN_DEVICE_EFFECTIVE="$QWEN_DEVICE_REQUEST"
if [[ "$QWEN_DEVICE_EFFECTIVE" == auto ]]; then
  if [[ -f "$QWEN_XPU_FAILURE_MARKER" ]]; then
    QWEN_DEVICE_EFFECTIVE=cpu
    echo "A prior automatic XPU probe failed ($(cat "$QWEN_XPU_FAILURE_MARKER")); skipping another large XPU-wheel attempt. Use --qwen-device xpu to retry deliberately." >&2
  elif "$QWEN_PY" - <<'PY' >/dev/null 2>&1
import torch, torchaudio
assert "+xpu" in torch.__version__.lower() and "+xpu" in torchaudio.__version__.lower()
assert hasattr(torch, "xpu") and torch.xpu.is_available()
PY
  then QWEN_DEVICE_EFFECTIVE=xpu
  elif [[ $INTEL_GPU -eq 1 && -e /dev/dri ]]; then QWEN_DEVICE_EFFECTIVE=xpu
  else QWEN_DEVICE_EFFECTIVE=cpu
  fi
fi
if [[ "$QWEN_DEVICE_EFFECTIVE" == xpu ]]; then
  if ! install_xpu_torch "$QWEN_PY"; then
    if [[ "$QWEN_DEVICE_REQUEST" == auto ]]; then
      echo "WARN: XPU Torch install/validation failed; retrying Qwen on CPU. Model files are preserved." >&2
      printf '%s\n' "$(date -Is) XPU wheel/runtime unavailable; switched to CPU" > "$QWEN_XPU_FAILURE_MARKER"
      install_cpu_torch "$QWEN_PY"; QWEN_DEVICE_EFFECTIVE=cpu
    else die "Could not install/validate Intel XPU Torch. Use --qwen-device auto or --qwen-device cpu to continue."
    fi
  else
    rm -f -- "$QWEN_XPU_FAILURE_MARKER"
  fi
else
  install_cpu_torch "$QWEN_PY"
fi
remove_forbidden_python_packages "$QWEN_PY"
if ! "$QWEN_PY" -c 'from qwen_asr import Qwen3ASRModel' >/dev/null 2>&1; then
  uv pip install --python "$QWEN_PY" qwen-asr
else echo "qwen-asr already imports successfully; skipping reinstall."
fi
"$QWEN_PY" - <<'PY'
import torch
from qwen_asr import Qwen3ASRModel
assert torch.version.cuda is None and not torch.cuda.is_available()
print("Qwen3-ASR import OK; Torch", torch.__version__, "XPU available:", bool(hasattr(torch, "xpu") and torch.xpu.is_available()))
PY
assert_no_forbidden_python_packages "$QWEN_PY"
"$QWEN_PY" -m pip check
printf '%s\n' "$QWEN_DEVICE_EFFECTIVE" > "$APP_DIR/venvs/qwen/device.txt"
echo "Qwen device mode configured as: $QWEN_DEVICE_EFFECTIVE (runtime may fall back on CPU)"

log "Install Canary models with a compatible CPU-only NeMo/PyTorch stack"
# NeMo 3.x can pull unsupported CUDA bindings into this CPU-only environment.
# Preserve an incompatible environment and create
# a clean environment. The model caches live outside the venv and are untouched.
CANARY_ENV="$APP_DIR/venvs/canary"
canary_env_healthy_for_pin() {
  local py="$CANARY_ENV/bin/python"
  [[ -x "$py" ]] || return 0
  "$py" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md, re
packages = {re.sub(r"[-_.]+", "-", d.metadata.get("Name", "")).lower(): d.version
            for d in md.distributions()}
bad = [n for n in packages if n in {"cuda", "cuda-python", "cuda-bindings", "cuda-pathfinder", "pytorch-cuda", "numba-cuda"}
       or n.startswith(("nvidia-", "cuda-"))]
if bad:
    raise SystemExit(1)
nemo = packages.get("nemo-toolkit")
if nemo and nemo != "2.5.3":
    raise SystemExit(1)
transformers = packages.get("transformers")
if transformers and transformers.split(".", 1)[0] != "4":
    raise SystemExit(1)
# NeMo 2.5.3 targets the PyTorch 2.6 generation; keep that dependency pair pinned.
torch_version = packages.get("torch")
if torch_version and not torch_version.startswith("2.6.0"):
    raise SystemExit(1)
if nemo:
    import torch
    import nemo.collections.asr
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise SystemExit(1)
PY
}
if [[ -x "$CANARY_ENV/bin/python" ]] && ! canary_env_healthy_for_pin; then
  backup_venv canary
fi
CANARY_PY="$(make_venv canary)"
CANARY_AVAILABLE=0
install_canary_torch() {
  local py="$1"
  if ! "$py" - <<'PY' >/dev/null 2>&1
import numpy
assert numpy.__version__ == "1.26.4", numpy.__version__
PY
  then
    echo "Ensuring Canary's pinned NumPy 1.26.4 (without touching valid Torch wheels)."
    uv pip install --python "$py" 'numpy==1.26.4' || return $?
  else
    echo "Reusing Canary NumPy 1.26.4."
  fi
  if "$py" - <<'PY' >/dev/null 2>&1
import re, importlib.metadata as md, torch, torchaudio
assert torch.__version__.split("+", 1)[0] == "2.6.0", torch.__version__
assert torchaudio.__version__.split("+", 1)[0] == "2.6.0", torchaudio.__version__
assert torch.version.cuda is None and not torch.cuda.is_available()
assert "+xpu" not in torch.__version__.lower() and "+xpu" not in torchaudio.__version__.lower()
bad=[]
for d in md.distributions():
    n=re.sub(r"[-_.]+", "-", d.metadata.get("Name", "")).lower()
    if n in {"cuda", "cuda-python", "cuda-bindings", "cuda-pathfinder", "pytorch-cuda", "numba-cuda"} or n.startswith(("nvidia-", "cuda-")): bad.append(n)
assert not bad, bad
PY
  then
    echo "Reusing compatible CPU Torch/TorchAudio 2.6.0 in Canary environment."
  else
    # NeMo 2.5.x is pinned to the Torch 2.6 era; do not float to NeMo 3.x,
    # which installed cuda-bindings/cuda-pathfinder in the earlier attempt.
    uv pip install --reinstall --python "$py" --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.6.0' 'torchaudio==2.6.0' || return $?
  fi
  "$py" - <<'PY' > "${py%/bin/python}/torch-constraints.txt" || return $?
import torch, torchaudio
assert torch.__version__.split("+", 1)[0] == "2.6.0"
assert torchaudio.__version__.split("+", 1)[0] == "2.6.0"
assert torch.version.cuda is None and not torch.cuda.is_available()
assert "+xpu" not in torch.__version__.lower() and "+xpu" not in torchaudio.__version__.lower()
print(f"torch=={torch.__version__}")
print(f"torchaudio=={torchaudio.__version__}")
PY
  assert_no_forbidden_python_packages "$py" || return $?
}

install_canary() {
  install_canary_torch "$CANARY_PY" || return $?
  if ! "$CANARY_PY" -c 'import nemo.collections.asr' >/dev/null 2>&1; then
    # Exact NeMo pin prevents the 3.x cuda-bindings regression. Transformers <5
    # and NumPy 1.x avoid unrelated backwards-incompatible upgrades in NeMo 2.5.
    # pyarrow 26 requires NumPy 2.0, while this NeMo stack pins NumPy 1.26.4; keep pyarrow/fsspec/datasets on compatible versions.
    uv pip install --python "$CANARY_PY" \
      --constraint "$APP_DIR/venvs/canary/torch-constraints.txt" \
      'numpy==1.26.4' 'transformers<5' 'pyarrow==17.0.0' 'fsspec==2024.12.0' 'datasets==3.6.0' 'nemo-toolkit[asr]==2.5.3' || return $?
  else
    echo "NeMo ASR already imports in the pinned environment; skipping reinstall."
    # Repair case: if existing env has pyarrow 26 + numpy 1.26.4, downgrade pyarrow
    "$CANARY_PY" - <<'PY' >/dev/null 2>&1 || uv pip install --python "$CANARY_PY" 'pyarrow==17.0.0' 'fsspec==2024.12.0' 'datasets==3.6.0'
import pyarrow
import numpy
assert not (pyarrow.__version__.startswith("26") and numpy.__version__.startswith("1.26")), f"pyarrow {pyarrow.__version__} needs numpy 2, found {numpy.__version__}"
PY
  fi
  assert_no_forbidden_python_packages "$CANARY_PY" || return $?
  "$CANARY_PY" - <<'PY' || return $?
import torch, nemo.collections.asr as nemo_asr
assert torch.__version__.split("+", 1)[0] == "2.6.0", torch.__version__
assert torch.version.cuda is None and not torch.cuda.is_available()
assert hasattr(nemo_asr.models, "ASRModel")
print("NeMo ASR import OK; CPU Torch", torch.__version__)
PY
  "$CANARY_PY" -m pip check || return $?
}
if install_canary; then CANARY_AVAILABLE=1
else
  echo "WARN: Canary could not pass the pinned, CUDA-free NeMo checks. The Canary venv is retained for diagnosis; all model caches and other backends are preserved." >&2
fi

log "Install faster-whisper separately (CTranslate2 CPU/int8)"
FW_PY="$(make_venv faster-whisper)"
# PyAV's av.open is a Cython/builtin callable and inspect.signature() raises
# ValueError on supported versions. Check the installed PyAV version instead.
pyav_version_supports_metadata_errors() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import av, re
match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(av.__version__))
assert match, f"Cannot parse PyAV version: {av.__version__!r}"
version = tuple(int(part or 0) for part in match.groups())
assert version >= (12, 3, 0), f"PyAV {av.__version__} is too old; need >=12.3.0"
PY
}
if ! pyav_version_supports_metadata_errors "$FW_PY"; then
  echo "Installing/upgrading PyAV >=12.3.0 for faster-whisper"
  uv pip install --python "$FW_PY" --upgrade "av==15.1.0" || uv pip install --python "$FW_PY" --upgrade "av>=12.3.0"
fi
if ! "$FW_PY" -c 'import ctranslate2; from faster_whisper import WhisperModel' >/dev/null 2>&1; then
  echo "faster-whisper import is unavailable (including possibly stale package metadata); force-reinstalling its package files."
  uv pip install --reinstall --python "$FW_PY" faster-whisper
else
  echo "faster-whisper already imports correctly; skipping package install."
fi
# Final compatibility verification - faster-whisper may downgrade PyAV.
if ! pyav_version_supports_metadata_errors "$FW_PY"; then
  echo "Re-upgrading PyAV after faster-whisper (installed version is too old)"
  uv pip install --python "$FW_PY" --upgrade "av==15.1.0" || uv pip install --python "$FW_PY" --upgrade "av>=12.3.0"
fi
remove_forbidden_python_packages "$FW_PY"
assert_no_forbidden_python_packages "$FW_PY"
# Verify the actual import used by inference, not just CTranslate2's metadata.
"$FW_PY" - <<'FWVERIFY'
import av, ctranslate2, re
from faster_whisper import WhisperModel
match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(av.__version__))
assert match, f"Cannot parse PyAV version: {av.__version__!r}"
assert tuple(int(part or 0) for part in match.groups()) >= (12, 3, 0), f"PyAV {av.__version__} lacks required compatibility level"
print("faster-whisper import OK; CTranslate2", ctranslate2.__version__, "PyAV", av.__version__, "CPU mode")
FWVERIFY
"$FW_PY" -m pip check

# Optional newer open-weight ASR models are deliberately isolated. Each model
# gets an independent install transaction; a failed resolver, unavailable model
# repository, or failed download marks just that candidate and then continues.
OPTIONAL_ASR_STATUS="$APP_DIR/state/optional-asr-install-status.txt"
: > "$OPTIONAL_ASR_STATUS"
record_optional_status() { printf '%s\t%s\n' "$1" "$2" | tee -a "$OPTIONAL_ASR_STATUS"; }
install_optional_asr_env() {
  local env_name="$1" extra_package="" py constraint arg
  local -a packages=()
  shift
  for arg in "$@"; do
    if [[ "$arg" == GIT_PACKAGE=* ]]; then extra_package="${arg#GIT_PACKAGE=}"; else packages+=("$arg"); fi
  done
  log "Optional offline ASR candidate: $env_name"
  if ! py="$(make_venv "$env_name")"; then
    record_optional_status "$env_name" "ENV FAILED (could not create venv)"
    return 1
  fi
  if ! install_cpu_torch "$py"; then
    record_optional_status "$env_name" "ENV FAILED (CPU Torch install/validation)"
    return 1
  fi
  constraint="$APP_DIR/venvs/$env_name/torch-constraints.txt"
  # Keep model stacks separate; no NVIDIA/CUDA packages, vLLM, IPEX, or oneCCL.
  if ! uv pip install --python "$py" --constraint "$constraint" "${packages[@]}"; then
    record_optional_status "$env_name" "ENV FAILED (Python package install; next model will still be attempted)"
    return 1
  fi
  if [[ -n "$extra_package" ]]; then
    if ! uv pip install --python "$py" --constraint "$constraint" --no-deps "$extra_package"; then
      record_optional_status "$env_name" "ENV FAILED (optional model adapter package)"
      return 1
    fi
  fi
  if ! remove_forbidden_python_packages "$py" || ! assert_no_forbidden_python_packages "$py"; then
    record_optional_status "$env_name" "ENV FAILED (CUDA/NVIDIA package policy check)"
    return 1
  fi
  if ! "$py" - <<'PY' >/dev/null
import torch, transformers, soundfile, huggingface_hub
assert torch.version.cuda is None and not torch.cuda.is_available()
assert "+xpu" not in torch.__version__.lower()
print("offline ASR dependencies OK", transformers.__version__)
PY
  then
    record_optional_status "$env_name" "ENV FAILED (smoke test)"
    return 1
  fi
  record_optional_status "$env_name" "ENV READY (CPU-only; weights handled separately)"
}

# Separate environments let one model's dependency conflicts leave every other
# candidate and every known-good environment intact. Distil-Whisper reuses the
# already verified CTranslate2 environment.
install_optional_asr_env granite5-ctc 'transformers>=5.16.0' accelerate soundfile librosa safetensors huggingface_hub || true
install_optional_asr_env cohere-asr 'transformers>=5.4.0,<6' accelerate soundfile librosa sentencepiece protobuf datasets safetensors huggingface_hub || true
install_optional_asr_env moss-diarize 'transformers>=5.6.0,<6' accelerate soundfile librosa safetensors av soxr packaging fastapi uvicorn python-multipart huggingface_hub GIT_PACKAGE=https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git || true
# The preview checkpoint declares transformers_version 4.57.1; its custom Qwen3-Omni audio frontend has a one-token feature-mask mismatch under Transformers 5.x. Keep this model isolated on the version used by its checkpoint.
install_optional_asr_env moss-preview 'transformers==4.57.1' accelerate soundfile librosa safetensors sentencepiece protobuf huggingface_hub || true
install_optional_asr_env ark-asr 'transformers>=5.0,<6' accelerate soundfile librosa safetensors sentencepiece protobuf tiktoken huggingface_hub || true

if (( SKIP_MODEL_DOWNLOAD == 0 )); then
  log "Prefetch optional open-weight ASR candidates independently (a failure never blocks later candidates)"
  "$APP_DIR/download_optional_asr_models.sh" || echo "WARN: One or more optional ASR model downloads failed; see $OPTIONAL_ASR_STATUS. Other installation steps will continue." >&2
else
  record_optional_status "optional model weights" "SKIPPED (--skip-model-download requested; run download-optional-asr-models later while online; transcription remains offline-only)"
  echo "Optional model weights were intentionally skipped. To fetch them later without reinstalling environments, run:"
  echo "  download-optional-asr-models"
fi

log "Install SeamlessM4T v2 and IBM Granite Speech in a separate CPU-only Transformers environment"
HFASR_PY="$(make_venv transformers-asr)"
install_cpu_torch "$HFASR_PY"

# Install the tokenizer dependencies in a small, separate transaction. The
# previous combined resolver command could fail as a whole and was allowed to
# continue, leaving protobuf absent while the top-level installer still ended
# with "Install complete". This check is deliberately performed on every run
# and repairs only these packages; it never touches model/cache directories.
if ! "$HFASR_PY" - <<'PY' >/dev/null 2>&1
import google.protobuf, sentencepiece, tiktoken
PY
then
  echo "Repairing required tokenizer packages in the existing Transformers ASR environment."
  uv pip install --python "$HFASR_PY" \
    --constraint "$APP_DIR/venvs/transformers-asr/torch-constraints.txt" \
    protobuf tiktoken sentencepiece || die "Could not install protobuf/tiktoken/SentencePiece in $HFASR_PY; see the package-manager error above."
fi
"$HFASR_PY" - <<'PY' || die "Tokenizer dependencies still cannot be imported in $HFASR_PY."
import google.protobuf, sentencepiece, tiktoken
print("Tokenizer dependencies OK:", google.protobuf.__version__, sentencepiece.__version__, tiktoken.__version__)
PY

if ! "$HFASR_PY" - <<'PY' >/dev/null 2>&1
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, SeamlessM4Tv2ForSpeechToText
import peft, sentencepiece, soundfile, tiktoken
import google.protobuf
assert torch.version.cuda is None and not torch.cuda.is_available()
assert not (hasattr(torch, "xpu") and torch.xpu.is_available()), "Transformers ASR must stay on CPU"
PY
then
  echo "Installing/updating Transformers main and ASR dependencies."
  # Keep the already-installed CPU Torch/TorchAudio pair pinned while resolving
  # the newer Transformers source and PEFT adapter support. Do not install vLLM,
  # IPEX, torchvision, CUDA runtime packages, or any NVIDIA wheels.
  uv pip install --python "$HFASR_PY" \
    --constraint "$APP_DIR/venvs/transformers-asr/torch-constraints.txt" \
    'git+https://github.com/huggingface/transformers.git' accelerate peft sentencepiece protobuf tiktoken soundfile safetensors || die "Transformers ASR packages failed to install; see the resolver error above. The incomplete environment has been preserved for diagnosis."
else
  echo "Transformers, SeamlessM4T, Granite, and PEFT imports already work; skipping package downloads."
fi
remove_forbidden_python_packages "$HFASR_PY"
assert_no_forbidden_python_packages "$HFASR_PY"
if ! "$HFASR_PY" - <<'PY'
import torch
import transformers
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, SeamlessM4Tv2ForSpeechToText
import peft, sentencepiece, soundfile, tiktoken
import google.protobuf
from transformers.utils.import_utils import is_protobuf_available
assert is_protobuf_available(), "Transformers does not detect protobuf despite import google.protobuf succeeding"
from transformers.convert_slow_tokenizer import import_protobuf
import_protobuf()  # Smoke-test the exact protobuf loader used for SentencePiece conversion.
assert torch.version.cuda is None and not torch.cuda.is_available(), torch.__version__
assert "+xpu" not in torch.__version__.lower(), torch.__version__
assert not (hasattr(torch, "xpu") and torch.xpu.is_available()), "Transformers ASR env unexpectedly has a usable XPU"
print("Transformers ASR imports and tokenizer protobuf loader OK; CPU Torch", torch.__version__, "Transformers", transformers.__version__)
PY
then
  die "Transformers ASR tokenizer validation failed in $HFASR_PY. Do not use SeamlessM4T/Granite until this environment passes; package/cache files were preserved."
else
  "$HFASR_PY" -m pip check || echo "WARN: Transformers ASR pip check reported conflicts; inspect its isolated venv before use." >&2
  HFASR_AVAILABLE=1
fi

log "Reuse or build whisper.cpp for CPU only (all GPU/accelerator backends disabled)"
WHISPER_SRC="$APP_DIR/whisper.cpp"
WHISPER_BIN="$APP_DIR/bin/whisper-cli"

whisper_cpu_build_config_valid() {
  local cache="$WHISPER_SRC/build/CMakeCache.txt" option line
  [[ -f "$cache" && -x "$WHISPER_SRC/build/bin/whisper-cli" ]] || return 1
  # Known options must be OFF. An option absent from an older source revision
  # cannot have enabled that backend; a cached ON value forces reconfiguration.
  for option in GGML_CUDA GGML_VULKAN GGML_HIP GGML_SYCL GGML_OPENCL \
                GGML_OPENVINO GGML_MUSA GGML_WEBGPU GGML_METAL GGML_RPC \
                GGML_VIRTGPU GGML_VIRTGPU_BACKEND GGML_BACKEND_DL WHISPER_OPENVINO; do
    line="$(grep -E "^${option}:[^=]+=" "$cache" | tail -n 1 || true)"
    [[ -z "$line" || "$line" == "${option}:BOOL=OFF" || "$line" == "${option}:UNINITIALIZED=OFF" ]] || return 1
  done
  return 0
}

if [[ ! -d "$WHISPER_SRC/.git" ]]; then
  # Do not trust an old standalone whisper-cli binary: --help cannot tell whether
  # its backend was compiled with CUDA/OpenCL/Vulkan/SYCL or other accelerators.
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$WHISPER_SRC"
else
  echo "Reusing existing whisper.cpp source checkout; not running git pull on reruns."
fi

if whisper_cpu_build_config_valid; then
  if [[ ! -x "$WHISPER_BIN" ]] || ! cmp -s "$WHISPER_SRC/build/bin/whisper-cli" "$WHISPER_BIN"; then
    install -m 0755 "$WHISPER_SRC/build/bin/whisper-cli" "$WHISPER_BIN"
  fi
  echo "Reusing verified CPU-only whisper.cpp build; CMake cache and executable match."
else
  cmake -S "$WHISPER_SRC" -B "$WHISPER_SRC/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF \
    -DGGML_OPENCL=OFF -DGGML_OPENVINO=OFF -DGGML_MUSA=OFF -DGGML_WEBGPU=OFF \
    -DGGML_METAL=OFF -DGGML_RPC=OFF -DGGML_VIRTGPU=OFF -DGGML_VIRTGPU_BACKEND=OFF \
    -DWHISPER_OPENVINO=OFF -DWHISPER_CURL=OFF -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON
  cmake --build "$WHISPER_SRC/build" --parallel "$JOBS" --config Release
  whisper_cpu_build_config_valid || die "whisper.cpp build cache does not verify that all GPU/accelerator backends are disabled."
  install -m 0755 "$WHISPER_SRC/build/bin/whisper-cli" "$WHISPER_BIN"
fi
"$WHISPER_BIN" --help >/dev/null || die "CPU-only whisper-cli failed its --help check."

# Reuse old whisper.cpp weights even when downloads were explicitly skipped.
for model_file in ggml-large-v3-turbo.bin ggml-large-v3.bin ggml-base.en.bin; do
  target="$APP_DIR/models/$model_file"
  if [[ ! -s "$target" ]]; then
    for old in "$HOME/transcribe-tools/models/$model_file" "$HOME/transcribe-tools-qwen/models/$model_file"; do
      if [[ -s "$old" ]]; then ln -sfn "$old" "$target"; echo "Linked existing model: $old"; break; fi
    done
  fi
done

cat > "$USER_BIN/norm_audio" <<EOF
#!/usr/bin/env bash
exec "$TOOL_PY" "$APP_DIR/norm_audio.py" "\$@"
EOF
cat > "$USER_BIN/transcribe" <<EOF
#!/usr/bin/env bash
exec "$TOOL_PY" "$APP_DIR/transcribe.py" "\$@"
EOF
cat > "$USER_BIN/dynamic_norm_audio" <<EOF
#!/usr/bin/env bash
exec "$APP_DIR/dynamic_norm_audio.sh" "\$@"
EOF
cat > "$USER_BIN/transcribe_audio" <<EOF
#!/usr/bin/env bash
exec "$APP_DIR/transcribe_audio.sh" "\$@"
EOF
cat > "$USER_BIN/download-optional-asr-models" <<EOF
#!/usr/bin/env bash
exec "$APP_DIR/download_optional_asr_models.sh" "\$@"
EOF
cat > "$USER_BIN/compare-transcribers" <<EOF
#!/usr/bin/env bash
exec "$APP_DIR/compare-transcribers.sh" "\$@"
EOF
cat > "$USER_BIN/compare-transcribers-long" <<EOF
#!/usr/bin/env bash
exec "$APP_DIR/compare-transcribers-long.sh" "\$@"
EOF
chmod 0755 "$USER_BIN/norm_audio" "$USER_BIN/transcribe" "$USER_BIN/dynamic_norm_audio" "$USER_BIN/transcribe_audio" "$USER_BIN/download-optional-asr-models" "$USER_BIN/compare-transcribers" "$USER_BIN/compare-transcribers-long"

if (( SKIP_MODEL_DOWNLOAD == 0 )); then
  log "Reuse existing Hugging Face snapshots before attempting any downloads"
  mkdir -p "$APP_DIR/models/hf-cache/hub"
  HF_HOME_TARGET="$APP_DIR/models/hf-cache"
  for root in "$APP_DIR/models/hf-cache" "$HOME/.cache/huggingface" "$HOME/transcribe-tools-qwen/hf-cache"; do
    [[ -d "$root/hub" ]] || continue
    if find "$root/hub" -maxdepth 1 -type d \( \
      -name 'models--Qwen--Qwen3-ASR-*' -o -name 'models--Systran--faster-whisper-*' \
      -o -name 'models--mobiuslabsgmbh--faster-whisper-large-v3-turbo' \
      -o -name 'models--nvidia--canary-1b-flash' \
      -o -name 'models--nvidia--canary-1b-v2' \
      -o -name 'models--ibm-granite--granite-speech-3.3-2b' \
      -o -name 'models--ibm-granite--granite-speech-3.3-8b' \
      -o -name 'models--ibm-granite--granite-speech-5.0-470m-turboctc' \
      -o -name 'models--CohereLabs--cohere-transcribe-03-2026' \
      -o -name 'models--OpenMOSS-Team--MOSS-Transcribe-Diarize' \
      -o -name 'models--OpenMOSS-Team--MOSS-Transcribe-preview-2B' \
      -o -name 'models--Edge0--ARK-ASR-3B' \
      -o -name 'models--distil-whisper--distil-large-v3.5-ct2' \) -print -quit | grep -q .; then
      HF_HOME_TARGET="$root"
      break
    fi
  done
  export HF_HOME="$HF_HOME_TARGET" HUGGINGFACE_HUB_CACHE="$HF_HOME_TARGET/hub"
  echo "Using HF cache root: $HF_HOME"
  "$QWEN_PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repos = ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B")
active = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
roots = []
for root in (active, Path.home()/".cache/huggingface/hub", Path.home()/"transcribe-tools-qwen/hf-cache/hub"):
    if root not in roots:
        roots.append(root)

def complete_local_dir(path):
    return (path/"config.json").is_file() and (any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")))

def ensure_repo(repo):
    direct = Path.home()/"transcribe-tools-qwen"/"models"/repo.rsplit("/", 1)[-1]
    if complete_local_dir(direct):
        print(f"REUSE existing local model {repo}: {direct}", flush=True)
        return
    # First find a complete snapshot anywhere; avoid downloading a second copy.
    for root in roots:
        if not root.is_dir():
            continue
        try:
            path = snapshot_download(repo_id=repo, cache_dir=str(root), local_files_only=True)
            print(f"REUSE existing {repo}: {path}", flush=True)
            return
        except Exception:
            pass
    # If only a partial cache exists, resume in that exact cache directory.
    key = "models--" + repo.replace("/", "--")
    partial_roots = [root for root in roots if (root/key).is_dir()]
    target = partial_roots[0] if partial_roots else active
    print(f"Downloading/resuming missing files for {repo} in {target}; existing cache files are retained.", flush=True)
    try:
        path = snapshot_download(repo_id=repo, cache_dir=str(target))
        print(f"Model cache ready: {path}", flush=True)
    except Exception as exc:
        print(f"WARN: could not finish {repo}: {exc}. Existing complete/partial files are kept; rerun later.", flush=True)

for repo in repos:
    ensure_repo(repo)
PY
  "$FW_PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repos = ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "Systran/faster-whisper-large-v3", "Systran/faster-whisper-base.en")
active = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
roots = []
for root in (active, Path.home()/".cache/huggingface/hub", Path.home()/"transcribe-tools-qwen/hf-cache/hub"):
    if root not in roots:
        roots.append(root)

def ensure_repo(repo):
    for root in roots:
        if not root.is_dir():
            continue
        try:
            path = snapshot_download(repo_id=repo, cache_dir=str(root), local_files_only=True)
            print(f"REUSE existing {repo}: {path}", flush=True)
            return
        except Exception:
            pass
    key = "models--" + repo.replace("/", "--")
    partial_roots = [root for root in roots if (root/key).is_dir()]
    target = partial_roots[0] if partial_roots else active
    print(f"Downloading/resuming missing files for {repo} in {target}; existing cache files are retained.", flush=True)
    try:
        path = snapshot_download(repo_id=repo, cache_dir=str(target))
        print(f"Model cache ready: {path}", flush=True)
    except Exception as exc:
        print(f"WARN: could not finish {repo}: {exc}. Existing complete/partial files are kept; rerun later.", flush=True)

for repo in repos:
    ensure_repo(repo)
PY
  download_whisper_model() {
    local file="$1" min_bytes="$2"
    local url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$file"
    local target="$APP_DIR/models/$file"
    local tmp="$target.download"
    if [[ -s "$target" ]] && (( $(stat -c %s "$target") >= min_bytes )); then
      echo "Already present: $target ($(du -h "$target" | cut -f1)); skipping download."
      return 0
    fi
    for old in "$HOME/transcribe-tools/models/$file" "$HOME/transcribe-tools-qwen/models/$file"; do
      if [[ -s "$old" ]] && (( $(stat -c %s "$old") >= min_bytes )); then
        echo "Reusing existing whisper.cpp model: $old"
        ln -sfn "$old" "$target"
        return 0
      fi
    done
    if [[ -s "$target" && ! -s "$tmp" ]]; then
      if [[ -L "$target" ]]; then
        # Never let curl follow a moved symlink and modify the source model in
        # an older cache. Copy its current bytes into the resumable local file.
        cp -L --reflink=auto -- "$target" "$tmp"
      else
        mv -- "$target" "$tmp"
      fi
    fi
    echo "Downloading/resuming $file (partial data kept at $tmp)"
    if ! curl --fail --location --retry 5 --retry-all-errors --connect-timeout 20 --continue-at - "$url" -o "$tmp"; then
      echo "WARN: download failed for $file; partial file preserved at $tmp for the next run." >&2
      return 1
    fi
    [[ -s "$tmp" ]] || { echo "Download empty; partial file preserved at $tmp" >&2; return 1; }
    local actual
    actual="$(stat -c %s "$tmp")"
    (( actual >= min_bytes )) || { echo "Download incomplete for $file ($actual bytes; need >= $min_bytes). Kept $tmp for resume." >&2; return 1; }
    mv -- "$tmp" "$target"
  }
  log "Download only missing whisper.cpp model weights (resumable)"
  download_whisper_model ggml-large-v3-turbo.bin 1620000000 || echo "WARN: large-v3-turbo unavailable; will retry on a later installer run." >&2
  download_whisper_model ggml-large-v3.bin 3080000000 || echo "WARN: large-v3 unavailable; will retry on a later installer run." >&2
  download_whisper_model ggml-base.en.bin 147000000 || echo "WARN: base.en unavailable; will retry on a later installer run." >&2
  canary_checkpoint_is_valid() {
    local candidate="$1"
    "$CANARY_PY" - "$candidate" <<'PY' >/dev/null 2>&1
import os, sys, tarfile
path = sys.argv[1]
with open(path, "rb") as raw:
    compressed = raw.read(2) == b"\x1f\x8b"
with tarfile.open(path, "r:*") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit(1)
    size = os.path.getsize(path)
    # NeMo checkpoints are usually tar archives. Catch incomplete payloads in
    # uncompressed archives; tarfile also checks headers and gzip integrity.
    if not compressed and any(m.isfile() and m.offset_data + m.size > size for m in members):
        raise SystemExit(1)
PY
  }

  CANARY_FOUND_PATH=""
  canary_search_checkpoint() {
    local filename="$1" checkpoint suspect
    CANARY_FOUND_PATH=""
    while IFS= read -r -d '' checkpoint; do
      if canary_checkpoint_is_valid "$checkpoint"; then
        CANARY_FOUND_PATH="$checkpoint"
        return 0
      fi
      # Hugging Face snapshot entries are commonly symlinks into the blobs/
      # directory. Never move/rename these links: doing so mutates the shared
      # snapshot structure and does not repair an incomplete underlying blob.
      if [[ -L "$checkpoint" ]]; then
        echo "WARN: Canary snapshot symlink did not pass archive validation; leaving cache link untouched: $checkpoint" >&2
        continue
      fi
      # Keep a suspect regular-file artifact for diagnosis, rather than
      # silently accepting a truncated checkpoint.
      suspect="$checkpoint.suspect.$(date +%Y%m%d-%H%M%S)"
      [[ ! -e "$suspect" && ! -L "$suspect" ]] || suspect="$suspect.$RANDOM"
      echo "Existing Canary checkpoint failed archive validation; preserving as $suspect." >&2
      mv -- "$checkpoint" "$suspect"
    done < <(find \
      "$HOME/.cache/torch/NeMo" \
      "$APP_DIR/models/nemo-cache" \
      "$APP_DIR/models/hf-cache/hub" \
      "$HF_HOME/hub" \
      "$HOME/.cache/huggingface/hub" \
      -name "$filename" \( -type f -o -type l \) -print0 2>/dev/null)
    return 1
  }

  ensure_canary_model() {
    local model_id="$1" filename="$2"
    if canary_search_checkpoint "$filename"; then
      echo "Reusing validated $model_id checkpoint: $CANARY_FOUND_PATH; no model download needed."
      return 0
    fi
    echo "No valid $filename found. Prefetching only $model_id; other Canary checkpoints are checked separately."
    if HF_HOME="$HF_HOME" \
       HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
       NEMO_CACHE_DIR="$APP_DIR/models/nemo-cache" \
       NEMO_DISABLE_TELEMETRY=1 ONELOGGER_DISABLE=1 \
       CANARY_MODEL_ID="$model_id" \
       "$CANARY_PY" - <<'PY'
import os
from pathlib import Path
import nemo.collections.asr as nemo_asr
mid = os.environ["CANARY_MODEL_ID"]
model_cls = getattr(nemo_asr.models, "EncDecMultiTaskModel", None)
if model_cls is None:
    from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel as model_cls
print(f"Resolving {mid} checkpoint on CPU without instantiating model weights or the auxiliary timestamp model.", flush=True)
checkpoint = model_cls.from_pretrained(mid, return_model_file=True)
if not isinstance(checkpoint, (str, os.PathLike)):
    raise RuntimeError(f"NeMo did not return a checkpoint path for {mid}: {checkpoint!r}")
path = Path(checkpoint)
if not path.is_file():
    raise RuntimeError(f"NeMo returned a checkpoint path that is not a file: {path}")
print(f"{mid} checkpoint fetched/resolved: {path} ({path.stat().st_size} bytes); model weights not loaded.", flush=True)
PY
    then
      if canary_search_checkpoint "$filename"; then
        echo "Validated newly cached checkpoint: $CANARY_FOUND_PATH"
      else
        # Do not fail all backends if NeMo changes its cache file layout, but
        # make the unverified cache state visible in the log.
        echo "WARN: $model_id checkpoint resolved, but no archive-validated $filename appeared in known cache roots." >&2
      fi
    else
      echo "WARN: $model_id checkpoint could not be fetched/resolved. Other model backends remain usable; rerun later to retry this checkpoint." >&2
    fi
    return 0
  }

  if (( CANARY_AVAILABLE == 1 )); then
    log "Reuse/prefetch Canary-1B-Flash and Canary-1B-v2 independently (CPU-only)"
    mkdir -p "$APP_DIR/models/nemo-cache"
    ensure_canary_model "nvidia/canary-1b-flash" "canary-1b-flash.nemo"
    ensure_canary_model "nvidia/canary-1b-v2" "canary-1b-v2.nemo"
    if (( INCLUDE_NONCOMMERCIAL_MODELS == 1 )); then
      echo "NOTICE: Opting into the CC BY-NC 4.0 Canary-1B weights; use is noncommercial only. See THIRD_PARTY_NOTICES.md."
      ensure_canary_model "nvidia/canary-1b" "canary-1b.nemo"
    else
      echo "Skipping nvidia/canary-1b weights by default (CC BY-NC 4.0); pass --include-noncommercial-models to opt in."
    fi
  else
    echo "Skipping Canary model downloads because CUDA-free NeMo did not validate."
  fi

  if (( HFASR_AVAILABLE == 1 )); then
    log "Reuse/prefetch Granite Speech 3.3 2B; 8B is opt-in"
    if (( INCLUDE_NONCOMMERCIAL_MODELS == 1 )); then
      echo "NOTICE: Opting into SeamlessM4T v2 weights under CC BY-NC 4.0; use is noncommercial only. See THIRD_PARTY_NOTICES.md."
    else
      echo "Skipping SeamlessM4T v2 weights by default (CC BY-NC 4.0); pass --include-noncommercial-models to opt in."
    fi
    HFASR_INCLUDE_GRANITE_8B="$INCLUDE_GRANITE_8B" HFASR_INCLUDE_NONCOMMERCIAL_MODELS="$INCLUDE_NONCOMMERCIAL_MODELS" "$HFASR_PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repos = ["ibm-granite/granite-speech-3.3-2b"]
if os.environ.get("HFASR_INCLUDE_NONCOMMERCIAL_MODELS") == "1":
    repos.append("facebook/seamless-m4t-v2-large")
if os.environ.get("HFASR_INCLUDE_GRANITE_8B") == "1":
    repos.append("ibm-granite/granite-speech-3.3-8b")

active = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
roots = []
for root in (active, Path.home()/".cache/huggingface/hub", Path.home()/"transcribe-tools-qwen/hf-cache/hub"):
    if root not in roots:
        roots.append(root)

# Fetch model weights and runtime configuration/tokenizer files, but not model
# cards, screenshots, or sample recordings bundled in the repositories.
patterns = [
    "*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.tiktoken", "*.jinja",
    "tokenizer*", "spiece.model", "vocab.json", "merges.txt",
]

for repo in repos:
    key = "models--" + repo.replace("/", "--")
    found = False
    for root in roots:
        if not (root/key).is_dir():
            continue
        try:
            path = snapshot_download(
                repo_id=repo, cache_dir=str(root), local_files_only=True,
                allow_patterns=patterns,
            )
            print(f"REUSE existing {repo}: {path}", flush=True)
            found = True
            break
        except Exception:
            # If the repo's cache is incomplete, choose that exact cache below
            # so Hugging Face can resume there without making a duplicate copy.
            pass
    if found:
        continue
    partial_roots = [root for root in roots if (root/key).is_dir()]
    target = partial_roots[0] if partial_roots else active
    print(f"Downloading/resuming {repo} under {target}; only model/runtime files are requested.", flush=True)
    try:
        path = snapshot_download(repo_id=repo, cache_dir=str(target), allow_patterns=patterns)
        print(f"Model cache ready: {path}", flush=True)
    except Exception as exc:
        print(f"WARN: could not finish {repo}: {exc}. Existing complete/partial files are retained; rerun install.sh to retry.", flush=True)
PY
  else
    echo "WARN: Transformers ASR environment is not healthy; skipping SeamlessM4T/Granite prefetch until its environment is repaired." >&2
  fi
else
  log "Skipping model downloads (--skip-model-download requested); existing weights were not altered"
fi

log "Run syntax and import checks"
bash -n "$SCRIPT_DIR/install.sh"
bash -n "$APP_DIR/compare-transcribers.sh"
bash -n "$APP_DIR/compare-transcribers-long.sh"
bash -n "$APP_DIR/dynamic_norm_audio.sh"
bash -n "$APP_DIR/transcribe_audio.sh"
bash -n "$APP_DIR/download_optional_asr_models.sh"
"$TOOL_PY" -m py_compile "$APP_DIR/norm_audio.py" "$APP_DIR/transcribe.py" "$APP_DIR/memory_guard.py"
"$TOOL_PY" "$APP_DIR/transcribe.py" --help >/dev/null
"$TOOL_PY" "$APP_DIR/norm_audio.py" --help >/dev/null

if [[ -s "$OPTIONAL_ASR_STATUS" ]]; then
  log "Optional ASR environment/weight outcomes (environment READY does not imply weights READY)"
  while IFS=$'\t' read -r candidate outcome; do
    [[ -n "$candidate" ]] || continue
    printf '  %-62s %s\n' "$candidate" "$outcome"
  done < "$OPTIONAL_ASR_STATUS"
fi

log "Install complete"
cat <<EOF
Commands (if ~/.local/bin is not already on PATH):
  export PATH="\$HOME/.local/bin:\$PATH"

Optional open-weight ASR environment/model outcomes are recorded in:
  $APP_DIR/state/optional-asr-install-status.txt

Single-gain EBU R128 loudness normalization (no denoising or compression):
  norm_audio /path/to/recording.wav -o /path/to/recording_norm.wav

FFmpeg dynamic normalization (dynaudnorm, mono 16 kHz PCM WAV):
  dynamic_norm_audio /path/to/recording.mp3 -o /path/to/recording_dynaudnorm.wav

Run the curated multi-model normalization/transcription comparison:
  transcribe_audio /path/to/recording.mp3

Run a model sweep or long-audio memory-aware comparison:
  compare-transcribers /path/to/recording.wav en
  compare-transcribers-long --models qwen3,faster-whisper /path/to/long_recording.wav en

Transcribe directly with a selected local model:
  transcribe /path/to/recording.wav --model faster-whisper --lang en

Download/resume optional open-weight ASR models:
  download-optional-asr-models --models distil-whisper-large-v3.5,ark-asr-3b

A standalone transcript comparator is installed as:
  $APP_DIR/transcript_railway_standalone.html
See README_TRANSCRIPT_RAILWAY_STANDALONE.md for its local browser workflow.
EOF
