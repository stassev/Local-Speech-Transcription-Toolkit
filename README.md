# Local Speech Transcription Toolkit

Copyright (C) 2026 Svetlin Tassev. Licensed under the GNU General Public License v3.0; see [`LICENSE`](LICENSE).

A command-line toolkit for running open-weight speech-recognition models locally, comparing transcription runs, and making two types of audio normalization. The installation script targets Manjaro/Arch Linux and keeps incompatible ASR stacks in separate Python environments.

The code contains a standalone html which is a fancy Transcription Comparator diffing tool. It is synced with the audio (if provided). It can be used for diffing ordinary text files as well. Here it is in action:

<img width="1920" height="887" alt="Screenshot From 2026-10-27 05-14-46" src="https://github.com/user-attachments/assets/da2314fa-ce15-4773-b109-fc321db8726e" />


## What's included

| File | Purpose |
|---|---|
| `install.sh` | Installs system prerequisites, isolated Python environments, local model weights, and commands under `~/.local/bin`. |
| `transcribe.py` | Main dispatcher for local ASR backends. It routes each selected backend to its own environment. |
| `transcribe_audio.sh` | Runs selected models against `norm_audio` and/or `dynamic_norm_audio` versions of the same input, sequentially to limit RAM use. |
| `norm_audio.py` | Single-gain EBU R128 loudness normalization. It does not denoise or compress dynamics. Default output is mono 44.1 kHz WAV at a target of -23 LUFS. |
| `dynamic_norm_audio.sh` | FFmpeg `dynaudnorm` normalization; exports mono 16 kHz PCM WAV. |
| `memory_guard.py` | Polls a Linux process tree's memory use and terminates a transcription run when the configured limit is exceeded. |
| `compare-transcribers.sh` | Runs a sequential model sweep on the supplied audio file, with separate output and log files for each run. |
| `compare-transcribers-long.sh` | Compares model-native long-audio handling with a process-tree memory watchdog. Audio is used as supplied. |
| `download_optional_asr_models.sh` | Downloads or resumes selected optional open-weight model files without reinstalling the environments. |
| `transcript_railway_standalone.html` | Browser-based transcript comparator. See [`README_TRANSCRIPT_RAILWAY_STANDALONE.md`](README_TRANSCRIPT_RAILWAY_STANDALONE.md). |

## Requirements

- Manjaro or Arch Linux for the provided installer (`pacman` is required).
- Internet access during installation if packages or model weights need downloading.
- FFmpeg/FFprobe for audio normalization and decoding.
- Sufficient disk space and memory for the model backends you choose.

The installer uses `uv` to manage Python 3.12 and creates separate environments for model stacks. Qwen can optionally use native Intel XPU support when explicitly enabled or selected by automatic detection; other configured ASR backends are CPU-only. The installer does not install CUDA/NVIDIA Python packages, IPEX, or oneCCL. An Intel XPU-capable wheel does not guarantee that a particular GPU/driver combination will work.

## Install

Run as your normal desktop user, not as root. `install.sh` uses `sudo` only when required system packages need to be installed.

```bash
git clone YOUR_REPOSITORY_URL speech-toolkit
cd speech-toolkit
chmod +x install.sh
./install.sh
```

Useful options:

```bash
# Set up the programs/environments without prefetching model weights.
./install.sh --skip-model-download

# Opt in to prefetching the larger Granite Speech 3.3 8B checkpoint.
./install.sh --include-granite-8b

# Keep Qwen on CPU, or request Intel XPU.
./install.sh --qwen-device cpu
./install.sh --qwen-device xpu
```

Model weights can occupy many gigabytes. Without `--skip-model-download`, the installer attempts to prefetch the configured common and optional open-weight models. Failures for optional model environments or downloads are recorded in `~/.local/share/speech-toolkit/state/optional-asr-install-status.txt` and do not necessarily prevent other backends from being configured. Run `download-optional-asr-models --help` to see how to fetch optional checkpoints later.

## Common commands

The installer places command wrappers in `~/.local/bin`. Add that directory to `PATH` if your shell does not already include it:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Transcribe one recording with the default backend (faster-whisper):

```bash
transcribe recording.wav --lang en --output ./recording-run
```

Choose another backend explicitly:

```bash
transcribe recording.wav --model qwen3 --lang en --output ./qwen-run
transcribe recording.wav --model canary-1b-v2 --lang en --output ./canary-run
transcribe recording.wav --model whisper-cpp --whisper-size large-v3-turbo --lang en
```

Run the curated model/normalizer matrix:

```bash
transcribe_audio recording.wav
transcribe_audio --add-models qwen3,granite-speech-5.0-turboctc recording.wav
transcribe_audio --normalizers canary-1b-v2=both recording.wav
```

Normalize audio independently:

```bash
# Constant-gain EBU R128 loudness normalization
norm_audio recording.wav -o recording_norm.wav

# Dynamic normalization, 16 kHz mono PCM WAV
dynamic_norm_audio recording.wav -o recording_dynaudnorm.wav
```

Compare backends or run a long-audio memory-aware sweep:

```bash
compare-transcribers recording.wav en
compare-transcribers-long --models qwen3,faster-whisper --ram-limit-gb 22 long_recording.wav en
```

The comparison scripts pass audio to the transcriber as supplied. Use `norm_audio`, `dynamic_norm_audio`, or `transcribe_audio` when you want to compare normalized variants explicitly.

## Model backends and model-weight licenses

The dispatcher supports Qwen3-ASR (0.6B and 1.7B), NVIDIA NeMo Canary (1B, 1B Flash, and 1B v2), faster-whisper, whisper.cpp, SeamlessM4T v2, IBM Granite Speech 3.3 (2B and 8B), and optional open-weight alternatives including Granite Speech 5.0 TurboCTC, Distil-Whisper Large-v3.5, Cohere Transcribe, MOSS Transcribe, and ARK-ASR-3B. Exact availability depends on the installed environment and cached/downloaded weights; see `transcribe --help` for the current command-line choices.

**Two noncommercial model checkpoints are not downloaded by default:** `nvidia/canary-1b` and `facebook/seamless-m4t-v2-large` are published under CC BY-NC 4.0. To explicitly opt into downloading them, run `./install.sh --include-noncommercial-models`. This opt-in does not change their license or grant commercial-use rights. The matching models are also omitted from the default `compare-transcribers` sweep; use `compare-transcribers --include-noncommercial-models recording.wav en` to include them intentionally. They remain available for direct use if their weights are already cached or separately acquired under their terms. Existing files and caches are never deleted by this change.

The code's GPL-3.0-only license does not relicense downloaded packages or model weights. Check [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream model-license links and attribution notes before downloading, using, or redistributing third-party material.

## Runtime and privacy

Transcription inference uses local model weights and does not call hosted transcription APIs. Installation and model download commands do access package/model repositories as needed. The scripts write transcripts, sidecars, logs, and intermediate audio to the output directories they report; review those directories before sharing them. Avoid placing recordings or generated outputs in the repository root.

The standalone HTML comparator also works locally in a browser and does not require Python or model weights. Its import behavior is documented separately in `README_TRANSCRIPT_RAILWAY_STANDALONE.md`.

## Local reference checks

A `testing/` directory may be kept on the developer's machine for reference checks. It is ignored by Git and is not intended to be published. Its retained checks use synthetic audio or source-policy assertions, not sample transcript passages.

---

**Development note:** AI tools were used in creating and editing parts of this codebase. The published source is licensed under GPL-3.0.
