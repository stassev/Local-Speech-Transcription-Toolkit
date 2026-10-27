# Third-party model and software notices

Copyright (C) 2026 Svetlin Tassev (this notice file).

The Speech Toolkit source code and standalone HTML in this repository are licensed under **GPL-3.0**; see [`LICENSE`](LICENSE). That license applies to this project's own code. It does not relicense third-party Python packages, external programs, or model weights downloaded by the installer. Those materials remain subject to their respective upstream terms.

No speech-model weights are bundled in this GitHub source archive. The installer obtains model artifacts from their upstream repositories. Model repositories and their license declarations can change; the upstream model card and license text are authoritative. Verify terms before downloading, using, or redistributing model artifacts.

## Model checkpoints with noncommercial restrictions

These checkpoints are **not downloaded by default**. The installer option `--include-noncommercial-models` is an explicit opt-in and does not waive, alter, or grant exceptions to their license terms.

- **NVIDIA Canary-1B** — repository: <https://huggingface.co/nvidia/canary-1b>. The model card identifies the weights as **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**: <https://creativecommons.org/licenses/by-nc/4.0/>. Commercial use of the licensed model material is not permitted by that license.
- **Meta SeamlessM4T v2 Large** — repository: <https://huggingface.co/facebook/seamless-m4t-v2-large>. The model card identifies the weights as **CC BY-NC 4.0**: <https://creativecommons.org/licenses/by-nc/4.0/>. Commercial use of the licensed model material is not permitted by that license.

## Related models with attribution requirements

- **NVIDIA Canary-1B-Flash** — <https://huggingface.co/nvidia/canary-1b-flash> — model card identifies the model as **CC BY 4.0**: <https://creativecommons.org/licenses/by/4.0/>. Follow the attribution and notice requirements when redistributing the model material.
- **NVIDIA Canary-1B-v2** — <https://huggingface.co/nvidia/canary-1b-v2> — model card identifies the model as **CC BY 4.0**: <https://creativecommons.org/licenses/by/4.0/>. Follow the attribution and notice requirements when redistributing the model material.

## Other downloaded packages and checkpoints

The project may install software and optionally download additional models, including Qwen, faster-whisper / CTranslate2 checkpoints, whisper.cpp models, IBM Granite Speech, Distil-Whisper, Cohere Transcribe, MOSS Transcribe, and ARK-ASR-3B. This is not an exhaustive dependency inventory and should not be treated as a guarantee that any third-party repository's license remains unchanged. Check the distribution metadata and upstream license files for the specific version you install. Preserve required copyright notices and licenses if you redistribute those materials.

The toolkit does not bundle the listed model weights in this repository. Downloading from upstream does not make this project the owner or licensor of those weights. This document is informational, not legal advice.
