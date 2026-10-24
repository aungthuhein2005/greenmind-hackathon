#!/usr/bin/env bash
# One-time setup: uv + venv with vLLM and harness deps.
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd ~/greenmind
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install vllm -r requirements.txt
python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, torch.cuda.is_available())"
