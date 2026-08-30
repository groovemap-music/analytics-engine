#!/usr/bin/env bash
set -euo pipefail

bash scripts/prepare-runtime-wheel.sh
install_tmp="$(mktemp -d)"
trap 'rm -rf "${install_tmp}"' EXIT

uv venv "${install_tmp}/venv"
uv pip install \
  --python "${install_tmp}/venv/bin/python" \
  --require-hashes \
  --requirements .build/requirements.txt
uv pip install \
  --python "${install_tmp}/venv/bin/python" \
  --no-deps \
  .build/runtime/*.whl \
  dist/*.whl
"${install_tmp}/venv/bin/python" -c 'import insights.insights; import insights.config'
