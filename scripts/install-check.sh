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
"${install_tmp}/venv/bin/python" -c 'import importlib.util; import insights.insights; import insights.config; from insights.schema_release_contract import create_artist_embedding_release, publish_artist_embedding_release, retire_artist_similar_artists_version; assert importlib.util.find_spec("groovemap_schema") is None'
