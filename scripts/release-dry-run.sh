#!/usr/bin/env bash
set -euo pipefail

uv build --out-dir dist --clear
(
  cd dist
  shasum -a 256 ./*.whl ./*.tar.gz > SHA256SUMS
)
uv run cyclonedx-py environment --output-file dist/sbom.json
notices_tmp="$(mktemp -d)"
trap 'rm -rf -- "${notices_tmp}"' EXIT
uv venv "${notices_tmp}/venv"
uv pip install --python "${notices_tmp}/venv/bin/python" --find-links .build/runtime dist/*.whl
uv run pip-licenses \
  --python "${notices_tmp}/venv/bin/python" \
  --ignore-packages groovemap-analytics-engine \
  --format=json \
  --with-urls \
  --with-license-file \
  --no-license-path \
  --output-file=dist/THIRD_PARTY_NOTICES.json
uv run python scripts/write-build-provenance.py
test -s dist/SHA256SUMS
test -s dist/sbom.json
test -s dist/THIRD_PARTY_NOTICES.json
test -s dist/provenance.json
