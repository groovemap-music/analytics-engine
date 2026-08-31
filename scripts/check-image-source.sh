#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! git -C "${repo_root}" rev-parse --verify 'HEAD^{commit}' >/dev/null 2>&1; then
  echo "Refusing to label an image without a verifiable source revision." >&2
  exit 2
fi

# The image revision describes committed first-party source. CI dependency
# checkouts and generated build artifacts do not alter that tree, while any
# staged or unstaged change to a tracked file invalidates its provenance label.
if ! git -C "${repo_root}" diff --quiet HEAD --; then
  echo "Refusing to label an image from modified tracked source." >&2
  exit 2
fi
