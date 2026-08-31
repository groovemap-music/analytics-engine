#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
bash "${repo_root}/scripts/check-image-source.sh"

vcs_ref="$(git -C "${repo_root}" rev-parse --verify 'HEAD^{commit}')"
if [[ ! "${vcs_ref}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Expected a full 40-character source revision, got: ${vcs_ref}" >&2
  exit 2
fi

build_date="$(git -C "${repo_root}" show -s --format=%cI "${vcs_ref}")"
build_version="$(uv version --short)"
docker_config="$(mktemp -d)"
trap 'rm -rf "${docker_config}"' EXIT
active_context="$(docker context ls --format '{{if .Current}}{{.Name}}{{end}}' | sed -n '1p')"
docker_host="$(docker context inspect "${active_context}" --format '{{.Endpoints.docker.Host}}')"

DOCKER_HOST="${docker_host}" docker --config "${docker_config}" build \
  --build-arg "BUILD_DATE=${build_date}" \
  --build-arg "BUILD_VERSION=${build_version}" \
  --build-arg "VCS_REF=${vcs_ref}" \
  --tag analytics-engine:local \
  .
