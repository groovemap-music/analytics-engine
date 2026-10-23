#!/usr/bin/env bash
set -euo pipefail

# Stages the two first-party git dependencies as wheels (they are not on an index, so they
# cannot appear in the hash-pinned requirements export): the python-libraries runtime and
# the database-schema release helpers similar-artist publication imports at runtime.
expected="771515349b592ff569b1b3e5712f7bef936bab58"
schema_expected="08a9f8021c48d2c0d9a2a81ce4b6a548845e8bf6"
checkouts=()
resolved=
trap 'for checkout in ${checkouts[@]+"${checkouts[@]}"}; do rm -rf -- "${checkout}"; done' EXIT

checkout_is_valid() {
  local repo="$1" rev="$2"
  [[ -d "${repo}/.git" ]] &&
    [[ "$(git -C "${repo}" rev-parse HEAD)" = "${rev}" ]] &&
    [[ -z "$(git -C "${repo}" status --short)" ]]
}

# Sets `resolved` to a clean checkout of `name` at `rev`: the override when set (which must
# match), else a matching adjacent checkout, else a temporary clone of the public source.
# Not a command substitution, so the temporary clone is recorded for the EXIT trap.
resolve_checkout() {
  local name="$1" rev="$2" override_var="$3"
  local override="${!override_var:-}"
  local repo="${override:-../${name}}"
  if checkout_is_valid "${repo}" "${rev}"; then
    resolved="${repo}"
    return
  fi
  if [[ -n "${override}" ]]; then
    echo "${override_var} must be a clean checkout at ${rev}." >&2
    exit 2
  fi
  local checkout
  checkout="$(mktemp -d)"
  case "${checkout}" in
  /tmp/* | /private/tmp/* | /var/folders/*) ;;
  *)
    echo "Unexpected temporary checkout path: ${checkout}" >&2
    exit 2
    ;;
  esac
  checkouts+=("${checkout}")
  git clone --quiet --filter=blob:none --no-checkout "https://github.com/groovemap-music/${name}.git" "${checkout}/${name}"
  git -C "${checkout}/${name}" checkout --quiet "${rev}"
  resolved="${checkout}/${name}"
}

resolve_checkout python-libraries "${expected}" GROOVEMAP_RUNTIME_REPO
runtime_repo="${resolved}"
resolve_checkout database-schema "${schema_expected}" GROOVEMAP_SCHEMA_REPO
schema_repo="${resolved}"

mkdir -p .build/runtime
find .build/runtime -type f -name '*.whl' -delete
uv build --wheel --out-dir .build/runtime "${runtime_repo}"
uv build --wheel --out-dir .build/runtime "${schema_repo}"
uv export \
  --frozen \
  --no-dev \
  --no-emit-project \
  --no-emit-package groovemap-runtime \
  --no-emit-package groovemap-database-schema \
  --output-file .build/requirements.txt \
  >/dev/null
