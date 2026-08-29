#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 SOURCE_REPOSITORY ABSOLUTE_OUTPUT_DIRECTORY" >&2
  exit 2
fi

source_repo="$1"
output_root="$2"
archive_commit="4d0ecef0a798aab2f769cb5eb2e93982236f4f91"
archive_repo="${PLANNING_ARCHIVE_REPO:-}"

if [[ "$(git -C "${source_repo}" rev-parse --is-inside-work-tree 2>/dev/null || true)" != "true" ]]; then
  echo "SOURCE_REPOSITORY must be a non-bare Git worktree." >&2
  exit 2
fi
if [[ "${output_root}" != /* ]] || [[ -e "${output_root}" ]]; then
  echo "ABSOLUTE_OUTPUT_DIRECTORY must be an absolute path that does not exist." >&2
  exit 2
fi
if [[ -z "${archive_repo}" ]] || ! git -C "${archive_repo}" cat-file -e "${archive_commit}^{commit}"; then
  echo "PLANNING_ARCHIVE_REPO must contain prerequisite commit ${archive_commit}." >&2
  exit 2
fi
for command in git git-filter-repo gitleaks; do
  if ! command -v "${command}" >/dev/null; then
    echo "Missing required command: ${command}" >&2
    exit 2
  fi
done

mkdir -p "${output_root}"
backup_repo="${output_root}/backup.git"
sanitized_repo="${output_root}/sanitized.git"
sanitized_worktree="${output_root}/sanitized-worktree"

git clone --quiet --mirror --no-local "${source_repo}" "${backup_repo}"
git clone --quiet --mirror --no-local "${source_repo}" "${sanitized_repo}"
git -C "${backup_repo}" for-each-ref --format='%(refname)\t%(objectname)' > "${output_root}/refs-before.tsv"

git -C "${sanitized_repo}" filter-repo --force --invert-paths \
  --path .planning/ \
  --path docs/superpowers/plans/ \
  --path docs/superpowers/specs/

cp "${sanitized_repo}/filter-repo/commit-map" "${output_root}/commit-map.tsv"
git -C "${sanitized_repo}" for-each-ref --format='%(refname)\t%(objectname)' > "${output_root}/refs-after.tsv"
git -C "${sanitized_repo}" fsck --full

private_paths="${output_root}/private-paths.txt"
git -C "${sanitized_repo}" rev-list --objects --all | awk '
  $2 == ".planning" || index($2, ".planning/") == 1 ||
  $2 == "docs/superpowers/plans" || index($2, "docs/superpowers/plans/") == 1 ||
  $2 == "docs/superpowers/specs" || index($2, "docs/superpowers/specs/") == 1 { print }
' > "${private_paths}"
if [[ -s "${private_paths}" ]]; then
  echo "Private planning paths remain reachable after filtering." >&2
  exit 1
fi

gitleaks git --log-opts=--all --redact --no-banner "${sanitized_repo}"
git clone --quiet "${sanitized_repo}" "${sanitized_worktree}"
gitleaks dir --redact --no-banner "${sanitized_worktree}"

cat > "${output_root}/CUTOVER-STATUS.txt" <<EOF
archive-prerequisite=${archive_commit}
source-head=$(git -C "${source_repo}" rev-parse HEAD)
backup=${backup_repo}
sanitized-clone=${sanitized_repo}
commit-map=${output_root}/commit-map.tsv
remote-cutover-approved=false
public-visibility-approved=false
EOF

echo "History-sanitization rehearsal passed."
echo "Evidence: ${output_root}"
echo "No remote was changed; cutover and visibility still require explicit approval."
