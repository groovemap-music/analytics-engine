set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

setup:
    uv sync --dev --frozen

source-check: format-check lint contract-check repository-check

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

contract-check:
    uv run python scripts/check-contracts.py

repository-check:
    uv run python scripts/check-repository-compliance.py

secret-scan:
    gitleaks git --redact --no-banner
    gitleaks dir . --redact --no-banner

check: source-check typecheck coverage secret-scan build install-check license-check release-artifacts bump-preview

format:
    uv run ruff format .
    uv run ruff check --fix .

typecheck:
    uv run mypy

test:
    uv run pytest -m "not integration" --cov=insights --cov-report=term-missing --cov-report=xml

coverage: test

# Advisory tier: applies a minimal stand-in for ADR 0013's schema (embedding_pipeline role
# and grants, the vector extension, public.artist_embeddings, and the graph edges this
# pipeline reads — see tests/integration/conftest.py) to the existing local PG19+pgvector
# image, and runs the embedding-pipeline suite that needs a live embedding_pipeline-scoped
# connection. Never builds the image itself — reuses database-schema's own
# `database-schema-postgres19-pgvector:local` tag, built by that repository's
# `test-integration-pg19` recipe.
test-integration-pg19:
    bash scripts/test-integration-pg19.sh

build:
    uv build --out-dir dist --clear

install-check: build
    bash scripts/install-check.sh

license-check: build
    uv run python scripts/check-license.py
    uv run pip-licenses --ignore-packages groovemap-analytics-engine --fail-on "GPL-2.0-only;GPL-3.0-only;AGPL-3.0-only"

audit:
    uv run pip-audit

prepare-runtime-wheel:
    bash scripts/prepare-runtime-wheel.sh

image: build prepare-runtime-wheel
    bash scripts/build-image.sh
    docker run --rm --entrypoint /app/.venv/bin/python analytics-engine:local -c 'import insights.insights'
    test "$(docker run --rm --entrypoint /usr/bin/id analytics-engine:local -u):$(docker run --rm --entrypoint /usr/bin/id analytics-engine:local -g)" = "1000:1000"
    uv run python scripts/check-image-metadata.py analytics-engine:local

bump-preview:
    uv run python scripts/check_bump_preview.py

# Update local version metadata and changelog only; do not commit, tag, push, or publish.
bump:
    uv run cz bump --version-files-only --changelog --yes --check-consistency
    uv lock

release-artifacts: build install-check
    bash scripts/release-dry-run.sh

release-dry-run: check
