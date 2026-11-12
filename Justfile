set shell := ["bash", "-euo", "pipefail", "-c"]

# Digest-pinned official PostgreSQL 18 image for the integration tier. CI pulls it; the local script never does.
pg18_image := "postgres:18-alpine@sha256:d3e1620b530c944afa6e887d22eb899824da68e19c52024bf98f5220c88a65b2"
# Official pgvector image on PostgreSQL 18 (linux/amd64 manifest verified), for the embedding-pipeline suite.
pgvector_pg18_image := "pgvector/pgvector:pg18@sha256:c29f30688a9ad961691bdf0ec1650717cc132a5814619d80f80463d6e4c42e47"

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

# Similar-artist tier: applies database-schema's own create_postgres_schema (the pinned
# groovemap-database-schema runtime dependency; see tests/integration/conftest.py) to a
# disposable official PostgreSQL 18 container -- the same digest database-schema's required
# tier uses -- and runs the synthetic compact-COPY, publish and retire suite under the real
# embedding_pipeline role, the style atlas's monthly writes against the real atlas tables, and the
# rarity replacement, including a dropped connection mid-run.
# Neither needs pgvector, so the bare image is enough.
# There is no PostgreSQL 19 GA image yet; this is the tier that stands in for it.
# Also runs the data-completeness and interrupted-run suites, which need only the shared schema.
# Not run here: test_embedding_pipeline_integration.py and test_similar_artists_job_role_integration.py, which need pgvector (artist_embeddings, halfvec).
test-integration-pg18:
    POSTGRES_INTEGRATION_IMAGE={{pg18_image}} \
        INTEGRATION_TESTS="tests/integration/test_similar_artists_integration.py tests/integration/test_style_atlas_integration.py tests/integration/test_rarity_integration.py tests/integration/test_label_longevity_integration.py tests/integration/test_data_completeness_integration.py tests/integration/test_interrupted_runs_integration.py" \
        bash scripts/test-integration.sh -v

# CI entry point: pull the pinned images (the script itself never pulls), then run the PG18 and pgvector tiers verbosely.
test-integration-ci:
    docker pull {{pg18_image}}
    docker pull {{pgvector_pg18_image}}
    just test-integration-pg18
    just test-integration-pgvector

# pgvector tier: the embedding-pipeline suite and the nightly similar-artist job run as the restricted
# embedding_pipeline LOGIN role, on the digest-pinned pgvector PG18 image. CI-only: it pulls the image
# first, because the local script never does.
test-integration-pgvector:
    docker pull {{pgvector_pg18_image}}
    POSTGRES_INTEGRATION_IMAGE={{pgvector_pg18_image}} \
        INTEGRATION_TESTS="tests/integration/test_embedding_pipeline_integration.py tests/integration/test_similar_artists_job_role_integration.py" \
        bash scripts/test-integration.sh -v

# Advisory tier: the same schema, plus the vector extension and public.artist_embeddings, on
# the existing local PG19 beta + pgvector image, running the whole integration suite including
# the embedding pipeline's. Never builds the image itself -- reuses database-schema's own
# `database-schema-postgres19-pgvector:local` tag, built by that repository's
# `test-integration-pg19` recipe.
test-integration-pg19:
    POSTGRES_INTEGRATION_IMAGE=database-schema-postgres19-pgvector:local bash scripts/test-integration.sh

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
    # The default entrypoint dispatches subcommands (args-only schedulers); an unknown one exits 2, never serving.
    docker run --rm analytics-engine:local similar-artists --help | grep -q -- --spool-dir
    docker run --rm analytics-engine:local --help | grep -q 'usage: analytics-engine'
    test "$(docker run --rm analytics-engine:local bogus >/dev/null 2>&1; echo $?)" = "2"
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
