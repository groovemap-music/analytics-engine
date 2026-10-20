#!/usr/bin/env bash
set -euo pipefail

# Disposable PostgreSQL integration tier (see the Justfile's test-integration-pg18 and
# test-integration-pg19 recipes). Starts one container from POSTGRES_INTEGRATION_IMAGE, which
# must already be present locally -- this script never builds or pulls it (disk budget) -- and
# runs INTEGRATION_TESTS (default: the whole suite) against it. No Neo4j: the embedding
# pipeline and the similar-artist stage read and write only PostgreSQL.
suffix="$$"
postgres_container="${POSTGRES_INTEGRATION_CONTAINER:-analytics-engine-postgres-${suffix}}"
postgres_image="${POSTGRES_INTEGRATION_IMAGE:?set POSTGRES_INTEGRATION_IMAGE; use just test-integration-pg18 or test-integration-pg19}"
integration_tests="${INTEGRATION_TESTS:-tests/integration}"
password="${EMBEDDINGS_INTEGRATION_PASSWORD:-integration-test-password}"
postgres_shm_size="${POSTGRES_INTEGRATION_SHM_SIZE:-64m}"

cleanup() {
    docker rm --force --volumes "${postgres_container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if ! docker image inspect "${postgres_image}" >/dev/null 2>&1; then
    echo "Image ${postgres_image} is not present locally." >&2
    echo "PG18: docker pull it once. PG19: build it once with (cd ../database-schema && just test-integration-pg19)." >&2
    echo "This script deliberately never builds or pulls it itself (disk budget)." >&2
    exit 1
fi

docker run --detach --rm \
    --name "${postgres_container}" \
    --publish 127.0.0.1::5432 \
    --shm-size "${postgres_shm_size}" \
    --env POSTGRES_USER=groovemap \
    --env "POSTGRES_PASSWORD=${password}" \
    --env POSTGRES_DB=groovemap \
    "${postgres_image}" >/dev/null

postgres_ready=false
for _attempt in $(seq 1 60); do
    if docker exec "${postgres_container}" pg_isready --username groovemap --dbname groovemap >/dev/null 2>&1; then
        postgres_ready=true
        break
    fi
    sleep 2
done

if [[ "${postgres_ready}" != true ]]; then
    docker logs "${postgres_container}" >&2
    echo "PostgreSQL did not become ready within 120 seconds" >&2
    exit 1
fi

postgres_published="$(docker port "${postgres_container}" 5432/tcp)"
postgres_port="${postgres_published##*:}"

POSTGRES_HOST="127.0.0.1:${postgres_port}" \
POSTGRES_DATABASE=groovemap \
POSTGRES_USERNAME=groovemap \
POSTGRES_PASSWORD="${password}" \
EMBEDDING_PIPELINE_POSTGRES_USERNAME=embedding_pipeline_login \
EMBEDDING_PIPELINE_POSTGRES_PASSWORD="${password}" \
    uv run pytest -m integration ${integration_tests}
