#!/usr/bin/env bash
set -euo pipefail

# Advisory tier for the embedding pipeline (gm-analytics-engine-ieu.2). Starts one disposable
# PostgreSQL 19 + pgvector container from the existing local image database-schema built
# (`database-schema-postgres19-pgvector:local`, from that repository's own
# `just test-integration-pg19`) — this script never builds it. No Neo4j: the embedding
# pipeline reads only the `graph` schema in PostgreSQL.
suffix="$$"
postgres_container="${POSTGRES_INTEGRATION_CONTAINER:-analytics-engine-postgres-${suffix}}"
postgres_image="${POSTGRES_INTEGRATION_IMAGE:-database-schema-postgres19-pgvector:local}"
password="${EMBEDDINGS_INTEGRATION_PASSWORD:-integration-test-password}"
postgres_shm_size="${POSTGRES_INTEGRATION_SHM_SIZE:-64m}"

cleanup() {
    docker rm --force --volumes "${postgres_container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if ! docker image inspect "${postgres_image}" >/dev/null 2>&1; then
    echo "Image ${postgres_image} is not present locally." >&2
    echo "Build it once with: (cd ../database-schema && just test-integration-pg19)" >&2
    echo "This script deliberately never builds it itself (disk budget)." >&2
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
    uv run pytest -m integration tests/integration
