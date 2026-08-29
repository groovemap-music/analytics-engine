# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32

ARG PYTHON_IMAGE=python:3.14.5-slim@sha256:c845af9399020c7e562969a13689e929074a10fd057acd1b1fad06a2fb068e97

FROM ${PYTHON_IMAGE} AS builder

RUN pip install --no-cache-dir uv==0.12.5
WORKDIR /app

COPY .build/runtime/*.whl /wheels/
COPY .build/requirements.txt /wheels/
COPY dist/groovemap_analytics_engine-*.whl /wheels/

RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python --require-hashes --requirements /wheels/requirements.txt && \
    uv pip install --python /app/.venv/bin/python --no-deps /wheels/*.whl && \
    find /app/.venv -type f -name '*.py[co]' -delete && \
    find /app/.venv -type d -name __pycache__ -prune -exec rm -rf '{}' +

FROM ${PYTHON_IMAGE}

ARG BUILD_DATE
ARG BUILD_VERSION=0.1.0
ARG VCS_REF

RUN case "${VCS_REF}" in *[!0-9a-f]*|"") exit 1 ;; esac && \
    [ "${#VCS_REF}" -eq 40 ]

LABEL org.opencontainers.image.title="analytics-engine" \
      org.opencontainers.image.description="Compute and cache scheduled music analytics" \
      org.opencontainers.image.authors="Robert Wlodarczyk <robert@simplicityguy.com>" \
      org.opencontainers.image.url="https://groovemap.music" \
      org.opencontainers.image.documentation="https://github.com/groovemap-music/analytics-engine/blob/main/README.md" \
      org.opencontainers.image.source="https://github.com/groovemap-music/analytics-engine" \
      org.opencontainers.image.vendor="GrooveMap" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="docker.io/library/python:3.14.5-slim"

RUN groupadd --gid 1000 groovemap && \
    useradd --uid 1000 --gid groovemap --create-home --shell /usr/sbin/nologin groovemap && \
    mkdir -p /app /logs && \
    chown -R 1000:1000 /app /logs

WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv

ENV HOME=/home/groovemap \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8008 8009
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8009/health', timeout=5).read()"]

USER 1000:1000
ENTRYPOINT ["/app/.venv/bin/analytics-engine"]
