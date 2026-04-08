# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---------------------------------------------------------------------------
# Adaptive Cyber Defense OpenEnv — Dockerfile
#
# Place this file at the project root (cyber-defense-openenv/).
# Build and run from the root:
#
#   docker build -t cyber_defense-env:latest .
#   docker run -p 8000:8000 cyber_defense-env:latest
#
# The server will be available at http://localhost:8000
# Health check: GET http://localhost:8000/health
# Schema:       GET http://localhost:8000/schema
# WebSocket:    WS  http://localhost:8000/ws
#
# The CMD uses `server.app:app` (matching openenv.yaml `app: server.app:app`)
# because we `cd /app/env` before launching, making `cyber_defense/server/app.py`
# addressable as `server.app`.
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git + curl required for VCS-sourced deps and uv install fallback
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=cyber_defense

# ---------------------------------------------------------------------------
# Copy full project into the build context.
# The Dockerfile sits at root, so `.` copies:
#   cyber_defense/           ← environment package (server/, models.py, client.py, ...)
#   inference.py             ← inference script (must be at root per PS requirement)
#   pyproject.toml
#   uv.lock (if present)
#   openenv.yaml
#   README.md
# ---------------------------------------------------------------------------
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available (base image may or may not include it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Step 1: install dependencies only — cached layer rebuilt only when
#         pyproject.toml / uv.lock changes, not on source code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Step 2: install the project itself (editable install for the package)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ---------------------------------------------------------------------------
# Runtime stage — lean final image
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy virtualenv from builder (all deps pre-installed, no rebuild needed)
COPY --from=builder /app/env/.venv /app/.venv

# Copy the full project (env code + inference.py at root + openenv.yaml + README)
COPY --from=builder /app/env /app/env

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

# Use the venv for all python/uvicorn calls
ENV PATH="/app/.venv/bin:$PATH"

# PYTHONPATH enables both import styles:
#   from cyber_defense.models import ...   (via /app/env)
#   from models import ...                 (via /app/env/cyber_defense, for flat imports)
#   from server.app import ...             (via /app/env/cyber_defense, for CMD below)
ENV PYTHONPATH="/app/env:/app/env/cyber_defense:$PYTHONPATH"

# Enable the OpenEnv Gradio web interface at /web (served alongside the API)
ENV ENABLE_WEB_INTERFACE=true

# Default server config — override at docker run time with -e PORT=xxxx
ENV HOST=0.0.0.0
ENV PORT=7860

# ---------------------------------------------------------------------------
# Expose & health check
# ---------------------------------------------------------------------------

EXPOSE 7860

# Start-period gives uvicorn time to boot before health checks begin.
# /health is a lightweight GET that always returns 200 if the server is up.
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ---------------------------------------------------------------------------
# Runtime command
#
# `cd /app/env/cyber_defense` makes `server.app:app` resolve correctly to
# cyber_defense/server/app.py — this matches openenv.yaml `app: server.app:app`
# exactly, so openenv validate sees the same module path the container uses.
#
# workers=1 keeps memory under the 8 GB / vcpu=2 constraint from the PS.
# Log-level info ensures [START]/[STEP]/[END] lines are visible in docker logs.
# ---------------------------------------------------------------------------
# Uses the full dotted module path which works from any working directory,
# because PYTHONPATH includes /app/env where cyber_defense/ package lives.
# Matches openenv.yaml `app: cyber_defense.server.app:app`.
CMD ["uvicorn", "cyber_defense.server.app:app",\
     "--host", "0.0.0.0",\
    "--port", "7860",\
     "--workers", "1",\
     "--log-level", "info"]