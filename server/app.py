# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# FastAPI application for the Cyber Defense Environment.

# This module creates an HTTP server that exposes the CyberDefenseEnvironment
# over HTTP and WebSocket endpoints, compatible with EnvClient.

# Endpoints:
#     - POST /reset: Reset the environment
#     - POST /step: Execute an action
#     - GET /state: Get current environment state
#     - GET /schema: Get action/observation schemas
#     - WS /ws: WebSocket endpoint for persistent sessions

# Usage:
#     # Development (with auto-reload):
#     uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

#     # Production:
#     uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

#     # Or run directly:
#     python -m server.app
# """

# try:
#     from openenv.core.env_server.http_server import create_app
# except Exception as e:  # pragma: no cover
#     raise ImportError(
#         "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
#     ) from e

# try:
#     from ..models import CyberDefenseAction, CyberDefenseObservation
#     from .cyber_defense_environment import CyberDefenseEnvironment
# except ModuleNotFoundError:
#     from models import CyberDefenseAction, CyberDefenseObservation
#     from server.cyber_defense_environment import CyberDefenseEnvironment


# # Create the app with web interface and README integration
# app = create_app(
#     CyberDefenseEnvironment,
#     CyberDefenseAction,
#     CyberDefenseObservation,
#     env_name="cyber_defense",
#     max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
# )


# def main(host: str = "0.0.0.0", port: int = 8000):
#     """
#     Entry point for direct execution via uv run or python -m.

#     This function enables running the server without Docker:
#         uv run --project . server
#         uv run --project . server --port 8001
#         python -m cyber_defense.server.app

#     Args:
#         host: Host address to bind to (default: "0.0.0.0")
#         port: Port number to listen on (default: 8000)

#     For production deployments, consider using uvicorn directly with
#     multiple workers:
#         uvicorn cyber_defense.server.app:app --workers 4
#     """
#     import uvicorn

#     uvicorn.run(app, host=host, port=port)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=8000)
#     args = parser.parse_args()
#     main(port=args.port)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Adaptive Cyber Defense Environment.

Exposes CyberDefenseEnvironment via OpenEnv's create_app factory.
The factory registers all required endpoints automatically:
    POST /reset   — Reset environment (accepts seed, task kwargs via WS)
    POST /step    — Execute a defensive action
    GET  /state   — Return full hidden state
    GET  /schema  — Return action + observation + state JSON schemas
    GET  /health  — Liveness check
    WS   /ws      — WebSocket for persistent sessions (used by EnvClient)

DO NOT re-register /reset, /step, or /state manually — create_app
already registers them and double-registration causes route conflicts.

Task switching is handled via the task= kwarg passed through EnvClient.reset()
→ WS {type:'reset', data:{task:'...', seed:42}} → environment.reset(task=...)
The CyberDefenseEnvironment.reset() signature accepts task= and switches
its internal config, so a single WS session can run all 3 tasks in sequence.

Usage:
    # Development (auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Direct:
    python -m server.app
"""

from fastapi.middleware.cors import CORSMiddleware

try:
    from openenv_core import create_app
except ImportError:
    try:
        from openenv.core import create_app
    except ImportError as e:
        raise ImportError(
            "openenv-core is required. Install with:\n    pip install openenv-core\n"
        ) from e

try:
    from ..models import (
        CyberDefenseAction,
        CyberDefenseObservation,
    )
    from .cyber_defense_environment import CyberDefenseEnvironment
except ModuleNotFoundError:
    from models import (
        CyberDefenseAction,
        CyberDefenseObservation,
    )
    from server.cyber_defense_environment import CyberDefenseEnvironment


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
# create_app requires a zero-argument callable that returns a fresh Environment.
# CyberDefenseEnvironment defaults to "easy_breach_prevention".
# Task switching happens via reset(task=...) over the WebSocket session —
# no need to bake the task into the factory.

def _env_factory() -> CyberDefenseEnvironment:
    """Zero-argument factory: creates a default env instance per WS session."""
    return CyberDefenseEnvironment(task_name="easy_breach_prevention")


# ---------------------------------------------------------------------------
# Create the OpenEnv-compatible FastAPI app
# ---------------------------------------------------------------------------

app = create_app(
    env=_env_factory,
    action_cls=CyberDefenseAction,
    observation_cls=CyberDefenseObservation,
    env_name="cyber_defense",
    max_concurrent_envs=4,  # SUPPORTS_CONCURRENT_SESSIONS=True allows this
)

# Add CORS so HF Space browser UI can reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run("cyber_defense.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
