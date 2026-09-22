"""The FastAPI application: one service over one artifact store.

The API never inspects text for safety. It exposes what the harness, the
training loop, and the red team already produced.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from plastic.api.jobs import JobManager
from plastic.api.registry import SessionRegistry
from plastic.api.routers import data, health, models, redteam, sessions, sleep, train
from plastic.store import ArtifactStore

ROUTERS = (health.router, data.router, models.router, sessions.router, train.router, redteam.router, sleep.router)


async def _not_found(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


async def _conflict(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=409)


def create_app(artifacts_root: str, device: str = "cpu") -> FastAPI:
    store = ArtifactStore(artifacts_root)
    store.ensure()
    store.ensure_sessions()

    app = FastAPI(
        title="plastic",
        description="A test-time-training state-space model with a transactional safety harness.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # credentials and a wildcard origin cannot be combined
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.device = device
    app.state.artifacts_root = store.root
    app.state.registry = SessionRegistry(store.root, device=device)
    app.state.jobs = JobManager(store, device=device)

    for router in ROUTERS:
        app.include_router(router, prefix="/api")

    app.add_exception_handler(FileNotFoundError, _not_found)
    app.add_exception_handler(FileExistsError, _conflict)
    return app
