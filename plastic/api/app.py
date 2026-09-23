"""The FastAPI application: one service over one artifact store.

The API never inspects text for safety. It exposes sessions over the transaction harness and the
registered models; sleep runs as a background process per job; training, red team and physics are CLI research tools.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from plastic.api.registry import SessionRegistry
from plastic.api.routers import health, models, sessions, sleep
from plastic.api.sleep_jobs import SleepJobs
from plastic.store import ArtifactStore

ROUTERS = (health.router, models.router, sessions.router, sleep.router)

# what a client may do; a deployment passes a restricted set and the UI renders exactly that
DEFAULT_CAPABILITIES = {"create_session": True, "fork": True, "reset": True, "delete": True, "resume": True, "calibrate": True, "sleep": True}


async def _not_found(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


async def _conflict(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=409)


def create_app(artifacts_root: str, device: str = "cpu", *, capabilities: dict[str, bool] | None = None, public: bool = False) -> FastAPI:
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
    app.state.sleep_jobs = SleepJobs(store.root, device=device)
    app.state.capabilities = {**DEFAULT_CAPABILITIES, **(capabilities or {})}
    app.state.public = bool(public)

    for router in ROUTERS:
        app.include_router(router, prefix="/api")

    app.add_exception_handler(FileNotFoundError, _not_found)
    app.add_exception_handler(FileExistsError, _conflict)
    return app
