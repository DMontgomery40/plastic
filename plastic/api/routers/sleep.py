"""Sleep: consolidate a model's accepted session learning into a child model, as a background process."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from plastic.api.schemas import SleepRequest
from plastic.api.service import bad_request, not_found, require_model, sanitize

router = APIRouter(tags=["sleep"])


@router.post("/models/{model_id}/sleep")
def start_sleep(model_id: str, body: SleepRequest, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    record = require_model(store, model_id)
    if record.get("backend") != "ttt":
        raise bad_request("sleep is available for TTT chat models; this record has no fast-weight backend")
    if not store.model_exists(model_id):
        raise bad_request(f"model {model_id} has no checkpoint yet")
    if body.sessions:
        known = {m["session_id"] for m in store.list_sessions() if m.get("model_id") == model_id}
        missing = [s for s in body.sessions if s not in known]
        if missing:
            raise bad_request(f"sessions not found for {model_id}: {missing}")
    options = body.model_dump(exclude={"sessions", "probes"}, exclude_none=True)
    probes = [p.model_dump() for p in body.probes] if body.probes else None
    job = request.app.state.sleep_jobs.start(model_id, options, sessions=body.sessions, probes=probes)
    return sanitize(job)


@router.get("/sleep")
def list_sleep(request: Request) -> list[dict[str, Any]]:
    return sanitize(request.app.state.sleep_jobs.list())


@router.get("/sleep/{run_id}")
def sleep_status(run_id: str, request: Request) -> dict[str, Any]:
    try:
        status = request.app.state.sleep_jobs.status(run_id)
    except FileNotFoundError as e:
        raise not_found(str(e)) from e
    if status["status"] == "accepted" and status.get("report", {}).get("model_id"):
        # a new model exists: make sure the catalog the client reads next is current
        request.app.state.registry.invalidate_model(status["report"]["model_id"])
    return sanitize(status)
