"""Sleep: consolidate session traces into the slow weights, gated by the canaries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from plastic.api.schemas import SleepRequest
from plastic.api.service import bad_request, require_model, sanitize

router = APIRouter(tags=["sleep"])


@router.post("/sleep")
def sleep(body: SleepRequest, request: Request) -> dict[str, Any]:
    from plastic.sleep.consolidate import consolidate

    store = request.app.state.store
    record = require_model(store, body.model_id)
    if not store.model_exists(body.model_id):
        raise bad_request(f"model {body.model_id} has no checkpoint yet")
    if record.get("domain") != "text":
        raise bad_request("sleep consolidation is defined for text models")
    try:
        manifest = consolidate(
            store,
            body.model_id,
            sessions=body.sessions,
            core_data_dir=body.core_data_dir,
            steps=body.steps,
            lr=body.lr,
            core_ratio=body.core_ratio,
            seq_len=body.seq_len,
            batch_size=body.batch_size,
            device=request.app.state.device,
            tolerance=body.tolerance,
        )
    except (ValueError, FileNotFoundError) as e:
        raise bad_request(str(e)) from e
    return sanitize(manifest)
