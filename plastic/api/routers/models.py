"""Trained models: listing, detail, training log, and calibration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from plastic.api.schemas import CalibrateRequest
from plastic.api.service import (
    bad_request,
    calibration_payload,
    model_detail,
    model_summary,
    require_model,
    sanitize,
)

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models(request: Request) -> list[dict[str, Any]]:
    store = request.app.state.store
    return sanitize([model_summary(store, rec) for rec in store.list_models()])


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    require_model(store, model_id)
    return sanitize(model_detail(store, model_id))


@router.get("/models/{model_id}/log")
def get_model_log(model_id: str, request: Request, limit: int = Query(default=200, ge=1, le=10000)) -> list[dict[str, Any]]:
    store = request.app.state.store
    require_model(store, model_id)
    return sanitize(store.read_log(model_id, limit=limit))


@router.post("/models/{model_id}/calibrate")
def calibrate(model_id: str, body: CalibrateRequest, request: Request) -> dict[str, Any]:
    from plastic.harness.calibrate import calibrate_model

    store = request.app.state.store
    record = require_model(store, model_id)
    if not store.model_exists(model_id):
        raise bad_request(f"model {model_id} has no checkpoint yet")
    data_dir = body.data_dir or (record.get("train_config") or {}).get("data_dir")
    if record.get("domain") == "text" and not data_dir:
        raise bad_request("text calibration needs data_dir with validation.bin")
    try:
        cal = calibrate_model(
            store,
            model_id,
            data_dir=data_dir,
            n_chunks=body.chunks,
            fisher_chunks=body.fisher_chunks,
            target_fpr=body.fpr,
            device=request.app.state.device,
        )
    except (ValueError, FileNotFoundError) as e:
        raise bad_request(str(e)) from e
    request.app.state.registry.invalidate_model(model_id)
    return sanitize(calibration_payload(cal))
