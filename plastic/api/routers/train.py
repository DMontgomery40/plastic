"""Local training jobs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from plastic.api.schemas import TrainRequest
from plastic.api.service import bad_request, conflict, not_found, sanitize

router = APIRouter(tags=["train"])


@router.get("/train/jobs")  # declared before /train/{model_id} so "jobs" is not read as an id
def list_jobs(request: Request) -> list[dict[str, Any]]:
    return sanitize(request.app.state.jobs.jobs())


@router.post("/train")
def start_training(body: TrainRequest, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    jobs = request.app.state.jobs
    if body.domain == "text" and not body.data_dir:
        raise bad_request("text training needs data_dir with train.bin and validation.bin")
    if body.model_id:
        try:
            store.load_model_record(body.model_id)
        except FileNotFoundError:
            pass
        else:
            raise conflict(f"model already exists: {body.model_id}")
    return sanitize(jobs.start(body))


@router.get("/train/{model_id}")
def training_status(model_id: str, request: Request) -> dict[str, Any]:
    status = request.app.state.jobs.status(model_id)
    if status is None:
        raise not_found(f"no training job or model record for: {model_id}")
    return sanitize(status)


@router.post("/train/{model_id}/cancel")
def cancel_training(model_id: str, request: Request) -> dict[str, Any]:
    result = request.app.state.jobs.cancel(model_id)
    if result is None:
        raise not_found(f"no training job for: {model_id}")
    return sanitize(result)
