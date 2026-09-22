"""Red team campaigns against a model's token path and harness."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request

from plastic.api.schemas import RedteamRequest
from plastic.api.service import (
    bad_request,
    list_redteam,
    not_found,
    redteam_results,
    redteam_summary,
    require_model,
    sanitize,
)

router = APIRouter(tags=["redteam"])


@router.get("/redteam")
def list_runs(request: Request) -> list[dict[str, Any]]:
    return sanitize(list_redteam(request.app.state.store))


@router.post("/redteam")
def run(body: RedteamRequest, request: Request) -> dict[str, Any]:
    from plastic.redteam.attack import FAMILIES, AttackConfig, run_redteam

    store = request.app.state.store
    record = require_model(store, body.model_id)
    if not store.model_exists(body.model_id):
        raise bad_request(f"model {body.model_id} has no checkpoint yet")
    if record.get("domain") != "text":
        raise bad_request("the red team attacks text models")
    if not os.path.exists(store.canary_path(body.model_id)):
        raise bad_request(f"model {body.model_id} has no canary suite; calibrate it first")
    data_dir = body.data_dir or (record.get("train_config") or {}).get("data_dir")
    if not data_dir or not os.path.exists(os.path.join(data_dir, "validation.bin")):
        raise bad_request("the red team needs data_dir with validation.bin")
    families = tuple(body.families) if body.families else FAMILIES
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise bad_request(f"unknown attack families: {unknown}")
    cfg = AttackConfig(suffix_len=body.suffix_len, steps=body.steps, families=families)
    try:
        summary = run_redteam(
            store,
            body.model_id,
            cfg=cfg,
            data_dir=data_dir,
            n_prefixes=body.prefixes,
            prefix_len=body.prefix_len,
            device=request.app.state.device,
            record=body.record,
        )
    except (ValueError, FileNotFoundError) as e:
        raise bad_request(str(e)) from e
    out = redteam_summary(store, str(summary["run_id"]))
    if "recorded_payloads" in summary:
        out["recorded_payloads"] = summary["recorded_payloads"]
    return sanitize(out)


@router.get("/redteam/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    try:
        summary = redteam_summary(store, run_id)
    except FileNotFoundError as e:
        raise not_found(str(e)) from e
    return sanitize({"summary": summary, "results": redteam_results(store, run_id)})
