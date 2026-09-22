"""Prepared corpora under ``<artifacts_root>/data``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from plastic.api.service import data_sets, sanitize

router = APIRouter(tags=["data"])


@router.get("/data")
def list_data(request: Request) -> list[dict[str, Any]]:
    return sanitize(data_sets(request.app.state.store))
