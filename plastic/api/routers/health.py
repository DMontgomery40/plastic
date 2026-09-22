"""Liveness and a count of what the store holds."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    store = request.app.state.store
    return {
        "ok": True,
        "artifacts_root": store.root,
        "device": request.app.state.device,
        "n_models": len(store.list_models()),
        "n_sessions": len(store.list_sessions()),
    }
