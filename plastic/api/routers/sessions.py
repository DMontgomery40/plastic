"""Sessions: lineage, transactions, state, chat, physics episodes, and control."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from fastapi import APIRouter, Query, Request

from plastic.api.registry import SignatureMismatch
from plastic.api.schemas import ChatRequest, CreateSessionRequest, ForkRequest, PhysicsRequest
from plastic.api.service import (
    bad_request,
    conflict,
    lineage,
    not_found,
    require_session_meta,
    sanitize,
    session_meta_payload,
    session_summary,
    state_payload,
    transactions_page,
    transactions_tail,
)
from plastic.harness.config import HarnessConfig
from plastic.session.runner import Session

router = APIRouter(tags=["sessions"])


@contextmanager
def _open(request: Request, session_id: str) -> Iterator[Session]:
    store = request.app.state.store
    if not store.session_exists(session_id):
        raise not_found(f"session not found: {session_id}")
    try:
        cm = request.app.state.registry.use(session_id)
        session = cm.__enter__()
    except SignatureMismatch as e:
        raise conflict(str(e)) from e
    except FileNotFoundError as e:
        raise not_found(str(e)) from e
    try:
        yield session
    finally:
        cm.__exit__(None, None, None)


@router.get("/sessions")
def list_sessions(request: Request) -> list[dict[str, Any]]:
    return sanitize(request.app.state.store.list_sessions())


@router.post("/sessions")
def create_session(body: CreateSessionRequest, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    registry = request.app.state.registry
    if not store.model_exists(body.model_id):
        raise not_found(f"model not found: {body.model_id}")
    if body.session_id and store.session_exists(body.session_id):
        raise conflict(f"session already exists: {body.session_id}")
    try:
        harness = HarnessConfig.from_dict({**HarnessConfig().to_dict(), **(body.harness or {})})
    except (ValueError, TypeError) as e:
        raise bad_request(str(e)) from e
    try:
        session = registry.create(model_id=body.model_id, session_id=body.session_id, harness=harness)
    except FileExistsError as e:
        raise conflict(str(e)) from e
    return sanitize(session_summary(store, store.load_session_meta(session.session_id)))


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    meta = require_session_meta(store, session_id)
    with _open(request, session_id) as session:
        summary = session.summary()
    return sanitize(
        {
            "meta": session_meta_payload(store, meta),
            "summary": summary,
            "lineage": lineage(store, session_id),
            "transactions": transactions_tail(store, session_id, 100),
            "trace": store.read_trace(session_id, limit=50),
        }
    )


@router.get("/sessions/{session_id}/transactions")
def get_transactions(
    session_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    store = request.app.state.store
    require_session_meta(store, session_id)
    return sanitize(transactions_page(store, session_id, limit=limit, offset=offset))


@router.get("/sessions/{session_id}/state")
def get_state(session_id: str, request: Request) -> dict[str, Any]:
    with _open(request, session_id) as session:
        return sanitize(state_payload(session))


@router.post("/sessions/{session_id}/chat")
def chat(session_id: str, body: ChatRequest, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    meta = require_session_meta(store, session_id)
    if meta.get("domain") != "text":
        raise bad_request(f"session {session_id} is a {meta.get('domain')} session; chat requires a text session")
    with _open(request, session_id) as session:
        result = session.chat(
            body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            seed=body.seed,
        )
    return sanitize(result.to_dict())


@router.post("/sessions/{session_id}/physics")
def physics(session_id: str, body: PhysicsRequest, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    meta = require_session_meta(store, session_id)
    if meta.get("domain") != "physics":
        raise bad_request(f"session {session_id} is a {meta.get('domain')} session; episodes require a physics session")
    with _open(request, session_id) as session:
        result = session.physics_episode(steps=body.steps, mu=body.mu, seed=body.seed, nonlinear=body.nonlinear)
    return sanitize(result.to_dict())


@router.post("/sessions/{session_id}/fork")
def fork(session_id: str, body: ForkRequest, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    require_session_meta(store, session_id)
    if body.child_session_id and store.session_exists(body.child_session_id):
        raise conflict(f"session already exists: {body.child_session_id}")
    with _open(request, session_id) as session:
        try:
            child = session.fork(body.child_session_id)
        except FileExistsError as e:
            raise conflict(str(e)) from e
        except ValueError as e:
            raise conflict(str(e)) from e
    return sanitize(session_summary(store, store.load_session_meta(child)))


@router.post("/sessions/{session_id}/reset")
def reset(session_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    with _open(request, session_id) as session:
        meta = session.reset()
    return sanitize(session_summary(store, meta))


@router.post("/sessions/{session_id}/resume")
def resume(session_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    with _open(request, session_id) as session:
        meta = session.resume()
    return sanitize(session_summary(store, meta))


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.store
    require_session_meta(store, session_id)
    request.app.state.registry.close(session_id)
    store.delete_session(session_id)
    return {"deleted": True}
