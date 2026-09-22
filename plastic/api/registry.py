"""Open sessions held across requests, with one loaded model per model id.

``Session`` loads its own checkpoint in ``__init__``, so model sharing is done
by handing it a store whose ``load_checkpoint`` memoizes. Two sessions of the
same model then hold the same ``nn.Module``; compute on it is serialized by a
per-model lock, and the per-session lock keeps two requests from touching one
session's state at the same time. Locks are always taken session first, then
model.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from plastic.harness.config import HarnessConfig
from plastic.session.runner import Session
from plastic.store import ArtifactStore


class SignatureMismatch(RuntimeError):
    """The session on disk was created against a different checkpoint of its model."""


class ModelCachingStore(ArtifactStore):
    """An artifact store that returns the same loaded model for repeated calls."""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self._models: dict[tuple[str, str], tuple[Any, Any, dict[str, Any]]] = {}
        self._guard = threading.Lock()

    def load_checkpoint(self, model_id: str, device: torch.device | str = "cpu"):  # type: ignore[override]
        key = (model_id, str(device))
        with self._guard:
            hit = self._models.get(key)
        if hit is not None:
            return hit
        loaded = super().load_checkpoint(model_id, device)
        with self._guard:
            return self._models.setdefault(key, loaded)

    def drop_model(self, model_id: str) -> None:
        with self._guard:
            for key in [k for k in self._models if k[0] == model_id]:
                self._models.pop(key, None)


class SessionRegistry:
    def __init__(self, artifacts_root: str, *, device: str = "cpu") -> None:
        self.store = ModelCachingStore(artifacts_root)
        self.device = device
        self._sessions: dict[str, Session] = {}
        self._session_locks: dict[str, threading.Lock] = {}
        self._model_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ locks
    def session_lock(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._session_locks.setdefault(session_id, threading.Lock())

    def model_lock(self, model_id: str) -> threading.Lock:
        with self._guard:
            return self._model_locks.setdefault(model_id, threading.Lock())

    # ------------------------------------------------------------------ sessions
    def _open(self, session_id: str) -> Session:
        with self._guard:
            session = self._sessions.get(session_id)
        if session is not None:
            return session
        try:
            session = Session.open(self.store, session_id, device=self.device)
        except ValueError as e:  # the store refuses a session whose model has moved
            raise SignatureMismatch(str(e)) from e
        with self._guard:
            return self._sessions.setdefault(session_id, session)

    @contextmanager
    def use(self, session_id: str) -> Iterator[Session]:
        """Yield the open session with its session lock and its model lock held."""
        with self.session_lock(session_id):
            session = self._open(session_id)
            with self.model_lock(session.model_id):
                yield session

    def create(
        self,
        *,
        model_id: str,
        session_id: str | None = None,
        harness: HarnessConfig | None = None,
    ) -> Session:
        with self.model_lock(model_id):
            session = Session.create(
                self.store,
                model_id=model_id,
                harness_cfg=harness or HarnessConfig(),
                session_id=session_id,
                device=self.device,
            )
        with self._guard:
            self._sessions[session.session_id] = session
        return session

    def close(self, session_id: str) -> None:
        with self._guard:
            self._sessions.pop(session_id, None)

    def invalidate_model(self, model_id: str) -> None:
        """Drop the cached model and every open session of it (state is on disk)."""
        self.store.drop_model(model_id)
        with self._guard:
            for sid in [s for s, sess in self._sessions.items() if sess.model_id == model_id]:
                self._sessions.pop(sid, None)
