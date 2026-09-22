"""On-disk artifact store. Models live under ``<root>/models/<model_id>/``.

Layout:
    models/index.json            { "models": { model_id: record } }
    models/<id>/config.json      ModelConfig
    models/<id>/checkpoint.pt    { config, model_state, step, extra }
    models/<id>/tokenizer.json   text models only
    models/<id>/train_log.jsonl
    models/<id>/eval.json

Sessions are added to the same store by the harness milestone.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import torch
from torch import nn

from plastic.config import ModelConfig


def atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, rec: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def read_jsonl(path: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    if limit is not None and len(out) > limit:
        return out[-limit:]
    return out


class ArtifactStore:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.models_dir = os.path.join(self.root, "models")

    # ---- paths ----
    @property
    def models_index(self) -> str:
        return os.path.join(self.models_dir, "index.json")

    def model_dir(self, model_id: str) -> str:
        return os.path.join(self.models_dir, model_id)

    def config_path(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "config.json")

    def checkpoint_path(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "checkpoint.pt")

    def tokenizer_path(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "tokenizer.json")

    def train_log_path(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "train_log.jsonl")

    def eval_path(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "eval.json")

    # ---- registry ----
    def ensure(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)
        if not os.path.exists(self.models_index):
            atomic_write_json(self.models_index, {"models": {}})

    def _load_index(self) -> dict[str, Any]:
        self.ensure()
        idx = read_json(self.models_index)
        models = idx.get("models", {}) if isinstance(idx, dict) else {}
        return {"models": dict(models) if isinstance(models, dict) else {}}

    def new_model_id(self, prefix: str = "lm") -> str:
        self.ensure()
        base = f"{prefix}_{int(time.time())}"
        known = set(self._load_index()["models"].keys())
        mid, n = base, 1
        while os.path.exists(self.model_dir(mid)) or mid in known:
            mid = f"{base}_{n}"
            n += 1
        return mid

    def register_model(self, model_id: str, record: dict[str, Any]) -> None:
        idx = self._load_index()
        rec = dict(idx["models"].get(model_id, {}))
        # A record that declares a fresh terminal/lifecycle status is authoritative about the
        # run's outcome: a later success (or a retry) at the same model_id must not inherit a
        # previous attempt's stale error. Only an incoming record that carries its own error
        # keeps one.
        if "status" in record and "error" not in record:
            rec.pop("error", None)
        rec.update(record)
        rec["model_id"] = model_id
        rec.setdefault("created_at_unix", int(time.time()))
        rec["updated_at_unix"] = int(time.time())
        idx["models"][model_id] = rec
        atomic_write_json(self.models_index, idx)

    def list_models(self) -> list[dict[str, Any]]:
        models = list(self._load_index()["models"].values())
        models.sort(key=lambda r: int(r.get("created_at_unix", 0)), reverse=True)
        return models

    def load_model_record(self, model_id: str) -> dict[str, Any]:
        rec = self._load_index()["models"].get(model_id)
        if rec is None:
            raise FileNotFoundError(f"model not found: {model_id}")
        return dict(rec)

    def model_exists(self, model_id: str) -> bool:
        return os.path.exists(self.checkpoint_path(model_id)) and os.path.exists(self.config_path(model_id))

    # ---- checkpoints ----
    def save_checkpoint(
        self,
        model_id: str,
        cfg: ModelConfig,
        model: nn.Module,
        *,
        step: int,
        extra: dict[str, Any] | None = None,
    ) -> str:
        os.makedirs(self.model_dir(model_id), exist_ok=True)
        with open(self.config_path(model_id), "w", encoding="utf-8") as f:
            f.write(cfg.to_json())
        payload = {
            "config": cfg.to_dict(),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "step": int(step),
            "extra": dict(extra or {}),
        }
        path = self.checkpoint_path(model_id)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def load_config(self, model_id: str) -> ModelConfig:
        with open(self.config_path(model_id), "r", encoding="utf-8") as f:
            return ModelConfig.from_json(f.read())

    def load_checkpoint(self, model_id: str, device: torch.device | str = "cpu") -> tuple[ModelConfig, nn.Module, dict[str, Any]]:
        from plastic.model.lm import build_model

        if not self.model_exists(model_id):
            raise FileNotFoundError(f"checkpoint not found for model: {model_id}")
        ckpt = torch.load(self.checkpoint_path(model_id), map_location="cpu")
        cfg = ModelConfig.from_dict(ckpt["config"])
        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device).eval()
        return cfg, model, {"step": int(ckpt.get("step", 0)), "extra": dict(ckpt.get("extra", {}))}

    def model_signature(self, model_id: str) -> str:
        # A pretrained-backend model (Qwen) has no local plastic checkpoint/config to hash; its
        # signature is its backend and the content digest recorded at registration (the same digest
        # QwenBackend binds a saved session to), so a session created against one checkpoint is
        # refused if the registered checkpoint content changes.
        rec = self._load_index()["models"].get(model_id, {})
        if rec.get("backend") and rec.get("backend") != "plastic":
            return f"{rec['backend']}:{rec.get('checkpoint_digest') or rec.get('checkpoint_dir') or model_id}"
        cfg = self.load_config(model_id)
        h = hashlib.sha256()
        h.update(cfg.signature_material().encode("utf-8"))
        with open(self.checkpoint_path(model_id), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ---- eval and logs ----
    def write_eval(self, model_id: str, ev: dict[str, Any]) -> None:
        atomic_write_json(self.eval_path(model_id), ev)

    def read_eval(self, model_id: str) -> dict[str, Any] | None:
        p = self.eval_path(model_id)
        return read_json(p) if os.path.exists(p) else None

    def append_log(self, model_id: str, rec: dict[str, Any]) -> None:
        append_jsonl(self.train_log_path(model_id), rec)

    def read_log(self, model_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return read_jsonl(self.train_log_path(model_id), limit=limit)


# ---------------------------------------------------------------------- sessions
class SessionStoreMixin:
    """Session directories under ``<root>/sessions/<session_id>/``:

        meta.json            lineage, model id and signature, harness config, summary
        runner_state.pt      TransactionRunner.state_dict()
        transactions.jsonl   one record per chunk decision with every signal
        trace.jsonl          prompts/completions (text) or episodes (physics)
    """

    root: str

    @property
    def sessions_dir(self) -> str:
        return os.path.join(self.root, "sessions")

    @property
    def sessions_index(self) -> str:
        return os.path.join(self.sessions_dir, "index.json")

    def session_dir(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id)

    def session_meta_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "meta.json")

    def runner_state_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "runner_state.pt")

    def transactions_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "transactions.jsonl")

    def trace_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), "trace.jsonl")

    def canary_path(self, model_id: str) -> str:
        return os.path.join(self.model_dir(model_id), "canary.json")  # type: ignore[attr-defined]

    def ensure_sessions(self) -> None:
        os.makedirs(self.sessions_dir, exist_ok=True)
        if not os.path.exists(self.sessions_index):
            atomic_write_json(self.sessions_index, {"sessions": {}})

    def _load_sessions_index(self) -> dict[str, Any]:
        self.ensure_sessions()
        idx = read_json(self.sessions_index)
        sessions = idx.get("sessions", {}) if isinstance(idx, dict) else {}
        return {"sessions": dict(sessions) if isinstance(sessions, dict) else {}}

    def _upsert_session_index(self, session_id: str, summary: dict[str, Any]) -> None:
        idx = self._load_sessions_index()
        idx["sessions"][session_id] = summary
        atomic_write_json(self.sessions_index, idx)

    def session_exists(self, session_id: str) -> bool:
        return os.path.exists(self.session_meta_path(session_id))

    def new_session_id(self, prefix: str = "sess") -> str:
        self.ensure_sessions()
        base = f"{prefix}_{int(time.time())}"
        known = set(self._load_sessions_index()["sessions"].keys())
        sid, n = base, 1
        while self.session_exists(sid) or sid in known:
            sid = f"{base}_{n}"
            n += 1
        return sid

    def _session_summary(self, meta: dict[str, Any]) -> dict[str, Any]:
        keys = ("session_id", "model_id", "domain", "parent_session_id", "root_session_id", "created_at_unix",
                "updated_at_unix", "pos", "n_transactions", "commits", "rollbacks", "scales", "projects",
                "readonly", "budget_used", "read_only", "read_only_reason", "forked_at_pos")
        return {k: meta.get(k) for k in keys}

    def create_session(
        self,
        session_id: str,
        *,
        model_id: str,
        domain: str,
        harness_cfg: Any,
        parent_session_id: str | None = None,
        runner_state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_sessions()
        if self.session_exists(session_id):
            raise FileExistsError(f"session already exists: {session_id}")
        signature = self.model_signature(model_id)  # type: ignore[attr-defined]
        root_id = session_id
        forked_at = None
        if parent_session_id is not None:
            parent = self.load_session_meta(parent_session_id)
            if parent.get("model_signature") != signature:
                raise ValueError("parent session was created with a different model")
            root_id = str(parent.get("root_session_id", parent_session_id))
            forked_at = parent.get("pos", 0)
        now = int(time.time())

        def _committed_pos(rs: dict[str, Any] | None) -> int:
            # prefer the backend-independent cursor; fall back to a plastic state's serialized 'pos'
            if not rs:
                return 0
            if "committed_pos" in rs:
                return int(rs["committed_pos"])
            return int(rs.get("committed", {}).get("pos", rs.get("working", {}).get("pos", 0)))

        committed_pos = _committed_pos(runner_state)
        if parent_session_id is not None:
            forked_at = committed_pos  # a fork starts from the parent's committed state
        meta: dict[str, Any] = {
            "session_id": session_id,
            "model_id": model_id,
            "model_signature": signature,
            "domain": domain,
            "parent_session_id": parent_session_id,
            "root_session_id": root_id,
            "forked_at_pos": forked_at,
            "created_at_unix": now,
            "updated_at_unix": now,
            "harness": harness_cfg.to_dict() if hasattr(harness_cfg, "to_dict") else dict(harness_cfg),
            "pos": committed_pos,
            "n_transactions": 0,
            "commits": 0,
            "rollbacks": 0,
            "scales": 0,
            "projects": 0,
            "readonly": 0,
            "budget_used": 0.0 if runner_state is None else float(runner_state.get("budget_used", 0.0)),
            "read_only": False if runner_state is None else bool(runner_state.get("read_only", False)),
            "read_only_reason": None if runner_state is None else runner_state.get("read_only_reason"),
            "extra": dict(extra or {}),
        }
        os.makedirs(self.session_dir(session_id), exist_ok=True)
        atomic_write_json(self.session_meta_path(session_id), meta)
        torch.save(runner_state or {}, self.runner_state_path(session_id))
        self._upsert_session_index(session_id, self._session_summary(meta))
        return meta

    def load_session_meta(self, session_id: str) -> dict[str, Any]:
        if not self.session_exists(session_id):
            raise FileNotFoundError(f"session not found: {session_id}")
        return read_json(self.session_meta_path(session_id))

    def verify_session_model(self, session_id: str) -> None:
        meta = self.load_session_meta(session_id)
        sig = self.model_signature(str(meta["model_id"]))  # type: ignore[attr-defined]
        if sig != meta.get("model_signature"):
            raise ValueError(
                f"session {session_id} was created with model {meta['model_id']} at a different signature; refusing to load"
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = list(self._load_sessions_index()["sessions"].values())
        sessions.sort(key=lambda r: int(r.get("created_at_unix", 0) or 0), reverse=True)
        return sessions

    def save_runner_state(self, session_id: str, state: dict[str, Any], *, summary: dict[str, Any], counts: dict[str, int] | None = None) -> dict[str, Any]:
        meta = self.load_session_meta(session_id)
        meta["updated_at_unix"] = int(time.time())
        meta["pos"] = int(summary.get("pos", meta.get("pos", 0)))
        meta["n_transactions"] = int(summary.get("n_transactions", meta.get("n_transactions", 0)))
        meta["budget_used"] = float(summary.get("budget_used", 0.0))
        meta["read_only"] = bool(summary.get("read_only", False))
        meta["read_only_reason"] = summary.get("read_only_reason")
        for k, v in (counts or {}).items():
            meta[k] = int(meta.get(k, 0)) + int(v)
        tmp = self.runner_state_path(session_id) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.runner_state_path(session_id))
        atomic_write_json(self.session_meta_path(session_id), meta)
        self._upsert_session_index(session_id, self._session_summary(meta))
        return meta

    def load_runner_state(self, session_id: str) -> dict[str, Any]:
        p = self.runner_state_path(session_id)
        if not os.path.exists(p):
            return {}
        st = torch.load(p, map_location="cpu", weights_only=False)
        return st if isinstance(st, dict) else {}

    def append_transaction(self, session_id: str, record: dict[str, Any]) -> None:
        append_jsonl(self.transactions_path(session_id), record)

    def read_transactions(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return read_jsonl(self.transactions_path(session_id), limit=limit)

    def append_trace(self, session_id: str, record: dict[str, Any]) -> None:
        append_jsonl(self.trace_path(session_id), record)

    def read_trace(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return read_jsonl(self.trace_path(session_id), limit=limit)

    def fork_session(self, parent_session_id: str, child_session_id: str) -> dict[str, Any]:
        from plastic.harness.config import HarnessConfig

        from plastic.harness.transaction import fork_state_dict

        parent = self.load_session_meta(parent_session_id)
        state = self.load_runner_state(parent_session_id)
        if state:
            state = fork_state_dict(state)
        meta = self.create_session(
            child_session_id,
            model_id=str(parent["model_id"]),
            domain=str(parent["domain"]),
            harness_cfg=HarnessConfig.from_dict(parent["harness"]),
            parent_session_id=parent_session_id,
            runner_state=state,
            extra=dict(parent.get("extra", {})),
        )
        return meta

    def delete_session(self, session_id: str) -> None:
        import shutil

        if self.session_exists(session_id):
            shutil.rmtree(self.session_dir(session_id))
        idx = self._load_sessions_index()
        idx["sessions"].pop(session_id, None)
        atomic_write_json(self.sessions_index, idx)


class ArtifactStore(ArtifactStore, SessionStoreMixin):  # type: ignore[no-redef]
    """Models plus sessions."""
