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
