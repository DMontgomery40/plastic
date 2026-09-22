"""Request bodies. Responses are plain dicts shaped by ``plastic/api/service.py``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CalibrateRequest(BaseModel):
    data_dir: str | None = None
    chunks: int = Field(default=128, ge=1)
    fisher_chunks: int = Field(default=16, ge=0)
    fpr: float = Field(default=0.01, gt=0.0, lt=1.0)


class CreateSessionRequest(BaseModel):
    model_id: str
    session_id: str | None = None
    harness: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=128, ge=0)
    temperature: float = Field(default=0.9, gt=0.0)
    top_k: int = Field(default=50, ge=0)
    seed: int | None = None


class PhysicsRequest(BaseModel):
    steps: int = Field(default=256, ge=1)
    mu: float = 0.12
    seed: int = 0
    nonlinear: bool = False


class ForkRequest(BaseModel):
    child_session_id: str | None = None


class TrainRequest(BaseModel):
    domain: Literal["text", "physics"]
    data_dir: str | None = None
    model_id: str | None = None
    steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    seq_len: int = Field(ge=8)
    d_model: int | None = None
    layers: int | None = None
    heads: int | None = None
    chunk: int | None = None
    adversarial: bool = False
    device: str | None = None
    eval_every: int | None = None
    save_every: int | None = None


class RedteamRequest(BaseModel):
    model_id: str
    data_dir: str | None = None
    prefixes: int = Field(default=4, ge=1)
    prefix_len: int = Field(default=128, ge=1)
    suffix_len: int = Field(default=64, ge=1)
    steps: int = Field(default=30, ge=0)
    families: list[str] | None = None
    record: bool = False


class SleepRequest(BaseModel):
    model_id: str
    sessions: list[str] | None = None
    core_data_dir: str | None = None
    steps: int = Field(default=200, ge=1)
    lr: float = Field(default=1e-4, gt=0.0)
    core_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    seq_len: int = Field(default=256, ge=8)
    batch_size: int = Field(default=8, ge=1)
    tolerance: dict[str, float] | None = None
