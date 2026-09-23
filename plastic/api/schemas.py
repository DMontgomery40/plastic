"""Request bodies. Responses are plain dicts shaped by ``plastic/api/service.py``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CalibrateRequest(BaseModel):
    # toy (plastic) models: a corpus calibration
    data_dir: str | None = None
    chunks: int = Field(default=128, ge=1)
    fisher_chunks: int = Field(default=16, ge=0)
    fpr: float = Field(default=0.01, gt=0.0, lt=1.0)
    # pretrained chat backends (qwen, ttt): a real-chat calibration; prompts default to the bundled benign set
    prompts: list[str] | None = Field(default=None, min_length=1, max_length=512)
    cusum_prompts: list[str] | None = Field(default=None, min_length=1, max_length=512)
    max_new_tokens: int = Field(default=32, ge=1, le=256)


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


class ForkRequest(BaseModel):
    child_session_id: str | None = None




class RecallProbeIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=500)
    paraphrase: str | None = Field(default=None, max_length=2000)


class SleepRequest(BaseModel):
    """Options for one sleep run on a TTT chat model (see docs/research/2026-09-23-sleep-consolidation.md)."""

    method: Literal["replay", "distill", "anchor", "dream"] = "replay"
    target: Literal["w0", "all"] = "w0"
    steps: int = Field(default=40, ge=1, le=2000)
    lr: float = Field(default=1e-4, gt=0.0, le=1e-2)
    seq_len: int = Field(default=512, ge=32, le=4096)
    batch_size: int = Field(default=2, ge=1, le=16)
    replay_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    replay_rows: int = Field(default=64, ge=0, le=2000)
    heldout_rows: int = Field(default=24, ge=0, le=500)
    anchor_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    distill_temperature: float = Field(default=1.0, gt=0.0, le=10.0)
    tolerance_nll: float = Field(default=0.05, ge=0.0, le=5.0)
    seed: int = Field(default=0, ge=0)
    sessions: list[str] | None = Field(default=None, min_length=1, max_length=64)
    probes: list[RecallProbeIn] | None = Field(default=None, min_length=1, max_length=64)
