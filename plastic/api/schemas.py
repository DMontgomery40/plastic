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


