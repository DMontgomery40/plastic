"""The Backend interface the transaction harness drives.

Enumerated from ``TransactionRunner``'s actual calls on the model and state, so the runner's
backend-agnostic control flow (commit / rollback / scale / project / read-only, calibration,
canaries) can run unchanged over either the native ``plastic`` model or a pretrained model
(Qwen3.5) wrapped as a backend. See docs/superpowers/specs/2026-09-22-pretrained-backend-integration.md.

This module defines only the contract; ``PlasticBackend`` (wrapping today's model + ``SessionState``)
and ``QwenBackend`` implement it. It is a ``Protocol`` — structural, so neither backend needs to
import it, and it stays import-light (no torch/transformers at module load beyond typing).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# A backend's state is opaque to the harness — it only ever clones it, measures the change between
# two of them, checks finiteness, and persists it. Kept as ``Any`` so a backend can use its native
# type (plastic's SessionState, or a QwenState wrapping a transformers cache).
State = Any
Tensor = Any  # torch.Tensor at runtime; not imported here to keep this module torch-free


@runtime_checkable
class Backend(Protocol):
    """What the harness needs from a model + state. Every method is per-session and side-effect-free
    except where noted (``forward`` advances the passed state's working copy)."""

    # ---- identity / signals a session records, so the harness can report them honestly ----
    def signal_names(self) -> tuple[str, ...]:
        """The STAT_SIGNALS this backend can actually produce. A backend that cannot compute a
        signal (e.g. Qwen has no surprise/write-norm without kernel instrumentation) omits it; the
        harness carries the rest as ``None`` end-to-end rather than substituting zero/NaN."""
        ...

    def writes_for_source(self, source: str) -> bool:
        """Whether a token from ``source`` (``user`` / ``model``) writes to memory under this
        backend. Plastic: generation (``model``) is frozen read-only unless learn_from_generation.
        Qwen: generation writes its recurrent state (it is the model's language context) and must be
        accounted, never counted as a free observation. Explicit read-only / spent budget / requested
        freeze still take precedence over this — ``source=model`` cannot escape those controls."""
        ...

    # ---- state lifecycle ----
    def init_state(self) -> State:
        """A fresh session state that exposes stable, correctly-shaped memory units at zero — so
        ``score_suite`` and ``canary_gradient`` are defined on the FIRST chunk, before any update."""
        ...

    def position(self, state: State) -> int:
        """The cursor (number of absorbed tokens/steps) — the runner's ``pos``. Plastic exposes it as
        ``state.pos``, Qwen as ``state.position``; the abstraction names it here so the runner never
        reads a backend-specific field."""
        ...

    def clone(self, state: State) -> State:
        ...

    def state_delta(self, a: State, b: State) -> list[Tensor]:
        """Per-memory-unit change ``a − b`` (plastic: per-layer S; Qwen: the recurrent tensors), for
        the update-norm signals and projection."""
        ...

    def is_finite(self, state: State) -> bool:
        ...

    def state_dict(self, state: State) -> dict[str, Any]:
        """Full serialization — all memory units, KV/conv, cursor/positions and init flags, plus
        backend/checkpoint/tokenizer identity for a runtime-compat check on load."""
        ...

    def load_state_dict(self, data: dict[str, Any]) -> State:
        ...

    # ---- forward ----
    def forward(self, chunk: list[int], state: State, *, freeze: bool, beta_scale: float) -> tuple[Tensor, State, list[Any]]:
        """Advance ``chunk`` through the model, returning (logits, new_state, per-token signals).
        ``freeze`` is a genuine no-write (no memory change; activation/position advance).
        ``beta_scale`` scales the write only (leaving decay); ``beta_scale=0`` writes nothing while
        decay continues — distinct from ``freeze``."""
        ...

    # ---- canaries (both frozen: measured at the committed state, never perturbed by the probe) ----
    def score_suite(self, state: State, suite: Any) -> dict[str, float]:
        ...

    def canary_gradient(self, state: State, suite: Any) -> list[Tensor]:
        """∂(coherence loss)/∂(memory unit) at ``state`` — the projection direction. Frozen."""
        ...

    def apply_projected(self, working: State, committed: State, projected: list[Tensor]) -> None:
        """Write a corrected per-unit delta back onto ``working``'s memory units (for the project
        decision). Optional: a backend may declare it unsupported and the policy uses a tested
        conservative fallback rather than calling a missing method."""
        ...
