"""Shared helpers for the routers: response shaping, paging, and error mapping.

Routers stay thin; everything that turns a Python object from the harness, the
store, or the red team into the JSON shape of the API contract lives here.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from fastapi import HTTPException

from plastic.harness.calibrate import Calibration
from plastic.harness.canary import CanarySuite
from plastic.store import ArtifactStore, read_json

# ---------------------------------------------------------------------- json safety


def sanitize(obj: Any) -> Any:
    """Replace non-finite floats with null, recursively.

    The harness and the red team legitimately produce NaN (a family with no
    finite provisional damage, a canary score that was never measured).
    ``json.dumps(allow_nan=False)`` inside Starlette would turn each of those
    into a 500, so every response body passes through here.
    """
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar, 0-dim tensor
        try:
            return sanitize(obj.item())
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return sanitize(obj.tolist())
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def _finite(value: Any) -> float | None:
    """One float, or null when it is infinite or NaN (an unbounded threshold)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------- models


def canary_counts(store: ArtifactStore, model_id: str) -> dict[str, int] | None:
    path = store.canary_path(model_id)
    if not os.path.exists(path):
        return None
    try:
        suite = CanarySuite.load(path)
    except Exception:  # noqa: BLE001
        return None
    return {"n_coherence": len(suite.coherence), "n_poison": len(suite.poison)}


def model_summary(store: ArtifactStore, rec: dict[str, Any]) -> dict[str, Any]:
    model_id = str(rec.get("model_id"))
    model_dir = store.model_dir(model_id)
    return {
        "model_id": model_id,
        "domain": rec.get("domain"),
        "status": rec.get("status"),
        "params": int(rec.get("params") or 0),
        "created_at_unix": int(rec.get("created_at_unix") or 0),
        "updated_at_unix": int(rec.get("updated_at_unix") or 0),
        "steps": rec.get("steps"),
        "tokens": rec.get("tokens"),
        "eval": rec.get("eval"),
        "calibrated": Calibration.exists(model_dir),
        "has_canary": os.path.exists(store.canary_path(model_id)),
        "parent_model_id": rec.get("parent_model_id"),
        "type": rec.get("type"),
        "sleep": rec.get("sleep"),
    }


def calibration_payload(cal: Calibration) -> dict[str, Any]:
    """The calibration without the raw reference windows (sizes only).

    ``thresholds`` values may be null: a signal whose quantile saturates or
    whose bound is unlimited has no finite threshold. ``achievable_fpr`` is the
    per-signal false-positive rate the calibration sample can actually support,
    which is not the requested ``target_fpr`` when the sample is small.
    """
    return {
        "n_chunks": int(cal.n_chunks),
        "thresholds": {k: _finite(v) for k, v in cal.thresholds.items()},
        "achievable_fpr": {k: _finite(v) for k, v in cal.achievable_fpr.items()},
        "canary_baseline": {k: _finite(v) for k, v in cal.canary_baseline.items()},
        "reference_sizes": {k: len(v) for k, v in cal.reference.items()},
        "target_fpr": float(cal.target_fpr),
        "created_at_unix": int(cal.created_at_unix),
        # the MODEL-compatibility signature this calibration was built against (it identifies the model,
        # not distinct calibration content -- two calibrations of the same model share it). The session
        # detail is truthful about the active artifact because it serves the calibration the runner
        # actually loaded, not because this signature distinguishes content (ASTRA-096 #2 / ASTRA-100).
        "model_signature": cal.model_signature or None,
    }


def model_config_dict(store: ArtifactStore, model_id: str, rec: dict[str, Any]) -> dict[str, Any]:
    if os.path.exists(store.config_path(model_id)):
        return store.load_config(model_id).to_dict()
    # a job that has only just started has a record but no checkpoint yet
    return dict((rec.get("train_config") or {}).get("model") or {})


def model_detail(store: ArtifactStore, model_id: str) -> dict[str, Any]:
    rec = store.load_model_record(model_id)
    model_dir = store.model_dir(model_id)
    cal = Calibration.load(model_dir) if Calibration.exists(model_dir) else None
    return {
        "record": model_summary(store, rec),
        "config": model_config_dict(store, model_id, rec),
        "eval": store.read_eval(model_id),  # the full eval, including beta_hist
        "calibration": None if cal is None else calibration_payload(cal),
        "canary": canary_counts(store, model_id),
        "log": store.read_log(model_id, limit=200),
    }


def latest_log_record(store: ArtifactStore, model_id: str) -> dict[str, Any] | None:
    log = store.read_log(model_id, limit=200)
    for rec in reversed(log):
        if "loss" in rec:
            return rec
    return log[-1] if log else None


# ---------------------------------------------------------------------- sessions


def session_summary(store: ArtifactStore, meta: dict[str, Any]) -> dict[str, Any]:
    """The listing shape, straight from the store so the two never drift."""
    return store._session_summary(meta)  # noqa: SLF001


def session_meta_payload(store: ArtifactStore, meta: dict[str, Any]) -> dict[str, Any]:
    out = session_summary(store, meta)
    out["harness"] = meta.get("harness", {})
    out["model_signature"] = meta.get("model_signature")
    return out


def lineage(store: ArtifactStore, session_id: str) -> list[str]:
    """Ancestors of ``session_id``, root first, ending with the session itself."""
    chain = [session_id]
    seen = {session_id}
    current = session_id
    while True:
        try:
            meta = store.load_session_meta(current)
        except FileNotFoundError:
            break
        parent = meta.get("parent_session_id")
        if not parent or parent in seen or not store.session_exists(str(parent)):
            break
        chain.append(str(parent))
        seen.add(str(parent))
        current = str(parent)
    chain.reverse()
    return chain


def transactions_tail(store: ArtifactStore, session_id: str, limit: int) -> list[dict[str, Any]]:
    """The last ``limit`` transactions, exactly as the runner wrote them.

    A record is ``{index, t_unix, pos_start, pos_end, decision, requested,
    signals, accepted, read_only, read_only_reason, seconds}``. The three that
    matter for reading a chunk are distinct and all pass through untouched:

    - ``requested``: what the policy asked for, before the harness applied it.
    - ``decision``: what the harness applied (kind, reasons, scale).
    - ``signals``: the PROPOSED update, measured on the provisional state before
      the decision, so a rollback still reports the delta it would have made.
    - ``accepted``: the committed outcome, measured after the decision
      (``delta_norm``, ``budget_charge``, ``budget_used``, ``budget_remaining``,
      and the canary values ``canary_coherence_after``, ``canary_poison_after``,
      ``canary_delta_coherence``, ``canary_delta_poison``, which are null when the
      model has no canary suite). A rollback's accepted ``delta_norm`` is zero.

    The API never computes these. They come from the runner.
    """
    return store.read_transactions(session_id, limit=limit)


def transactions_page(store: ArtifactStore, session_id: str, *, limit: int, offset: int) -> dict[str, Any]:
    """A window into the transaction log from the start, with the full count.

    Records keep every field the runner wrote; see ``transactions_tail``.
    """
    items = store.read_transactions(session_id)
    return {"total": len(items), "items": items[offset : offset + limit]}


def _finite_or_none(x: Any) -> float | None:
    """A real finite float, or None for a missing/non-numeric/nonfinite value — so a failed or absent
    measurement renders as unavailable, never as a measured zero (ASTRA-094)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _native_state_payload(session: Any) -> dict[str, Any]:
    """State view for a pretrained Protocol backend (Qwen): honest per-memory-unit recurrent norms
    and drift-from-anchor over the backend's own memory units, with a ``kind`` discriminator so the
    client renders the GDN path distinctly. No fabricated per-head S / singular values / h tensors —
    those are the toy Plastic shape and do not exist here. A measurement that FAILS or is missing is
    null (unavailable), never a measured zero; a genuine zero is preserved."""
    runner = session.runner
    backend = runner.backend
    norms = backend.state_norms(runner.committed)
    per = norms.get("recurrent_norm")
    per = list(per) if isinstance(per, list) else []
    # drift is a separate measurement that can raise or come back short; an unmatched or failed unit
    # is UNKNOWN (null), not zero drift
    drift: list[Any] | None
    try:
        drift = [t.float().norm() for t in backend.state_delta(runner.committed, runner.anchor)]
    except Exception:  # noqa: BLE001 - a degenerate/failed delta reports unknown, not a route error or zero
        drift = None
    units = [
        {
            "index": i,
            "recurrent_norm": _finite_or_none(per[i]),
            "drift_from_anchor": _finite_or_none(drift[i]) if (isinstance(drift, list) and i < len(drift)) else None,
        }
        for i in range(len(per))
    ]
    return {
        "kind": "recurrent",
        "backend": getattr(session, "backend_kind", "qwen"),
        "pos": int(runner.pos),
        "units": units,
        "recurrent_norm_total": _finite_or_none(norms.get("recurrent_norm_total")),
    }


def state_payload(session: Any) -> dict[str, Any]:
    """Per-layer view of the committed state for the weights panel.

    The toy Plastic state is per-layer ``S``/``h``; a pretrained backend (Qwen) has no such shape, so
    it gets a distinct native payload rather than a 500 or fabricated tensors."""
    committed = session.runner.committed
    anchor = session.runner.anchor
    if not hasattr(committed, "layers"):
        return _native_state_payload(session)
    layers: list[dict[str, Any]] = []
    for i, layer in enumerate(committed.layers):
        s = layer.S[0].detach().float().cpu()  # (H, d_h, d_h)
        try:
            sv = torch.linalg.svdvals(s)
        except Exception:  # noqa: BLE001 - a degenerate state should not fail the route
            sv = torch.zeros(s.shape[0], s.shape[-1])
        drift = 0.0
        if i < len(anchor.layers):
            drift = float((layer.S - anchor.layers[i].S).detach().float().norm())
        layers.append(
            {
                "s_norm_per_head": [float(s[h].norm()) for h in range(s.shape[0])],
                "h_norm": float(layer.h[0].detach().float().norm()),
                "singular_values": [[float(x) for x in sv[h][:8]] for h in range(sv.shape[0])],
                "drift_from_anchor": drift,
            }
        )
    return {"kind": "plastic", "layers": layers, "pos": int(session.runner.pos)}


# ---------------------------------------------------------------------- red team


def redteam_root(store: ArtifactStore) -> str:
    return os.path.join(store.root, "redteam")


def redteam_created_at(run_id: str, summary_path: str) -> int:
    tail = run_id.rsplit("_", 1)[-1]  # run ids are rt_<model_id>_<unix>, and model ids contain underscores
    if tail.isdigit():
        return int(tail)
    try:
        return int(os.path.getmtime(summary_path))
    except OSError:
        return 0


def redteam_summary(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    path = os.path.join(redteam_root(store), run_id, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"red team run not found: {run_id}")
    summary = read_json(path)
    summary.setdefault("run_id", run_id)
    summary["created_at_unix"] = redteam_created_at(run_id, path)
    return summary


def redteam_results(store: ArtifactStore, run_id: str) -> list[dict[str, Any]]:
    path = os.path.join(redteam_root(store), run_id, "results.jsonl")
    if not os.path.exists(path):
        return []
    from plastic.store import read_jsonl

    return read_jsonl(path)


def list_redteam(store: ArtifactStore) -> list[dict[str, Any]]:
    base = redteam_root(store)
    if not os.path.isdir(base):
        return []
    out: list[dict[str, Any]] = []
    for name in os.listdir(base):
        if not os.path.isdir(os.path.join(base, name)):
            continue
        try:
            out.append(redteam_summary(store, name))
        except Exception:  # noqa: BLE001 - a half-written run must not break the listing
            continue
    out.sort(key=lambda r: int(r.get("created_at_unix", 0)), reverse=True)
    return out


# ---------------------------------------------------------------------- data


def data_sets(store: ArtifactStore) -> list[dict[str, Any]]:
    from plastic.data.text import load_corpus_meta

    base = os.path.join(store.root, "data")
    if not os.path.isdir(base):
        return []
    out: list[dict[str, Any]] = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        try:
            meta = load_corpus_meta(d)
        except Exception:  # noqa: BLE001 - an unreadable corpus is skipped, not fatal
            continue
        out.append(
            {
                "name": name,
                "dir": d,
                "corpus": meta.corpus,
                "vocab_size": int(meta.vocab_size),
                "splits": dict(meta.splits),
            }
        )
    return out


# ---------------------------------------------------------------------- errors


def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=409, detail=detail)


def require_model(store: ArtifactStore, model_id: str) -> dict[str, Any]:
    try:
        return store.load_model_record(model_id)
    except FileNotFoundError as e:
        raise not_found(str(e)) from e


def require_session_meta(store: ArtifactStore, session_id: str) -> dict[str, Any]:
    try:
        return store.load_session_meta(session_id)
    except FileNotFoundError as e:
        raise not_found(str(e)) from e
