"""Shared model/batching code for exp 4: PlasticCore LM (the repo's PlasticLM) driven on group-action lessons.

Protocols for a query after the lessons (every scored query is the FIRST query after the lessons; queries never
see each other):
  fw_reset    lessons read with fast-weight writes on; then the activation state (SSM h, conv buffers) is zeroed and
              ONLY the fast weights S are carried into the query. Anything the query gets from the lessons must be in S.
  continuous  lessons and query in one stream (S + activation state carried): the in-context arm.
  no_adapt    lessons read with freeze=True (beta=0: S stays at W0 = 0), activations zeroed: no adaptation. Equals
              asking the query with no lessons at all.
  ssm_only    lessons read with freeze=True, activation state carried: in-context through the SSM/conv path only.
  null_*      the same protocols on content-null lessons (element names deranged), scored against the TRUE answer.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import (EQ, GROUPS, PAD, VOCAB_PADDED, World, lesson_tokens, make_query, make_world, query_tokens,  # noqa: E402
                  relabel)

from plastic.config import ModelConfig  # noqa: E402
from plastic.model.lm import PlasticLM  # noqa: E402
from plastic.model.state import LayerState, SessionState  # noqa: E402

SIZE_CLASSES = {5: ["Z5"], 6: ["Z6", "S3"], 7: ["Z7"], 8: ["Z8", "D4"]}


def model_config(d_model: int = 128, n_layers: int = 4, n_heads: int = 4) -> ModelConfig:
    return ModelConfig(domain="text", d_model=d_model, n_heads=n_heads, n_layers=n_layers, chunk=16, vocab_size=VOCAB_PADDED)


def build(cfg: ModelConfig) -> PlasticLM:
    return PlasticLM(cfg)


def expand_state(st: SessionState, k: int, reset_activations: bool) -> SessionState:
    """Repeat each episode's post-lesson state k times along the batch (one row per query). With
    ``reset_activations`` only S is kept; h and the conv buffers are zeroed (the chunk-rule fields are unused)."""
    layers = []
    for L in st.layers:
        rep = lambda t: t.repeat_interleave(k, dim=0)  # noqa: E731
        h = rep(L.h)
        cs = None if L.conv_ssm is None else rep(L.conv_ssm)
        cm = None if L.conv_mem is None else rep(L.conv_mem)
        if reset_activations:
            h = torch.zeros_like(h)
            cs = None if cs is None else torch.zeros_like(cs)
            cm = None if cm is None else torch.zeros_like(cm)
        layers.append(LayerState(h=h, S=rep(L.S), M=None, conv_ssm=cs, conv_mem=cm, chunk=None))
    return SessionState(layers=layers, pos=st.pos)


def sample_batch(rng: random.Random, batch: int, k: int, depths: list[int], null: bool = False, layout: str = "old"):
    """One size class per batch so lesson lengths match. Returns worlds, lesson tensor (B, L), query tensor (B*k, Lq),
    EQ positions (B*k,), answer tokens (B*k,), query records, and (if null) the null worlds/lessons."""
    n = rng.choice(sorted(SIZE_CLASSES))
    worlds, lessons, qrows, qrecs = [], [], [], []
    null_worlds, null_lessons = [], []
    for _ in range(batch):
        w = make_world(rng, rng.choice(SIZE_CLASSES[n]))
        worlds.append(w)
        lessons.append(lesson_tokens(w, rng))
        if null:
            nw = relabel(w, rng)
            null_worlds.append(nw)
            null_lessons.append(lesson_tokens(nw, rng))
        for _ in range(k):
            q = make_query(w, rng, rng.choice(depths))
            qrows.append(query_tokens(w, q, layout))
            qrecs.append((len(worlds) - 1, q))
    return worlds, lessons, qrows, qrecs, null_worlds, null_lessons


def pad_queries(qrows: list[list[int]], answers: list[int] | None = None):
    Lq = max(len(r) for r in qrows) + 1
    x = torch.full((len(qrows), Lq), PAD, dtype=torch.long)
    pos = torch.empty(len(qrows), dtype=torch.long)
    for i, r in enumerate(qrows):
        x[i, :len(r)] = torch.tensor(r)
        pos[i] = len(r) - 1  # the last prompt token (EQ in the old layout, last generator in the aligned layout) predicts the answer
        if answers is not None:
            x[i, len(r)] = answers[i]
    return x, pos


def run_protocol(model: PlasticLM, lessons: torch.Tensor, qx: torch.Tensor, qpos: torch.Tensor, k: int, protocol: str) -> torch.Tensor:
    """Logits at each query's EQ position (B*k, V) under a protocol (see module docstring)."""
    B = lessons.shape[0]
    st0 = model.init_state(B, lessons.device)
    if protocol in ("fw_reset", "continuous"):
        _, st, _ = model(lessons, st0, freeze=False)
    elif protocol in ("no_adapt", "ssm_only"):
        _, st, _ = model(lessons, st0, freeze=True)
    else:
        raise ValueError(protocol)
    stq = expand_state(st, k, reset_activations=protocol in ("fw_reset", "no_adapt"))
    logits, _, _ = model(qx, stq, freeze=False)
    return logits[torch.arange(qx.shape[0]), qpos]
