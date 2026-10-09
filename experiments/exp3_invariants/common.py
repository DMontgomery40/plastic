"""Shared loading for exp3(b): the exp2 model, its calibration, and the two canary suites."""
from __future__ import annotations

import os

import torch

from plastic.harness.calibrate import Calibration
from plastic.harness.canary import CanarySuite
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_ID = "lm_small_cpu"
DATA = os.path.join(ROOT, "artifacts", "data", "wikitext")
RES = os.path.join(ROOT, "experiments", "results", "exp3")


def load(suite_kind: str = "shipped"):
    store = ArtifactStore(os.path.join(ROOT, "artifacts"))
    cfg, model, _ = store.load_checkpoint(MODEL_ID, "cpu")
    model.eval()
    cal = Calibration.load(store.model_dir(MODEL_ID))
    if suite_kind == "shipped":
        # what `plastic calibrate` built for this text model: 3 held-out windows (coherence),
        # 3 random-token + 1 single-token-run probes (poison), 128 tokens each
        suite = CanarySuite.load(store.canary_path(MODEL_ID))
    elif suite_kind == "statements":
        # the README's 12 public true / 12 public false statements + 3 repetition probes
        # (CanarySuite.default_chat), encoded with this model's own BPE tokenizer
        tok = Tokenizer.load(store.tokenizer_path(MODEL_ID))
        suite = CanarySuite.default_chat(lambda t: tok.encode(t))
    else:
        raise ValueError(suite_kind)
    return store, cfg, model, cal, suite
