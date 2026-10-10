"""Build a tiny, disposable artifact store so the dashboard can be run and screenshotted on CPU.

    python scripts/ui_demo_store.py /tmp/plastic-ui-store

The model is a few-step toy (d_model 32), not a useful language model: it exists so the real API,
harness and dashboard produce real transactions. Nothing here is a research result.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

torch.set_num_threads(2)

from plastic.config import ModelConfig  # noqa: E402
from plastic.data.text import encode_documents_to_bin  # noqa: E402
from plastic.harness import calibrate as calibrate_mod  # noqa: E402
from plastic.harness.canary import CanarySuite  # noqa: E402
from plastic.store import ArtifactStore  # noqa: E402
from plastic.tokenizer.bpe import Tokenizer  # noqa: E402
from plastic.train.loop import TrainConfig, train  # noqa: E402

DOCS = [
    ("The river runs past the old mill and into the valley where the farmers grow wheat and barley. " * 40),
    ("Maria keeps a small garden behind her house with tomatoes, basil and a single lemon tree. " * 40),
    ("The train to the coast leaves at nine and arrives before noon if the weather holds. " * 40),
    ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi " * 40),
]


def main(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    data_dir = os.path.join(root, "data", "tiny")
    os.makedirs(data_dir, exist_ok=True)
    tok = Tokenizer.train(DOCS, vocab_size=300)
    tok.save(os.path.join(data_dir, "tokenizer.json"))
    n_train = encode_documents_to_bin(tok, DOCS, os.path.join(data_dir, "train.bin"))
    n_val = encode_documents_to_bin(tok, DOCS, os.path.join(data_dir, "validation.bin"))
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"corpus": "tiny", "vocab_size": tok.vocab_size, "splits": {"train": n_train, "validation": n_val, "test": 0},
                   "created_at_unix": int(time.time()), "tokenizer_lines": len(DOCS)}, f)
    model_id = train(
        TrainConfig(domain="text", model=ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=tok.vocab_size),
                    artifacts_root=root, model_id="lm_tiny", data_dir=data_dir, steps=60, batch_size=4, seq_len=48,
                    warmup_steps=5, eval_every=0, save_every=0, eval_batches=2, log_every=20, device="cpu", mqar_frac=0.0),
        log=print,
    )
    store = ArtifactStore(root)
    CanarySuite.default_text(os.path.join(data_dir, "validation.bin"), vocab_size=tok.vocab_size, n_probe=4, probe_len=16).save(store.canary_path(model_id))
    calibrate_mod.calibrate_model(store, model_id, data_dir=data_dir, n_chunks=64, fisher_chunks=8, target_fpr=0.05, device="cpu", seed=0)
    print("store ready:", root, "model:", model_id)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/plastic-ui-store")
