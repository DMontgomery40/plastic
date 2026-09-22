import numpy as np
import torch

from plastic.data.mqar import mqar_accuracy, mqar_batch
from plastic.data.physics import PhysicsEnv, physics_batch
from plastic.data.text import TokenWindows, encode_documents_to_bin, split_wikitext_documents
from plastic.tokenizer.bpe import Tokenizer

DOCS = ["alpha beta gamma delta " * 30, "one two three four five " * 30, "the end of it " * 30]


def _tok() -> Tokenizer:
    return Tokenizer.train(DOCS, vocab_size=300)


def test_encode_documents_writes_uint16_with_eos(tmp_path):
    tok = _tok()
    path = str(tmp_path / "train.bin")
    n = encode_documents_to_bin(tok, DOCS, path)
    arr = np.fromfile(path, dtype="<u2")
    assert len(arr) == n
    assert int((arr == tok.eos_id).sum()) == len(DOCS)
    assert arr[-1] == tok.eos_id


def test_encode_documents_honors_max_tokens(tmp_path):
    tok = _tok()
    path = str(tmp_path / "small.bin")
    n = encode_documents_to_bin(tok, DOCS, path, max_tokens=50)
    assert n == 50
    assert len(np.fromfile(path, dtype="<u2")) == n


def test_token_windows_sample_and_sequential(tmp_path):
    tok = _tok()
    path = str(tmp_path / "train.bin")
    n = encode_documents_to_bin(tok, DOCS, path)
    win = TokenWindows(path, seq_len=16)
    g = torch.Generator().manual_seed(0)
    x = win.sample(4, g)
    assert x.shape == (4, 17) and x.dtype == torch.long
    assert int(x.max()) < tok.vocab_size
    total = 0
    last = None
    for batch in win.sequential(batch=2):
        assert batch.shape[1] == 17
        total += batch.shape[0] * 16
        last = batch
    assert total <= n and total >= n - 2 * 16 - 1
    assert last is not None


def test_split_wikitext_documents():
    lines = [
        " \n",
        " = Article One = \n",
        " Some text about one . \n",
        " = = Section = = \n",
        " More text . \n",
        " = Article Two = \n",
        " Text about two . \n",
    ]
    docs = list(split_wikitext_documents(lines))
    assert len(docs) == 2
    assert docs[0].startswith("= Article One =") and "Section" in docs[0]
    assert docs[1].startswith("= Article Two =")


def test_mqar_batch_structure():
    g = torch.Generator().manual_seed(0)
    toks, mask = mqar_batch(3, n_pairs=4, seq_len=32, vocab_size=64, key_range=(3, 30), value_range=(30, 64), rng=g)
    assert toks.shape == (3, 32) and mask.shape == (3, 32) and mask.dtype == torch.bool
    for b in range(3):
        keys = toks[b, 1:9:2].tolist()
        vals = toks[b, 2:9:2].tolist()
        assert len(set(keys)) == 4 and all(3 <= k < 30 for k in keys) and all(30 <= v < 64 for v in vals)
        for t in torch.nonzero(mask[b]).flatten().tolist():
            k, v = int(toks[b, t]), int(toks[b, t + 1])
            assert vals[keys.index(k)] == v
    assert int(mask.sum()) == 3 * 4


def test_mqar_oracle_accuracy():
    g = torch.Generator().manual_seed(1)
    toks, mask = mqar_batch(2, n_pairs=3, seq_len=20, vocab_size=40, key_range=(3, 20), value_range=(20, 40), rng=g)
    logits = torch.full((2, 20, 40), -10.0)
    logits.scatter_(2, toks[:, 1:].unsqueeze(-1).expand(-1, -1, 1).clone().roll(0, 1)[:, :, :1].new_zeros(2, 19, 1) + toks[:, 1:].unsqueeze(-1), 10.0) if False else None
    # oracle: at every position predict the next token
    for b in range(2):
        for t in range(19):
            logits[b, t, int(toks[b, t + 1])] = 10.0
    assert mqar_accuracy(logits, toks, mask) == 1.0


def test_physics_env_matches_closed_form():
    env = PhysicsEnv(mu=0.1)
    obs = env.reset()
    assert obs.shape == (4,)
    vel = torch.zeros(2)
    pos = torch.zeros(2)
    for _ in range(5):
        a = torch.randn(2)
        obs = env.step(a)
        vel = (1 - 0.1) * vel + a
        pos = pos + vel
        assert torch.allclose(obs, torch.cat([pos, vel]), atol=1e-6)


def test_physics_batch_shapes_resets_and_targets():
    g = torch.Generator().manual_seed(0)
    batch = physics_batch(3, seq_len=64, episodes_per_seq=4, mu_range=(0.02, 0.25), nonlinear=False, action_std=0.5, rng=g)
    assert batch.inputs.shape == (3, 64, 7) and batch.target_delta.shape == (3, 64, 4) and batch.mu.shape == (3, 4)
    assert int(batch.inputs[0, :, 6].sum()) == 4
    assert batch.inputs[0, 0, 6] == 1 and batch.inputs[0, 16, 6] == 1
    obs = batch.inputs[:, :, :4]
    inside = batch.inputs[:, 1:, 6] == 0
    assert torch.allclose((obs[:, 1:] - obs[:, :-1])[inside], batch.target_delta[:, :-1][inside], atol=1e-5)
    assert (obs[:, 16] == 0).all()
