"""The statistics behind the surprise-vs-entropy measurement (build-order step 4) on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.surprise_vs_entropy import (
    auc,
    bootstrap_increment,
    compare_predictors,
    logistic_fit,
    loop_separation,
    rank,
    repeated_ngram_mask,
    spearman,
)


def test_rank_auc_and_spearman_on_known_cases():
    assert rank(np.array([10.0, 30.0, 20.0])).tolist() == [1.0, 3.0, 2.0]
    assert rank(np.array([1.0, 1.0, 2.0])).tolist() == [1.5, 1.5, 3.0]          # ties averaged
    assert auc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == 0.75      # one of four pairs inverted
    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.5, 0.5], [True, False]) == 0.5                                # a tie counts half
    assert auc([0.1, 0.2], [True, True]) is None                                # one class only
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None                               # constant input


def test_logistic_fit_recovers_the_sign_and_direction_of_a_planted_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    noise = rng.normal(size=4000)
    y = (2.0 * x + noise > 0).astype(float)
    w, ll, p = logistic_fit(y, [x])
    assert w[1] > 1.0 and ll > np.log(0.5)                                     # better than the coin flip
    assert auc(p, y.astype(bool)) > 0.9
    # a pure-noise second column adds no likelihood to speak of and gets a small coefficient
    w2, ll2, _ = logistic_fit(y, [x, noise * 0 + rng.normal(size=4000)])
    assert abs(w2[2]) < 0.1 and ll2 - ll < 0.002


def test_repeated_ngram_mask_marks_only_the_loop_and_needs_a_full_window():
    assert repeated_ngram_mask([1, 2, 3, 4, 9, 1, 2, 3, 4], 4) == [False] * 5 + [True] * 4
    assert repeated_ngram_mask([1, 2, 3], 4) == [False, False, False]
    assert repeated_ngram_mask([7, 7, 7, 7, 7], 4) == [False, True, True, True, True]      # the window 1..4 repeats window 0..3
    assert not any(repeated_ngram_mask(list(range(50)), 4))


def _rows(rng, n, *, informative_surprise: bool, conv_size: int = 50):
    """Synthetic tokens: entropy drives error; surprise either adds independent information or is pure noise."""
    H = rng.gamma(2.0, 1.0, size=n)
    S_signal = rng.normal(size=n)
    S = S_signal if informative_surprise else rng.normal(size=n)
    logit = -1.5 + 0.8 * H + (1.2 * S_signal if informative_surprise else 0.0)
    miss = rng.random(n) < 1 / (1 + np.exp(-logit))
    nll = 0.5 * H + (0.8 * S_signal if informative_surprise else 0.0) + rng.normal(scale=0.3, size=n)
    return [{"conv": i // conv_size, "entropy": float(H[i]), "surprise": float(S[i]), "nll_next": float(nll[i]), "top1_miss": bool(miss[i])} for i in range(n)]


def test_compare_predictors_and_bootstrap_separate_an_informative_surprise_from_noise():
    rng = np.random.default_rng(1)
    informative = compare_predictors(_rows(rng, 3000, informative_surprise=True))
    noise = compare_predictors(_rows(rng, 3000, informative_surprise=False))
    assert informative["increment"]["r2_nll"] > 0.3 and informative["increment"]["auc_top1_miss"] > 0.05
    assert abs(noise["increment"]["r2_nll"]) < 0.01 and abs(noise["increment"]["auc_top1_miss"]) < 0.02
    assert abs(noise["logistic_top1_miss"]["surprise_coefficient_given_entropy"]) < 0.15
    boot = bootstrap_increment(_rows(rng, 2000, informative_surprise=False), reps=30, seed=0)
    # in-sample R2 cannot fall when a column is added, so the increment of a noise column is tiny and non-negative
    assert boot["conversations"] == 40 and 0.0 <= boot["r2_nll"]["lo"] and boot["r2_nll"]["hi"] < 0.01
    assert compare_predictors([])["reason"] == "too few tokens"


def test_loop_separation_uses_only_replies_with_both_kinds_of_token():
    rows = [{"reply_has_both": True, "in_loop": True, "entropy": 0.5, "surprise": 2.0},
            {"reply_has_both": True, "in_loop": False, "entropy": 3.0, "surprise": 1.0},
            {"reply_has_both": False, "in_loop": False, "entropy": 9.0, "surprise": 9.0}]
    sep = loop_separation(rows)
    assert sep["n"] == 2 and sep["auc_in_loop"] == {"entropy": 0.0, "surprise": 1.0}
    assert loop_separation([rows[2]])["n"] == 0


def test_parser_help_lists_the_measurement_controls():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "scripts.experiments.surprise_vs_entropy", "--help"], capture_output=True, text=True, check=True).stdout
    assert "--replay-revision" in out and "--temperatures" in out and "--skip-generation" in out and "--bootstrap" in out


@pytest.mark.parametrize("device,skip_generation,replay_revision", [
    ("cpu", True, ""),
    ("mps", False, "pinned-test-revision"),
])
def test_cli_output_preserves_backend_identity_and_run_provenance(
    tmp_path, monkeypatch, device, skip_generation, replay_revision,
):
    """Exercise saved results without model, accelerator or dataset access."""
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from scripts.experiments import surprise_vs_entropy as experiment

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sft_meta.json").write_text(json.dumps({"device": device}))
    backend = SimpleNamespace(checkpoint_digest="a" * 64)
    loaded = []

    def load(path, *, device):
        loaded.append((path, device))
        return backend

    replay_calls = []
    conversations = [[{"role": "user", "content": "test"}]]

    def load_replay(subset, split, rows, seed, log, *, revision):
        replay_calls.append((subset, split, rows, seed, revision))
        return conversations

    modules = {
        "plastic.backends.ttt_lm.backend": {"TTTBackend": SimpleNamespace(load=load)},
        "plastic.sleep.ttt": {"load_replay_conversations": load_replay},
        "scripts.train.eval_ttt_chat": {"BOUNDARY_PROMPTS": [], "NEUTRAL_PROMPTS": ["test"]},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    token_rows = [{"conv": 0, "assistant": False}, {"conv": 0, "assistant": True}]
    monkeypatch.setattr(experiment, "teacher_forced_pass", lambda *a, **kw: token_rows)
    monkeypatch.setattr(experiment, "generation_pass", lambda *a, **kw: [])
    monkeypatch.setattr(experiment, "_git_head", lambda: "test-source+dirty")
    output = tmp_path / "output"
    argv = ["surprise_vs_entropy", "--checkpoint", str(checkpoint), "--out", str(output),
            "--device", device, "--rows", "3", "--seed", "7", "--max-len", "48", "--chunk", "16",
            "--replay-revision", replay_revision, "--bootstrap", "2", "--temperatures", "0.3"]
    if skip_generation:
        argv.append("--skip-generation")
    monkeypatch.setattr(sys, "argv", argv)

    experiment.main()

    result = json.loads((output / "surprise_vs_entropy.json").read_text())
    expected = {"checkpoint": str(checkpoint.resolve()), "checkpoint_digest": backend.checkpoint_digest,
                "device": device, "code_commit": "test-source+dirty", "replay_revision": replay_revision or None,
                "rows": 1, "max_len": 48, "chunk": 16, "seed": 7}
    assert {key: result[key] for key in expected} == expected
    assert isinstance(result["started_at_unix"], int) and result["seconds"] >= 0
    assert loaded == [(str(checkpoint), device)]
    assert replay_calls == [("everyday-conversations", "test", 3, 8, replay_revision or None)]
    assert ("generation" in result) is not skip_generation
    assert [json.loads(line) for line in (output / "teacher_forced_tokens.jsonl").read_text().splitlines()] == token_rows
