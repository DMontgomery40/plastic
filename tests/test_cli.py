import pytest

from plastic.cli import build_parser, main
from plastic.store import ArtifactStore


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for cmd in ("data", "train", "models", "calibrate", "chat"):
        assert cmd in out


def test_calibrate_pretrained_chat_model_requires_prompts(tmp_path, capsys):
    root = str(tmp_path / "artifacts")
    store = ArtifactStore(root)
    store.register_model("chat_m", {"backend": "ttt", "domain": "text"})
    assert main(["calibrate", "chat_m", "--artifacts-root", root]) == 2
    assert "--prompts" in capsys.readouterr().err
    args = build_parser().parse_args(["calibrate", "chat_m", "--prompts", "p.txt", "--cusum-prompts", "c.txt", "--max-new-tokens", "8"])
    assert (args.prompts, args.cusum_prompts, args.max_new_tokens) == ("p.txt", "c.txt", 8)


def test_read_prompts_accepts_lines_or_json_list(tmp_path):
    from plastic.cli import _read_prompts

    lines = tmp_path / "p.txt"
    lines.write_text("first prompt\n\n  second prompt  \n", encoding="utf-8")
    assert _read_prompts(str(lines)) == ["first prompt", "second prompt"]
    js = tmp_path / "p.json"
    js.write_text('["a", "b"]', encoding="utf-8")
    assert _read_prompts(str(js)) == ["a", "b"]
    bad = tmp_path / "bad.json"
    bad.write_text('[1, 2]', encoding="utf-8")
    with pytest.raises(SystemExit):
        _read_prompts(str(bad))


def test_train_physics_via_cli(tmp_path, capsys):
    root = str(tmp_path / "artifacts")
    rc = main(
        [
            "train", "physics",
            "--artifacts-root", root,
            "--steps", "3", "--batch-size", "2", "--seq-len", "32", "--episodes-per-seq", "2",
            "--d-model", "32", "--heads", "2", "--layers", "1", "--chunk", "16",
            "--eval-every", "0", "--save-every", "0", "--eval-batches", "1", "--log-every", "1",
            "--device", "cpu", "--no-muon", "--warmup-steps", "1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    model_id = out[-1]
    assert model_id.startswith("phys_")
    assert ArtifactStore(root).load_model_record(model_id)["status"] == "completed"
    rc = main(["models", "--artifacts-root", root])
    assert rc == 0
    assert model_id in capsys.readouterr().out


def test_calibrate_session_chat_and_physics_via_cli(tmp_path, capsys):
    import os

    from plastic.data.text import encode_documents_to_bin
    from plastic.tokenizer.bpe import Tokenizer

    root = str(tmp_path / "artifacts")
    d = str(tmp_path / "data")
    os.makedirs(d)
    docs = ["alpha beta gamma delta epsilon " * 80, "one two three four five six " * 80]
    tok = Tokenizer.train(docs, vocab_size=300)
    tok.save(os.path.join(d, "tokenizer.json"))
    encode_documents_to_bin(tok, docs, os.path.join(d, "train.bin"))
    encode_documents_to_bin(tok, docs, os.path.join(d, "validation.bin"))
    assert main(["train", "text", "--data", d, "--artifacts-root", root, "--model-id", "lm_t", "--steps", "2",
                 "--batch-size", "2", "--seq-len", "32", "--d-model", "32", "--heads", "2", "--layers", "1", "--chunk", "8",
                 "--eval-every", "0", "--save-every", "0", "--eval-batches", "1", "--log-every", "1", "--device", "cpu",
                 "--warmup-steps", "1", "--mqar-frac", "0"]) == 0
    capsys.readouterr()
    assert main(["calibrate", "lm_t", "--artifacts-root", root, "--data", d, "--chunks", "24", "--fisher-chunks", "4", "--fpr", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "thresholds" in out and os.path.exists(os.path.join(root, "models", "lm_t", "calibration.json"))
    assert main(["session", "new", "--model", "lm_t", "--session-id", "s1", "--artifacts-root", root]) == 0
    assert capsys.readouterr().out.strip() == "s1"
    assert main(["chat", "s1", "alpha beta gamma delta epsilon alpha", "--artifacts-root", root, "--max-new-tokens", "4", "--seed", "0"]) == 0
    capsys.readouterr()
    assert main(["session", "fork", "s1", "s2", "--artifacts-root", root]) == 0
    assert capsys.readouterr().out.strip() == "s2"
    assert main(["session", "list", "--artifacts-root", root]) == 0
    out = capsys.readouterr().out
    assert "s1" in out and "s2" in out
    assert main(["session", "show", "s1", "--artifacts-root", root]) == 0
    assert "n_transactions" in capsys.readouterr().out
    # physics
    assert main(["train", "physics", "--artifacts-root", root, "--model-id", "ph_t", "--steps", "2", "--batch-size", "2",
                 "--seq-len", "32", "--episodes-per-seq", "2", "--d-model", "32", "--heads", "2", "--layers", "1", "--chunk", "8",
                 "--eval-every", "0", "--save-every", "0", "--eval-batches", "1", "--log-every", "1", "--device", "cpu",
                 "--no-muon", "--warmup-steps", "1"]) == 0
    capsys.readouterr()
    assert main(["calibrate", "ph_t", "--artifacts-root", root, "--chunks", "16", "--fisher-chunks", "2", "--fpr", "0.1"]) == 0
    capsys.readouterr()
    assert main(["session", "new", "--model", "ph_t", "--session-id", "p1", "--artifacts-root", root]) == 0
    capsys.readouterr()
    assert main(["physics", "p1", "--artifacts-root", root, "--steps", "24", "--mu", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "adaptive_mse" in out
