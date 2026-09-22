import pytest

from plastic.cli import build_parser, main
from plastic.store import ArtifactStore


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for cmd in ("data", "train", "models", "bench"):
        assert cmd in out


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
