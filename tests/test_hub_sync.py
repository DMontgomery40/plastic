"""Exercise the actual publication staging code without contacting the Hub."""

import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


def _stage(tmp_path, monkeypatch, *, chat=False, manifest=True):
    root = tmp_path / "source"
    remote = tmp_path / "remote"
    saved = tmp_path / "download"
    out = tmp_path / "export"
    root.mkdir()
    remote.mkdir()
    saved.mkdir()
    (root / "README.md").write_text("# Current source\n")
    (remote / "README.md").write_text("---\nlicense: mit\n---\nOld card\n")
    folders = ["text", "physics"] + (["text-chat"] if chat else [])
    for name in folders:
        folder = remote / name
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"saved-weight-bytes")
        if name != "text-chat" or manifest:
            (folder / "manifest.json").write_text("{}")
    files = [p for p in remote.rglob("*") if p.is_file()]
    info = SimpleNamespace(sha="hub-revision", siblings=[
        SimpleNamespace(rfilename=p.relative_to(remote).as_posix(),
                        blob_id=hashlib.sha1(p.read_bytes()).hexdigest(), size=p.stat().st_size) for p in files
    ])

    def snapshot_download(repo, *, repo_type, revision, allow_patterns):
        assert revision == info.sha
        for src in files:
            name = src.relative_to(remote).as_posix()
            if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
                dest = saved / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
        return str(saved)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(repo_info=lambda *a, **kw: info),
        snapshot_download=snapshot_download,
    ))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: b"README.md\0")
    for key, value in {
        "HF_TOKEN": "test-only", "GITHUB_WORKSPACE": str(root), "EXPORT_DIR": str(out),
        "HF_REPO_ID": "owner/model", "HF_REPO_TYPE": "model",
        "GITHUB_REPOSITORY": "owner/source", "GITHUB_SHA": "source-revision",
    }.items():
        monkeypatch.setenv(key, value)
    workflow = (Path(__file__).parents[1] / ".github/workflows/sync-to-hub.yml").read_text()
    code = workflow.split("python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    exec(compile(textwrap.dedent(code), "hub-staging", "exec"), {})
    return out


@pytest.mark.parametrize("chat", [False, True])
def test_sync_preserves_published_checkpoint_bytes(tmp_path, monkeypatch, chat):
    out = _stage(tmp_path, monkeypatch, chat=chat)
    hashes = json.loads(out.with_suffix(".hashes.json").read_text())
    for folder in ["text", "physics"]:
        name = f"{folder}/model.safetensors"
        assert (out / name).read_bytes() == b"saved-weight-bytes"
        assert hashes[name] == hashlib.sha256(b"saved-weight-bytes").hexdigest()
    assert (out / "README.md").read_text() == "---\nlicense: mit\n---\n\n# Current source\n"
    assert json.loads((out / "source_snapshot.json").read_text())["github_commit"] == "source-revision"
    if chat:
        preserved = json.loads(out.with_suffix(".preserved.json").read_text())
        assert preserved["text-chat/model.safetensors"]["size"] == len(b"saved-weight-bytes")
        assert not (out / "text-chat").exists()  # no multi-GB copy required to preserve it


def test_sync_never_mirrors_deletions_of_hub_owned_weights():
    workflow = (Path(__file__).parents[1] / ".github/workflows/sync-to-hub.yml").read_text()
    assert "delete_removed: 'false'" in workflow
