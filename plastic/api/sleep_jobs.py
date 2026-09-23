"""Sleep runs as a separate process per job.

The API process holds the models that live sessions chat with. Consolidation changes model weights, so
it must never run on those objects: each job is ``python -m plastic.cli sleep ...`` in its own process,
loading its own copy, writing ``sleep_report.json`` and ``log.txt`` under ``<artifacts>/sleep/<run>``.
This module starts jobs, reads their state back from disk, and keeps the child process handles.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any


def build_sleep_argv(model_id: str, run_dir: str, artifacts_root: str, device: str, options: dict[str, Any],
                     sessions: list[str] | None, probes_path: str | None) -> list[str]:
    argv = [sys.executable, "-m", "plastic.cli", "sleep", model_id, "--artifacts-root", artifacts_root, "--device", device, "--out", run_dir]
    for key, flag in (("method", "--method"), ("target", "--target"), ("steps", "--steps"), ("lr", "--lr"), ("seq_len", "--seq-len"),
                      ("batch_size", "--batch-size"), ("replay_ratio", "--replay-ratio"), ("replay_rows", "--replay-rows"),
                      ("heldout_rows", "--heldout-rows"), ("anchor_lambda", "--anchor-lambda"), ("distill_temperature", "--distill-temperature"),
                      ("tolerance_nll", "--tolerance-nll"), ("seed", "--seed"), ("replay_revision", "--replay-revision")):
        if options.get(key) is not None:
            argv += [flag, str(options[key])]
    if sessions:
        argv += ["--sessions", *sessions]
    if probes_path:
        argv += ["--recall", probes_path]
    return argv


def _tail(path: str, n: int = 30) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    return lines[-n:]


class SleepJobs:
    """Start and observe sleep processes. State lives on disk so a restarted API still lists past runs."""

    def __init__(self, artifacts_root: str, *, device: str) -> None:
        self.root = artifacts_root
        self.device = device
        self.dir = os.path.join(artifacts_root, "sleep")
        self._procs: dict[str, subprocess.Popen] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ start
    def start(self, model_id: str, options: dict[str, Any], *, sessions: list[str] | None, probes: list[dict[str, Any]] | None) -> dict[str, Any]:
        run_id = f"sleep_{int(time.time())}_{options.get('method', 'replay')}_{options.get('target', 'w0')}"
        run_dir = os.path.join(self.dir, run_id)
        n = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(self.dir, f"{run_id}_{n}")
            n += 1
        run_id = os.path.basename(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        probes_path = None
        if probes:
            probes_path = os.path.join(run_dir, "probes.json")
            with open(probes_path, "w", encoding="utf-8") as f:
                json.dump(probes, f, indent=1)
        request = {"model_id": model_id, "options": options, "sessions": sessions, "n_probes": len(probes or []), "started_at_unix": int(time.time())}
        with open(os.path.join(run_dir, "request.json"), "w", encoding="utf-8") as f:
            json.dump(request, f, indent=1)
        argv = build_sleep_argv(model_id, run_dir, self.root, self.device, options, sessions, probes_path)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)  # the child imports what the API imports
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        out = open(os.path.join(run_dir, "stdout.txt"), "ab")
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
        with self._guard:
            self._procs[run_id] = proc
            self._meta[run_id] = {**request, "pid": proc.pid, "run_dir": run_dir}
        return self.status(run_id)

    # ------------------------------------------------------------------ observe
    def _report(self, run_dir: str) -> dict[str, Any] | None:
        p = os.path.join(run_dir, "sleep_report.json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None  # being written

    def status(self, run_id: str) -> dict[str, Any]:
        run_dir = os.path.join(self.dir, run_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"sleep run not found: {run_id}")
        with self._guard:
            proc = self._procs.get(run_id)
            meta = dict(self._meta.get(run_id) or {})
        if not meta:
            p = os.path.join(run_dir, "request.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    meta = json.load(f)
        report = self._report(run_dir)
        exit_code = proc.poll() if proc is not None else None
        running = proc is not None and exit_code is None
        if report and report.get("status") in ("accepted", "accepted_unmeasured", "rejected"):
            status = report["status"]
        elif running:
            status = "running"
        elif proc is not None and exit_code not in (None, 0, 3):
            status = "failed"
        elif proc is None and report is None:
            status = "unknown"  # a run from a previous API process that never wrote a report
        elif proc is None:
            status = report.get("status", "unknown") if report else "unknown"
        else:
            status = "failed" if report is None else report.get("status", "failed")
        return {
            "run_id": run_id, "model_id": meta.get("model_id") or (report or {}).get("parent_model_id"),
            "status": status, "exit_code": exit_code, "pid": meta.get("pid"),
            "started_at_unix": meta.get("started_at_unix") or (report or {}).get("created_at_unix"),
            "options": meta.get("options"), "sessions": meta.get("sessions"), "n_probes": meta.get("n_probes"),
            "report": report, "log_tail": _tail(os.path.join(run_dir, "log.txt")),
            "stderr_tail": _tail(os.path.join(run_dir, "stdout.txt"), 12) if status == "failed" else [],
        }

    def list(self) -> list[dict[str, Any]]:
        if not os.path.isdir(self.dir):
            return []
        out = []
        for name in sorted(os.listdir(self.dir), reverse=True):
            if os.path.isdir(os.path.join(self.dir, name)):
                try:
                    s = self.status(name)
                except FileNotFoundError:
                    continue
                s.pop("log_tail", None)
                s.pop("stderr_tail", None)
                if s["report"]:
                    s["report"] = {k: v for k, v in s["report"].items() if k not in ("losses",)}
                out.append(s)
        return out
