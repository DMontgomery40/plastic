"""Local training jobs: ``python -m plastic.cli train ...`` as a child process.

Status is derived from two sources that can disagree for a second or two: the
process (alive or exited, with its exit code) and the model record the training
loop writes into the store once it has started.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from plastic.api.schemas import TrainRequest
from plastic.api.service import latest_log_record
from plastic.store import ArtifactStore

LOG_NAME = "train_stdout.log"


@dataclass
class TrainJob:
    model_id: str
    domain: str
    pid: int
    started_at_unix: int
    log_path: str
    proc: subprocess.Popen
    handle: Any = None

    def poll(self) -> int | None:
        code = self.proc.poll()
        if code is not None and self.handle is not None:
            try:
                self.handle.close()
            finally:
                self.handle = None
        return code

    def status(self) -> str:
        return "running" if self.poll() is None else "finished"

    def to_dict(self) -> dict[str, Any]:
        code = self.poll()
        return {
            "model_id": self.model_id,
            "pid": self.pid,
            "status": "running" if code is None else "finished",
            "exit_code": code,
            "started_at_unix": self.started_at_unix,
        }


class JobManager:
    def __init__(self, store: ArtifactStore, *, device: str = "cpu") -> None:
        self.store = store
        self.device = device
        self._jobs: dict[str, TrainJob] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ start
    def command(self, req: TrainRequest, model_id: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "plastic.cli",
            "train",
            req.domain,
            "--artifacts-root",
            self.store.root,
            "--model-id",
            model_id,
            "--steps",
            str(req.steps),
            "--batch-size",
            str(req.batch_size),
            "--seq-len",
            str(req.seq_len),
            "--device",
            str(req.device or self.device),
        ]
        if req.domain == "text" and req.data_dir:  # --data exists only on the text subparser
            cmd += ["--data", req.data_dir]
        for flag, value in (
            ("--d-model", req.d_model),
            ("--layers", req.layers),
            ("--heads", req.heads),
            ("--chunk", req.chunk),
            ("--eval-every", req.eval_every),
            ("--save-every", req.save_every),
        ):
            if value is not None:
                cmd += [flag, str(value)]
        if req.adversarial:
            cmd.append("--adversarial")
        return cmd

    def start(self, req: TrainRequest) -> dict[str, Any]:
        model_id = req.model_id or self.store.new_model_id("lm" if req.domain == "text" else "phys")
        model_dir = self.store.model_dir(model_id)
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, LOG_NAME)
        handle = open(log_path, "ab")  # noqa: SIM115 - closed when the job is reaped
        cmd = self.command(req, model_id)
        handle.write(f"$ {' '.join(cmd)}\n".encode())
        handle.flush()
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
        job = TrainJob(
            model_id=model_id,
            domain=req.domain,
            pid=proc.pid,
            started_at_unix=int(time.time()),
            log_path=log_path,
            proc=proc,
            handle=handle,
        )
        with self._guard:
            self._jobs[model_id] = job
        return {"model_id": model_id, "pid": proc.pid}

    # ------------------------------------------------------------------ read
    def jobs(self) -> list[dict[str, Any]]:
        with self._guard:
            jobs = list(self._jobs.values())
        out = [j.to_dict() for j in jobs]
        out.sort(key=lambda j: int(j["started_at_unix"]), reverse=True)
        return out

    def get(self, model_id: str) -> TrainJob | None:
        with self._guard:
            return self._jobs.get(model_id)

    def _log_tail(self, path: str, n: int = 800) -> str | None:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - n))
                return f.read().decode("utf-8", "replace").strip() or None
        except OSError:
            return None

    def status(self, model_id: str) -> dict[str, Any] | None:
        job = self.get(model_id)
        try:
            record: dict[str, Any] | None = self.store.load_model_record(model_id)
        except FileNotFoundError:
            record = None
        if job is None and record is None:
            return None

        exit_code = job.poll() if job is not None else None
        if job is not None:
            status = "running" if exit_code is None else "finished"
        elif record is not None and record.get("status") == "running":
            status = "running"
        else:
            status = "finished"
        # the training loop registers the model a moment after the process starts
        phase = str(record.get("status")) if record else "starting"

        error = record.get("error") if record else None
        if error is None and exit_code not in (None, 0) and job is not None:
            error = self._log_tail(job.log_path)
        return {
            "model_id": model_id,
            "status": status,
            "phase": phase,
            "pid": job.pid if job is not None else None,
            "exit_code": exit_code,
            "latest": latest_log_record(self.store, model_id) if record is not None else None,
            "eval": self.store.read_eval(model_id) if record is not None else None,
            "error": error,
        }

    # ------------------------------------------------------------------ cancel
    def cancel(self, model_id: str) -> dict[str, Any] | None:
        job = self.get(model_id)
        if job is None:
            return None
        if job.poll() is None:
            job.proc.terminate()
            try:
                job.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.proc.kill()
                job.proc.wait(timeout=5)
            job.poll()
        return {"model_id": model_id, "status": job.status()}
