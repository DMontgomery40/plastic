"""The TTT chat model under the text learning contract: baselines and propose-and-verify.

``TextRuleLearner`` implements ``plastic.eval.text_contract.TextLearner`` over a ``TTTBackend``:

- ``score(batch, adapt)`` renders each episode as a chat (``Episode.messages``), runs it from a fresh state in one
  forward pass with the fast path writing (``adapt=True``) or frozen (``freeze=True``), and returns per situation
  the mean NLL over the assistant's answer tokens and a teacher-forced exact match (every answer token is the
  argmax given the true prefix). Teacher-forced exact is cheaper than greedy generation and is labelled as such in
  reports (``exact_tf``); greedy generation can be added per report when needed.
- ``consume(stream)`` is the lasting update, by mode:
  ``frozen``: nothing (with ``adapt=True`` at scoring time this is the TTT-only baseline);
  ``continued``: cross-entropy on the stream's assistant spans, no verification, accepts everything;
  ``in_context``: nothing lasting; the stream's lessons are read into the fast weights immediately before each scored
  episode without a reset, and counted. On this model "context" is the fast-weight state (plus a short convolution
  state), not an attention window, so this arm measures what one pass over the lessons leaves in the fast weights;
  ``replay_verify``: propose = the same cross-entropy update; verify on training-distribution material only
  (exact and nll on fresh training compositions must not fall by more than a tolerance; held-out chat NLL, when a
  corpus is given, must not rise by more than a tolerance); accept installs, refuse restores the snapshot.
- ``snapshot_slow`` / ``restore_slow`` copy the initial fast weights W0 (target ``w0``) or every parameter
  (target ``all``) to CPU.

Nothing here decides anything with keywords; verification uses the model's own numbers on held-in material.
Spec: docs/superpowers/specs/2026-09-23-text-rule-contract.md
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

import torch
import torch.nn.functional as F

from plastic.data.rules import Episode, RuleBatch, rule_batch
from plastic.sleep.ttt import fast_weight_parameters, select_target

MODES = ("frozen", "continued", "in_context", "replay_verify")


def answer_spans(labels: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of supervised positions in ``labels`` (``-100`` elsewhere): one run per assistant turn,
    in order. Positions index the token being predicted, i.e. ``labels[t]`` is predicted by ``logits[t-1]``."""
    spans: list[tuple[int, int]] = []
    start = None
    for t, lab in enumerate(labels):
        if lab != -100 and start is None:
            start = t
        elif lab == -100 and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def span_scores(logits: torch.Tensor, ids: list[int], labels: list[int]) -> list[dict[str, float]]:
    """Per assistant span: mean NLL over its tokens and teacher-forced exact (all argmaxes equal the targets)."""
    out: list[dict[str, float]] = []
    lp = torch.log_softmax(logits.float(), dim=-1)
    for a, b in answer_spans(labels):
        tgt = torch.tensor(labels[a:b], dtype=torch.long, device=lp.device)
        rows = lp[a - 1:b - 1]  # logits at t-1 predict token t
        nll = -rows.gather(1, tgt[:, None])[:, 0]
        exact = bool((rows.argmax(-1) == tgt).all())
        out.append({"exact": 1.0 if exact else 0.0, "nll": float(nll.mean()), "tokens": int(b - a)})
    return out


class TextRuleLearner:
    def __init__(self, backend, *, mode: str = "frozen", target: str = "w0", lr: float = 1e-4, steps: int = 20,
                 verify_tolerance_exact: float = 0.05, verify_tolerance_nll: float = 0.1, verify_tolerance_chat_nll: float = 0.05,
                 chat_nll: Callable[[], dict[str, float]] | None = None, verify_seed: int = 7, verify_episodes: int = 6,
                 verify_adapt: bool = True, log: Callable[[str], None] = lambda s: None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if target not in ("w0", "all"):
            raise ValueError(f"unknown target {target!r}")
        self.backend, self.mode, self.target, self.lr, self.steps = backend, mode, target, float(lr), int(steps)
        self.tol_exact, self.tol_nll, self.tol_chat = verify_tolerance_exact, verify_tolerance_nll, verify_tolerance_chat_nll
        self.chat_nll, self.verify_seed, self.verify_episodes, self.log = chat_nll, verify_seed, verify_episodes, log
        # verifier v2 scores the held-in checks with the fast path adapting (v1 scored it frozen, where exact was 0 on this
        # model before and after: a vacuous check) and adds the first-situation exact, which no example precedes
        self.verify_adapt = bool(verify_adapt)
        self.context: list[Episode] = []          # in_context mode: the stream's episodes, prepended at scoring time
        self.context_situations_measured = 0
        self.train_compositions: list[tuple[str, ...]] | None = None  # set by the caller for verification material
        self.rule_set = "transform"

    # ---------------------------------------------------------------- slow state
    def _slow_params(self):
        return fast_weight_parameters(self.backend.model) if self.target == "w0" else list(self.backend.model.named_parameters())

    def snapshot_slow(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().to("cpu", copy=True) for n, p in self._slow_params()}

    def restore_slow(self, snapshot: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in self._slow_params():
                p.copy_(snapshot[n].to(p.device, p.dtype))
        self.context = []

    def parameter_count(self) -> int:
        return int(sum(p.numel() for _, p in self._slow_params()))

    # ---------------------------------------------------------------- scoring
    def _encode(self, episode: Episode) -> tuple[list[int], list[int], int]:
        """Token ids and labels for the episode, with the in-context prefix (if any) rendered as earlier turns whose
        answers are NOT scored; returns the number of leading label positions to ignore."""
        from plastic.backends.ttt_lm.backend import encode_conversation

        msgs = []
        if self.mode == "in_context" and self.context:
            for e in self.context:
                msgs += e.messages()
        prefix_msgs = len(msgs)
        msgs += episode.messages()
        ids, labels = encode_conversation(self.backend.tokenizer, msgs)
        if prefix_msgs:
            # everything before the scored episode's first user turn is context: mask its labels
            _, prefix_labels = encode_conversation(self.backend.tokenizer, msgs[:prefix_msgs])
            n_prefix = len(prefix_labels)
            labels = [-100] * n_prefix + labels[n_prefix:]
            self.context_situations_measured += sum(len(e.situations) for e in self.context)
        return ids, labels, prefix_msgs

    @torch.no_grad()
    def score(self, batch: RuleBatch, *, adapt: bool) -> list[list[dict[str, float]]]:
        out: list[list[dict[str, float]]] = []
        for ep in batch.episodes:
            ids, labels, _ = self._encode(ep)
            logits, _, _ = self.backend.forward(ids, self.backend.init_state(), freeze=not adapt, beta_scale=1.0)
            scores = span_scores(logits, ids, labels)
            if len(scores) != len(ep.situations):
                raise RuntimeError(f"expected {len(ep.situations)} answer spans, found {len(scores)}")
            out.append(scores)
        return out

    # ---------------------------------------------------------------- the lasting update
    def _train_on(self, stream: RuleBatch) -> dict[str, Any]:
        from plastic.backends.ttt_lm.backend import encode_conversation

        model = self.backend.model
        params = select_target(model, self.target)
        opt = torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.0)
        rows = [encode_conversation(self.backend.tokenizer, e.messages()) for e in stream.episodes]
        rng = random.Random(self.verify_seed)
        losses: list[float] = []
        model.eval()
        for step in range(1, self.steps + 1):
            ids, labels = rows[rng.randrange(len(rows))]
            x = torch.tensor([ids], dtype=torch.long, device=self.backend.device)
            y = torch.tensor([labels], dtype=torch.long, device=self.backend.device)
            with torch.enable_grad():
                logits = model(x, use_cache=False).logits[:, :-1].float()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)
                if not math.isfinite(float(loss)):
                    raise RuntimeError("non-finite loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            losses.append(float(loss.detach()))
        model.requires_grad_(False)
        if self.backend.device.type == "mps":
            torch.mps.synchronize()
        return {"steps": self.steps, "lr": self.lr, "target": self.target, "loss_first": losses[0], "loss_last": losses[-1]}

    def _verify_batch(self) -> RuleBatch:
        if not self.train_compositions:
            raise RuntimeError("replay_verify needs train_compositions for its held-in verification material")
        return rule_batch(self.train_compositions, episodes=self.verify_episodes, n_situations=3, seed=self.verify_seed, split_tag="train", rule_set=self.rule_set)

    def _summ(self, scores) -> dict[str, float]:
        flat = [s for ep in scores for s in ep]
        first = [ep[0] for ep in scores if ep]
        return {"exact": sum(s["exact"] for s in flat) / len(flat), "nll": sum(s["nll"] for s in flat) / len(flat),
                "exact_first": sum(s["exact"] for s in first) / len(first), "nll_first": sum(s["nll"] for s in first) / len(first)}

    @property
    def verifier_version(self) -> str | None:
        """Named only for the mode that verifies; the baselines have no verifier."""
        if self.mode != "replay_verify":
            return None
        return "v2" if self.verify_adapt else "v1"

    def _verify_checks(self, before: dict[str, float], after: dict[str, float]) -> list[dict[str, Any]]:
        checks = [
            {"name": "train_exact_drop", "value": before["exact"] - after["exact"], "limit": self.tol_exact, "passed": before["exact"] - after["exact"] <= self.tol_exact},
            {"name": "train_nll_rise", "value": after["nll"] - before["nll"], "limit": self.tol_nll, "passed": after["nll"] - before["nll"] <= self.tol_nll},
        ]
        if self.verify_adapt:
            drop = before["exact_first"] - after["exact_first"]
            checks.append({"name": "train_exact_first_drop", "value": drop, "limit": self.tol_exact, "passed": drop <= self.tol_exact})
        return checks

    def consume(self, stream: RuleBatch) -> dict[str, Any]:
        rec: dict[str, Any] = {"mode": self.mode, "stream_poisoned": stream.poisoned, "episodes": len(stream.episodes), "situations": stream.situations}
        if self.mode == "frozen":
            rec.update(accepted=None, note="no lasting update")
            return rec
        if self.mode == "in_context":
            self.context = list(stream.episodes)
            rec.update(accepted=None, note="the lessons are read into the fast weights immediately before each scored episode, without a reset "
                                            "(this model has no attention window: 'context' is the fast-weight state plus a short convolution state); nothing lasting")
            return rec
        if self.mode == "continued":
            rec.update(self._train_on(stream), accepted=True, note="continued training, no verification")
            return rec
        # replay_verify: propose, then verify on held-in material only
        snapshot = self.snapshot_slow()
        vb = self._verify_batch()
        before = self._summ(self.score(vb, adapt=self.verify_adapt))
        chat_before = self.chat_nll() if self.chat_nll else None
        rec.update(self._train_on(stream))
        after = self._summ(self.score(vb, adapt=self.verify_adapt))
        chat_after = self.chat_nll() if self.chat_nll else None
        checks = self._verify_checks(before, after)
        if chat_before is not None and chat_after is not None:
            rise = float(chat_after["mean"]) - float(chat_before["mean"])
            checks.append({"name": "chat_nll_rise", "value": rise, "limit": self.tol_chat, "passed": rise <= self.tol_chat})
        accepted = all(c["passed"] for c in checks)
        if not accepted:
            self.restore_slow(snapshot)
        rec.update(accepted=accepted, verifier=self.verifier_version,
                   verify={"adapt": self.verify_adapt, "before": before, "after": after, "chat_before": chat_before, "chat_after": chat_after, "checks": checks},
                   note="held-in verification only: training compositions with fresh inputs, never the held-out compositions; "
                        + ("scored with the fast path adapting, first-situation exact included (v2)" if self.verify_adapt else "scored with the fast path frozen (v1)"))
        self.log(f"[text-learner] {self.mode} ({self.verifier_version}): {'accepted' if accepted else 'refused'} " + ", ".join(f"{c['name']} {c['value']:+.3f}/{c['limit']}" for c in checks))
        return rec
