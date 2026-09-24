"""The TTT chat model under the text learning contract: baselines and propose-and-verify.

``TextRuleLearner`` implements ``plastic.eval.text_contract.TextLearner`` over a ``TTTBackend``:

- ``score(batch, adapt)`` renders each episode as a chat (``Episode.messages``), runs it from a fresh state in one
  forward pass with the fast path writing (``adapt=True``) or frozen (``freeze=True``), and returns per situation
  the mean NLL over the assistant's answer tokens and a teacher-forced exact match (every answer token is the
  argmax given the true prefix). Teacher-forced exact is cheaper than greedy generation and is labelled as such in
  reports (``exact_tf``); greedy generation can be added per report when needed.
- ``consume(stream)`` is the lasting update, by mode:
  ``frozen``: nothing (with ``adapt=True`` at scoring time this is the TTT-only baseline);
  ``continued``: cross-entropy on the stream's assistant spans, no verification, accepts everything. The steps visit
  the stream in full shuffled passes (``sampling="passes"``: every episode once per pass); ``sampling="draws"``
  reproduces the archived runs, which drew 20 episodes with replacement and so trained 11 of 17 compositions
  (OPUS-LEAD-001). Every record names the compositions actually trained and how many steps were poisoned;
  ``in_context``: nothing lasting; the stream's lessons are read into the fast weights immediately before each scored
  episode without a reset, and counted. On this model "context" is the fast-weight state (plus a short convolution
  state), not an attention window, so this arm measures what one pass over the lessons leaves in the fast weights;
  ``replay_verify``: propose = the same cross-entropy update; verify on training-distribution material only
  (exact and nll on fresh training compositions must not fall by more than a tolerance; held-out chat NLL, when a
  corpus is given, must not rise by more than a tolerance); accept installs, refuse restores the snapshot.
Sleep's consolidation operators as lasting updates (step 2 of OPUS-LEAD-001; ungated like ``continued``, the same
AdamW settings, one step per stream episode in the same order):

  ``anchor``: forward-only. Each stream episode is read from a fresh state with the fast path writing; W0 moves
  ``anchor_lambda`` of the way to the mean final fast weights (the Sleep anchor; a Reptile-form update whose inner
  optimizer is the TTT reconstruction loop, not the task loss, so Reptile's rationale is not assumed).
  ``distill``: context distillation. The teacher is the model before this consume reading each whole episode, so every
  answer after the first is predicted with the earlier worked examples in its fast weights; its answer-token
  distributions are fixed before the student moves (W0 is shared). The student reads each situation alone (the
  preface and that one prompt) from a fresh state and is trained on KL(teacher || student) over the answer tokens.
  The chat Sleep distill reads a saved session state with the fast path frozen; this is the in-context form.
  ``dream``: new inputs, generated programmatically for the stream's compositions; the teacher, after reading the
  stream episode, answers each greedily; the student trains on those episodes like ``continued``. No stream answer
  is used as a label, and the record states how often the teacher's answers match the stream's labelling and the
  true rule. ``dream_truth``: the same new inputs with the stream's own labelling (the teacher-accuracy control).

- ``snapshot_slow`` / ``restore_slow`` copy the initial fast weights W0 (target ``w0``) or every parameter
  (target ``all``) to CPU.
- ``choice(batch)``: for the first situation of each episode (no worked example before it), the summed log-probability
  of every composition's output on the same input, and whether the correct one ranks first. What the slow weights
  carry about a composition shows here as a ranking against one chance level, where a teacher-forced exact on a few
  items cannot separate content from answer format.

Nothing here decides anything with keywords; verification uses the model's own numbers on held-in material.
Spec: docs/superpowers/specs/2026-09-23-text-rule-contract.md
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Callable

import torch
import torch.nn.functional as F

from plastic.data.rules import (Episode, RuleBatch, Situation, all_compositions, apply, normalize_output, permute_names, poison_batch, rule_batch,
                                shuffle_answers)
from plastic.sleep.ttt import fast_weight_parameters, select_target

MODES = ("frozen", "continued", "in_context", "replay_verify", "anchor", "distill", "dream", "dream_truth")
SLEEP_MODES = ("anchor", "distill", "dream", "dream_truth")
SAMPLING = ("passes", "draws")


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
                 verify_adapt: bool = True, sampling: str = "passes", passes: int = 1, anchor_lambda: float = 0.5, dream_seed: int = 11,
                 dream_max_new_tokens: int = 32, log: Callable[[str], None] = lambda s: None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if sampling not in SAMPLING:
            raise ValueError(f"unknown sampling {sampling!r}; expected one of {SAMPLING}")
        if passes < 1:
            raise ValueError("passes must be at least 1")
        self.sampling, self.passes = sampling, int(passes)
        self.anchor_lambda, self.dream_seed, self.dream_max_new_tokens = float(anchor_lambda), int(dream_seed), int(dream_max_new_tokens)
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
        self.stated_rules = True  # the verification material's preface follows the spec (set by the caller with rule_set)

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

    def _context_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.mode == "in_context" and self.context:
            for e in self.context:
                msgs += e.messages()
        return msgs

    @torch.no_grad()
    def _answer_logprobs(self, rows: list[tuple[list[int], list[int]]], token_budget: int = 16384) -> list[tuple[float, int]]:
        """Summed log-probability and token count of the LAST answer span of each (ids, labels) row, with the fast path
        writing as in ordinary use. Rows are right-padded into batches; the model is causal, so padding after a row
        cannot reach its positions (a test pins batched against single-row values)."""
        model, dev = self.backend.model, self.backend.device
        tok = self.backend.tokenizer
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        out: list[tuple[float, int]] = []
        L = max(len(ids) for ids, _ in rows)
        per = max(1, token_budget // max(L, 1))
        for k in range(0, len(rows), per):
            chunk = rows[k:k + per]
            x = torch.full((len(chunk), L), pad, dtype=torch.long, device=dev)
            for j, (ids, _) in enumerate(chunk):
                x[j, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            logits = model(x, use_cache=False).logits.float()
            for j, (ids, labels) in enumerate(chunk):
                a, b = answer_spans(labels)[-1]
                tgt = torch.tensor(labels[a:b], dtype=torch.long, device=dev)
                lp = torch.log_softmax(logits[j, a - 1:b - 1], dim=-1).gather(1, tgt[:, None])[:, 0]
                out.append((float(lp.sum()), int(b - a)))
        return out

    def choice(self, batch: RuleBatch) -> list[dict[str, Any]]:
        """First-situation ranking of every composition's output (``all_compositions``, deduplicated) on the same input,
        by summed answer log-probability including the end-of-answer token. ``choice`` is 1 when the correct output ranks
        first; ``margin`` is its log-probability minus the best other candidate's; chance is 1 / candidates."""
        from plastic.backends.ttt_lm.backend import encode_conversation

        comps = all_compositions(batch.rule_set)
        ctx = self._context_messages()
        out: list[dict[str, Any]] = []
        for ep in batch.episodes:
            s = ep.situations[0]
            owners: dict[str, list[str]] = {}
            for c in comps:
                owners.setdefault(" ".join(apply(c, list(s.words), rule_set=batch.rule_set)), []).append(" ".join(c))
            texts = list(owners)
            if s.answer_text not in owners:
                raise RuntimeError(f"the correct output for {ep.ops} is not among the candidates")
            first_user = ep.messages()[0]
            rows = [encode_conversation(self.backend.tokenizer, ctx + [first_user, {"role": "assistant", "content": t}]) for t in texts]
            lps = [lp for lp, _ in self._answer_logprobs(rows)]
            ci = texts.index(s.answer_text)
            best_other = max(lp for i, lp in enumerate(lps) if i != ci)
            rank = 1 + sum(1 for i, lp in enumerate(lps) if i != ci and lp > lps[ci])
            top = max(range(len(lps)), key=lambda i: lps[i])
            out.append({"ops": " ".join(ep.ops), "choice": 1.0 if rank == 1 else 0.0, "margin": lps[ci] - best_other, "rank": rank,
                        "correct_logp": lps[ci], "candidates": len(texts), "top": owners[texts[top]][0]})
        if ctx:
            self.context_situations_measured += len(batch.episodes) * sum(len(e.situations) for e in self.context)
        return out

    # ---------------------------------------------------------------- Sleep operators on the rule task
    @staticmethod
    def _coverage(stream: RuleBatch, order: list[int]) -> dict[str, Any]:
        trained: dict[str, int] = {}
        for i in order:
            key = " ".join(stream.episodes[i].ops)
            trained[key] = trained.get(key, 0) + 1
        all_keys = {" ".join(e.ops) for e in stream.episodes}
        return {"trained": trained, "compositions_trained": len(trained), "compositions_in_stream": len(all_keys),
                "untrained": sorted(all_keys - set(trained)), "poisoned_steps": sum(1 for i in order if stream.episodes[i].poisoned)}

    @torch.no_grad()
    def _anchor_on(self, stream: RuleBatch) -> dict[str, Any]:
        from plastic.backends.ttt_lm.backend import encode_conversation
        from plastic.sleep.ttt import anchor_update, w0_relative_change, w0_snapshot

        be = self.backend
        w0_before = w0_snapshot(be.model)
        leaves = []
        for e in stream.episodes:
            ids, _ = encode_conversation(be.tokenizer, e.messages())
            _, st, _ = be.forward(ids, be.init_state(), freeze=False, beta_scale=1.0)
            leaves.append([t.detach().clone() for t in st.effective_leaves(be._c15)])
        anchor_update(be.model, leaves, self.anchor_lambda)
        return {"sessions": len(leaves), "lambda": self.anchor_lambda, "w0_relative_change": w0_relative_change(be.model, w0_before)["total"], "forward_only": True,
                "steps": 0, "sampling": "every episode once, averaged", **self._coverage(stream, list(range(len(stream.episodes))))}

    def _single_situation(self, e: Episode, s: Situation) -> list[dict[str, str]]:
        """One situation as its own chat: the preface and that prompt, as the first situation of a measurement reads."""
        return replace(e, situations=(s,)).messages()

    def _distill_on(self, stream: RuleBatch) -> dict[str, Any]:
        from plastic.backends.ttt_lm.backend import encode_conversation

        model, tok, dev = self.backend.model, self.backend.tokenizer, self.backend.device
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        teacher: list[list[torch.Tensor]] = []
        students: list[list[tuple[list[int], list[int]]]] = []
        with torch.no_grad():
            for e in stream.episodes:
                ids, labels = encode_conversation(tok, e.messages())
                logits = model(torch.tensor([ids], dtype=torch.long, device=dev), use_cache=False).logits[0].float()
                spans = answer_spans(labels)
                teacher.append([torch.log_softmax(logits[a - 1:b - 1], dim=-1) for a, b in spans])
                rows = [encode_conversation(tok, self._single_situation(e, s)) for s in e.situations]
                for (a, b), (sids, slabels) in zip(spans, rows):
                    sa, sb = answer_spans(slabels)[-1]
                    if labels[a:b] != slabels[sa:sb]:
                        raise RuntimeError("teacher and student answer tokens differ; the renderings must tokenize the answer alike")
                students.append(rows)
        params = select_target(model, self.target)
        opt = torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.0)
        order = self.training_order(len(stream.episodes))
        losses: list[float] = []
        model.eval()
        for i in order:
            rows = students[i]
            L = max(len(r[0]) for r in rows)
            x = torch.full((len(rows), L), pad, dtype=torch.long, device=dev)
            for j, (sids, _) in enumerate(rows):
                x[j, :len(sids)] = torch.tensor(sids, dtype=torch.long, device=dev)
            with torch.enable_grad():
                s_logits = model(x, use_cache=False).logits.float()
                terms = []
                for j, (sids, slabels) in enumerate(rows):
                    a, b = answer_spans(slabels)[-1]
                    s_lp = torch.log_softmax(s_logits[j, a - 1:b - 1], dim=-1)
                    t_lp = teacher[i][j]
                    terms.append((t_lp.exp() * (t_lp - s_lp)).sum(-1))
                loss = torch.cat(terms).mean()
                if not math.isfinite(float(loss.detach())):
                    raise RuntimeError("non-finite loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            losses.append(float(loss.detach()))
        model.requires_grad_(False)
        if self.backend.device.type == "mps":
            torch.mps.synchronize()
        return {"steps": len(order), "lr": self.lr, "target": self.target, "loss_first": losses[0], "loss_last": losses[-1], "sampling": self.sampling,
                "teacher": "the pre-consume model reading each whole episode (in-context); fixed before the student moves",
                "objective": "KL(teacher || student) over answer tokens, student reads one situation alone", **self._coverage(stream, order)}

    def _dream_inputs(self, stream: RuleBatch) -> RuleBatch:
        """New word lists for the stream's compositions, labelled by the stream's own generating process (poison and
        format-only included), from a seed of the learner's own that no measurement uses."""
        n_sit = len(stream.episodes[0].situations)
        fresh = rule_batch([e.ops for e in stream.episodes], episodes=len(stream.episodes), n_situations=n_sit, seed=self.dream_seed,
                           split_tag="dream", rule_set=stream.rule_set, stated=stream.stated)
        if stream.poisoned:
            fresh = poison_batch(fresh, operator=stream.poisoned_operator, kind=stream.poison_kind or "consistent")
        if stream.format_only:
            fresh = shuffle_answers(fresh, seed=self.dream_seed)
        if stream.name_map:
            fresh = permute_names(fresh, mapping=dict(stream.name_map))
        return fresh

    @torch.no_grad()
    def _teacher_answers(self, lesson: Episode, prompts: list[Situation]) -> list[str]:
        """The model after reading ``lesson`` (fast path writing) answers each prompt greedily as the next turn of that
        episode, each from the same post-lesson state."""
        from plastic.backends.ttt_lm.backend import encode_conversation

        be, tok = self.backend, self.backend.tokenizer
        msgs = lesson.messages()
        lesson_ids, _ = encode_conversation(tok, msgs)
        _, st, _ = be.forward(lesson_ids, be.init_state(), freeze=False, beta_scale=1.0)
        eos = int(tok.eos_token_id)
        out: list[str] = []
        for s in prompts:
            full, _ = encode_conversation(tok, msgs + [{"role": "user", "content": s.prompt}, {"role": "assistant", "content": ""}])
            if full[:len(lesson_ids)] != lesson_ids or full[-1] != eos:
                raise RuntimeError("the next-turn rendering does not extend the lesson's ids")
            logits, st2, _ = be.forward(full[len(lesson_ids):-1], st.clone(), freeze=False, beta_scale=1.0)
            gen: list[int] = []
            for _ in range(self.dream_max_new_tokens):
                nxt = int(logits[-1].argmax())
                if nxt == eos:
                    break
                gen.append(nxt)
                logits, st2, _ = be.forward([nxt], st2, freeze=False, beta_scale=1.0)
            out.append(tok.decode(gen, skip_special_tokens=True))
        return out

    def _dream_on(self, stream: RuleBatch) -> dict[str, Any]:
        truth = self._dream_inputs(stream)
        rec: dict[str, Any] = {"dream_seed": self.dream_seed, "labels": "stream labelling" if self.mode == "dream_truth" else "teacher"}
        if self.mode == "dream_truth":
            dreams = truth
        else:
            eps, match_stream, match_true, empty, n = [], 0, 0, 0, 0
            for lesson, fresh in zip(stream.episodes, truth.episodes):
                answers = self._teacher_answers(lesson, list(fresh.situations))
                sits = []
                for s, a in zip(fresh.situations, answers):
                    words = tuple(normalize_output(a).split())
                    n += 1
                    empty += not words
                    match_stream += " ".join(words) == s.answer_text
                    match_true += " ".join(words) == " ".join(apply(s.ops, list(s.words), rule_set=stream.rule_set))
                    sits.append(Situation(s.ops, s.words, words))
                eps.append(replace(fresh, situations=tuple(sits), poisoned=lesson.poisoned))
            dreams = replace(truth, episodes=eps)
            rec.update(teacher_matches_stream=match_stream / n, teacher_matches_true_rule=match_true / n, teacher_empty=empty, teacher_answers=n)
            self.log(f"[text-learner] dream teacher: {match_stream}/{n} match the stream's labelling, {match_true}/{n} the true rule, {empty} empty")
        rec.update(self._train_on(dreams))
        return rec

    # ---------------------------------------------------------------- the lasting update
    def _train_on(self, stream: RuleBatch) -> dict[str, Any]:
        from plastic.backends.ttt_lm.backend import encode_conversation

        model = self.backend.model
        params = select_target(model, self.target)
        opt = torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.0)
        rows = [encode_conversation(self.backend.tokenizer, e.messages()) for e in stream.episodes]
        order = self.training_order(len(rows))
        losses: list[float] = []
        model.eval()
        for i in order:
            ids, labels = rows[i]
            x = torch.tensor([ids], dtype=torch.long, device=self.backend.device)
            y = torch.tensor([labels], dtype=torch.long, device=self.backend.device)
            with torch.enable_grad():
                logits = model(x, use_cache=False).logits[:, :-1].float()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)
                if not math.isfinite(float(loss.detach())):
                    raise RuntimeError("non-finite loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            losses.append(float(loss.detach()))
        model.requires_grad_(False)
        if self.backend.device.type == "mps":
            torch.mps.synchronize()
        trained: dict[str, int] = {}
        for i in order:
            key = " ".join(stream.episodes[i].ops)
            trained[key] = trained.get(key, 0) + 1
        all_keys = {" ".join(e.ops) for e in stream.episodes}
        return {"steps": len(order), "lr": self.lr, "target": self.target, "loss_first": losses[0], "loss_last": losses[-1],
                "sampling": self.sampling, "passes": self.passes if self.sampling == "passes" else None,
                "trained": trained, "compositions_trained": len(trained), "compositions_in_stream": len(all_keys),
                "untrained": sorted(all_keys - set(trained)), "poisoned_steps": sum(1 for i in order if stream.episodes[i].poisoned)}

    def training_order(self, n: int) -> list[int]:
        """Episode indices in the order the lasting update visits them. ``passes``: ``self.passes`` shuffled passes, every
        episode once per pass. ``draws``: ``self.steps`` draws with replacement from ``Random(verify_seed)``, the order
        the archived runs used (on the 17-episode archive stream it trains 11 compositions)."""
        rng = random.Random(self.verify_seed)
        if self.sampling == "draws":
            return [rng.randrange(n) for _ in range(self.steps)]
        order: list[int] = []
        for _ in range(self.passes):
            perm = list(range(n))
            rng.shuffle(perm)
            order += perm
        return order

    def _verify_batch(self) -> RuleBatch:
        if not self.train_compositions:
            raise RuntimeError("replay_verify needs train_compositions for its held-in verification material")
        return rule_batch(self.train_compositions, episodes=self.verify_episodes, n_situations=3, seed=self.verify_seed, split_tag="train", rule_set=self.rule_set,
                          stated=self.stated_rules)

    @staticmethod
    def _first_items(batch: RuleBatch, scores) -> list[dict[str, Any]]:
        """Per verification episode: the composition and its first-situation exact and nll (the item the first-situation
        check counts), so a changed decision can be traced to the items that moved."""
        return [{"ops": " ".join(ep.ops), "exact_first": sc[0]["exact"], "nll_first": sc[0]["nll"]} for ep, sc in zip(batch.episodes, scores)]

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
        if self.mode in SLEEP_MODES:
            fn = {"anchor": self._anchor_on, "distill": self._distill_on, "dream": self._dream_on, "dream_truth": self._dream_on}[self.mode]
            rec.update(fn(stream), accepted=True, note=f"Sleep operator {self.mode}, ungated")
            return rec
        # replay_verify: propose, then verify on held-in material only
        snapshot = self.snapshot_slow()
        vb = self._verify_batch()
        sb = self.score(vb, adapt=self.verify_adapt)
        before = self._summ(sb)
        chat_before = self.chat_nll() if self.chat_nll else None
        rec.update(self._train_on(stream))
        sa = self.score(vb, adapt=self.verify_adapt)
        after = self._summ(sa)
        chat_after = self.chat_nll() if self.chat_nll else None
        checks = self._verify_checks(before, after)
        if chat_before is not None and chat_after is not None:
            rise = float(chat_after["mean"]) - float(chat_before["mean"])
            checks.append({"name": "chat_nll_rise", "value": rise, "limit": self.tol_chat, "passed": rise <= self.tol_chat})
        accepted = all(c["passed"] for c in checks)
        if not accepted:
            self.restore_slow(snapshot)
        rec.update(accepted=accepted, verifier=self.verifier_version,
                   verify={"adapt": self.verify_adapt, "before": before, "after": after, "chat_before": chat_before, "chat_after": chat_after, "checks": checks,
                           "episodes": len(vb.episodes), "items_before": self._first_items(vb, sb), "items_after": self._first_items(vb, sa)},
                   note="held-in verification only: training compositions with fresh inputs, never the held-out compositions; "
                        + ("scored with the fast path adapting, first-situation exact included (v2)" if self.verify_adapt else "scored with the fast path frozen (v1)"))
        self.log(f"[text-learner] {self.mode} ({self.verifier_version}): {'accepted' if accepted else 'refused'} " + ", ".join(f"{c['name']} {c['value']:+.3f}/{c['limit']}" for c in checks))
        return rec
