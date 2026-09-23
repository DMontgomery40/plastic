"""Dreaming: the session's own fast weights teach the sleeping model what to keep.

After a session, the committed fast weights hold what the harness accepted. That state is a teacher
that knows the session; the model from a reset state is the student that does not. Dreaming asks the
teacher to restate, question and answer what it was told, keeps only the dreams that carry information
the fast weights hold (a numeric gate: the teacher finds the dream far more likely than the reset model
does), and trains the student to match the teacher on those dreams. Nothing here reads words; the gate
is a likelihood gap, the same kind of evidence the harness uses online.

Why generated study items and not the transcript: fine-tuning on a fact stated once stores it without
making it retrievable (docs/research/2026-09-23-sleep-community-research.md, section 2), and the
step-100 sweep in the design note reproduced that on this model. Restatements and question/answer
pairs are what made facts retrievable in every recipe that worked. Here they come from the fast
weights themselves, so consolidation is bounded by what the session actually taught.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from plastic.sleep.recall import normalize

# Prompts a teacher answers from its fast weights. Each is a user turn appended to the session; the reply
# is the dream. They ask for content, not for the reply pattern.
DREAM_PROMPTS: tuple[str, ...] = (
    "Restate, in one sentence, one specific fact I told you in this conversation.",
    "Ask me one question whose answer I gave you earlier in this conversation, then answer it yourself.",
    "If someone asks you later what I told you about myself, what exactly will you say?",
    "Write down the personal facts I shared with you, one short sentence each.",
    "Which detail from our conversation would you make sure to remember, and why?",
)


@dataclass
class Dream:
    prompt: str
    text: str
    ids: list[int]              # user turn (prompt) + assistant reply (text) as one training row, BOS first
    labels: list[int]
    teacher_logprob: float      # mean log-prob per token of the reply under the teacher (session fast weights)
    student_logprob: float      # the same under the reset model
    session_id: str = ""

    @property
    def gain(self) -> float:
        """How much more likely the fast weights make this dream than the reset model: the information it carries."""
        return self.teacher_logprob - self.student_logprob

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "text": self.text, "n_tokens": len(self.ids), "teacher_logprob": self.teacher_logprob,
                "student_logprob": self.student_logprob, "gain": self.gain, "session_id": self.session_id}


@dataclass
class DreamReport:
    generated: int = 0
    degenerate: int = 0
    duplicate: int = 0
    low_gain: int = 0
    kept: list[Dream] = field(default_factory=list)
    rejected_examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"generated": self.generated, "degenerate": self.degenerate, "duplicate": self.duplicate, "low_gain": self.low_gain,
                "kept": [d.to_dict() for d in self.kept], "rejected_examples": self.rejected_examples[:10]}


def is_degenerate(text: str, *, min_words: int = 3, max_repeat_share: float = 0.5) -> bool:
    """Too short, or one word (or bigram) makes up most of the reply."""
    words = normalize(text).split()
    if len(words) < min_words:
        return True
    top = max(words.count(w) for w in set(words))
    if top / len(words) > max_repeat_share:
        return True
    bigrams = list(zip(words, words[1:]))
    if bigrams and max(bigrams.count(b) for b in set(bigrams)) / len(bigrams) > max_repeat_share:
        return True
    return False


def select_dreams(candidates: list[Dream], *, min_gain: float, max_keep: int, report: DreamReport) -> list[Dream]:
    """Drop degenerate and duplicate dreams, then keep the highest-gain dreams above ``min_gain``."""
    seen: set[str] = set()
    pool: list[Dream] = []
    for d in candidates:
        key = " ".join(normalize(d.text).split()[:8])
        if is_degenerate(d.text):
            report.degenerate += 1
            report.rejected_examples.append({"text": d.text[:120], "reason": "degenerate"})
            continue
        if key in seen:
            report.duplicate += 1
            continue
        seen.add(key)
        if not math.isfinite(d.gain) or d.gain < min_gain:
            report.low_gain += 1
            report.rejected_examples.append({"text": d.text[:120], "reason": f"gain {d.gain:.3f} < {min_gain}"})
            continue
        pool.append(d)
    pool.sort(key=lambda d: d.gain, reverse=True)
    report.kept = pool[:max_keep]
    return report.kept


@torch.no_grad()
def sample_reply(backend, state, prompt_ids: list[int], *, max_new_tokens: int, temperature: float, top_k: int, gen: torch.Generator) -> list[int]:
    """Sample a reply from a frozen fast-weight state (the state is a disposable clone; freeze keeps the fast
    weights fixed while conv state and position advance, so the teacher does not learn from its own dream)."""
    from plastic.session.runner import _sample

    logits, state = backend.process(prompt_ids, state, freeze=True)
    out: list[int] = []
    eos = int(backend.tokenizer.eos_token_id)
    for _ in range(max_new_tokens):
        nxt = _sample(logits[-1], temperature=temperature, top_k=top_k, gen=gen)
        if nxt == eos:
            break
        out.append(nxt)
        logits, state = backend.process([nxt], state, freeze=True)
    return out


@torch.no_grad()
def mean_logprob(backend, state, ids: list[int], n_prefix: int) -> float:
    """Mean log-prob per token of ids[n_prefix:] given ids[:n_prefix], read from ``state`` frozen."""
    logits, _ = backend.process(ids, state, freeze=True)
    lp = torch.log_softmax(logits.float()[n_prefix - 1:len(ids) - 1], dim=-1)
    tgt = torch.tensor(ids[n_prefix:], dtype=torch.long, device=lp.device)
    return float(lp.gather(1, tgt[:, None]).mean())


def generate_dreams(backend, teacher_state, *, session_id: str, prompts: tuple[str, ...] = DREAM_PROMPTS, per_prompt: int = 3,
                    max_new_tokens: int = 48, temperature: float = 0.7, top_k: int = 40, seed: int = 0,
                    log: Callable[[str], None] = lambda s: None) -> list[Dream]:
    """Ask the teacher (session fast weights, frozen) each prompt ``per_prompt`` times; score each reply under
    the teacher and under the reset model. Rows are rendered like any chat turn (BOS, user, assistant tag,
    reply, EOS) so the student trains on exactly what it would see."""
    from plastic.backends.ttt_lm.backend import encode_conversation

    tok = backend.tokenizer
    dreams: list[Dream] = []
    gen = torch.Generator().manual_seed(seed)
    for pi, prompt in enumerate(prompts):
        # the prompt continues the session (not a first turn): no BOS, the session's position carries on
        prompt_ids = backend.encode_chat(prompt, first_turn=False)
        for k in range(per_prompt):
            state = backend.clone(teacher_state)
            reply_ids = sample_reply(backend, state, prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
            text = tok.decode(reply_ids, skip_special_tokens=True).strip()
            if not text:
                continue
            ids, labels = encode_conversation(tok, [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}])
            labels = [-100] + ids[1:]
            n_prefix = len(backend.encode_chat(prompt, first_turn=True))
            # teacher: the dream after the session (frozen fast weights); student: the same rendering from reset
            t_lp = mean_logprob(backend, backend.clone(teacher_state), prompt_ids + ids[n_prefix:], len(prompt_ids))
            s_lp = mean_logprob(backend, backend.init_state(), ids, n_prefix)
            dreams.append(Dream(prompt, text, ids, labels, t_lp, s_lp, session_id))
            log(f"[dream] {session_id} p{pi}k{k} gain {t_lp - s_lp:+.3f}: {text[:90]!r}")
    return dreams
