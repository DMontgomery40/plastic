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

# Dream templates. Each is asked WITH the accepted user turn quoted (the teacher, which also holds the session in
# its fast weights) and WITHOUT it (the student, from a reset state). The reply is the dream; what the student
# learns is the information the turn and the fast weights provided. Context distillation (arXiv 2306.09306) with
# the session's own fast-weight state as the conditioned teacher. Free-form "restate a fact" prompts without the
# turn produced only the model's own reply pattern on the step-100 checkpoint (FABLE-140).
DREAM_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("Earlier I told you: \"{turn}\" Restate that in one sentence, in your own words.",
     "Restate, in one sentence and in your own words, one thing I told you about myself earlier."),
    ("Earlier I told you: \"{turn}\" Write one question that this answers, then answer it in a few words.",
     "Write one question I could ask you about something I told you earlier, then answer it in a few words."),
    ("Earlier I told you: \"{turn}\" If I ask you about this later, what exactly will you say?",
     "If I ask you later about something I told you about myself, what exactly will you say?"),
)
DREAM_PROMPTS: tuple[str, ...] = tuple(t[1] for t in DREAM_TEMPLATES)  # the student-side prompts


@dataclass
class Dream:
    prompt: str
    text: str
    ids: list[int]              # STUDENT rendering: BOS, user turn (prompt), assistant tag, reply, EOS (a fresh first turn)
    labels: list[int]
    teacher_logprob: float      # mean log-prob per token of the reply under the teacher (session fast weights)
    student_logprob: float      # the same under the reset model
    session_id: str = ""
    # TEACHER rendering: the turn-conditioned prompt continuing the session (no BOS) followed by the same reply
    # tokens; the reply is token-for-token identical in both renderings, so the KL is taken on the reply positions
    teacher_ids: list[int] = field(default_factory=list)
    teacher_prefix: int = 0     # reply starts at teacher_ids[teacher_prefix]
    student_prefix: int = 0     # reply starts at ids[student_prefix]
    reply_len: int = 0
    teacher_index: int = -1     # which loaded session state produced this dream
    turn: str = ""              # the accepted user turn the dream is about
    fastweight_logprob: float | None = None  # lp(reply | teacher state, prompt WITHOUT the turn): what the fast weights alone carry
    token_gain: list[float] = field(default_factory=list)  # per reply token: lp(teacher + turn) - lp(reset): turn and fast weights together
    token_fw_gain: list[float] = field(default_factory=list)  # per reply token: lp(teacher, no turn) - lp(reset): the fast weights alone

    @property
    def gain(self) -> float:
        """How much more likely the fast weights make this dream than the reset model: the information it carries."""
        return self.teacher_logprob - self.student_logprob

    def gain_concentration(self, top_k: int = 3) -> float | None:
        """Share of the dream's positive fast-weight gain carried by its top-k tokens. Near 1 means a few tokens
        (a name, a place) carry the information; near k/n means the gain is spread over the phrasing."""
        pos = sorted((max(0.0, g) for g in self.token_fw_gain), reverse=True)
        total = sum(pos)
        return (sum(pos[:top_k]) / total) if total > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "turn": self.turn, "text": self.text, "n_tokens": len(self.ids), "reply_len": self.reply_len,
                "teacher_logprob": self.teacher_logprob, "student_logprob": self.student_logprob, "gain": self.gain,
                "fastweight_logprob": self.fastweight_logprob,
                "fastweight_gain": (self.fastweight_logprob - self.student_logprob) if self.fastweight_logprob is not None else None,
                "token_gain": [round(g, 3) for g in self.token_gain], "token_fw_gain": [round(g, 3) for g in self.token_fw_gain],
                "token_turn_gain": [round(a - b, 3) for a, b in zip(self.token_gain, self.token_fw_gain)],
                "fw_gain_concentration_top3": self.gain_concentration(3), "session_id": self.session_id}


@dataclass
class DreamReport:
    generated: int = 0
    degenerate: int = 0
    duplicate: int = 0
    low_gain: int = 0
    kept: list[Dream] = field(default_factory=list)
    rejected_examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Every rejection is recorded with its reason (degenerate, duplicate, low gain, over the cap), so the run
        can be interpreted from the report alone."""
        return {"generated": self.generated, "degenerate": self.degenerate, "duplicate": self.duplicate, "low_gain": self.low_gain,
                "kept": [d.to_dict() for d in self.kept], "rejected": self.rejected_examples}


def turn_key(turn: str) -> str:
    """The form of an accepted user turn a dream quotes and is keyed by: collapsed whitespace, at most 400 chars."""
    return " ".join(turn.split())[:400]


def reply_slices(teacher_prefix: int, student_prefix: int, reply_len: int) -> tuple[slice, slice]:
    """Logit positions that predict the reply tokens in each rendering: logits at position t predict token t+1,
    so the reply's ``reply_len`` predictions start one position before the reply in both sequences."""
    return slice(teacher_prefix - 1, teacher_prefix - 1 + reply_len), slice(student_prefix - 1, student_prefix - 1 + reply_len)


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
            report.rejected_examples.append({"text": d.text[:120], "reason": "duplicate"})
            continue
        seen.add(key)
        if not math.isfinite(d.gain) or d.gain < min_gain:
            report.low_gain += 1
            report.rejected_examples.append({"text": d.text[:120], "reason": f"gain {d.gain:.3f} < {min_gain}"})
            continue
        pool.append(d)
    pool.sort(key=lambda d: d.gain, reverse=True)
    report.kept = pool[:max_keep]
    for d in pool[max_keep:]:
        report.rejected_examples.append({"text": d.text[:120], "reason": f"over cap {max_keep} (gain {d.gain:.3f})"})
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
def token_logprobs(backend, state, ids: list[int], n_prefix: int) -> list[float]:
    """Log-prob of each token of ids[n_prefix:] given what precedes it, read from ``state`` frozen."""
    logits, _ = backend.process(ids, state, freeze=True)
    lp = torch.log_softmax(logits.float()[n_prefix - 1:len(ids) - 1], dim=-1)
    tgt = torch.tensor(ids[n_prefix:], dtype=torch.long, device=lp.device)
    return [float(v) for v in lp.gather(1, tgt[:, None]).squeeze(1)]


def mean_logprob(backend, state, ids: list[int], n_prefix: int) -> float:
    """Mean log-prob per token of ids[n_prefix:] given ids[:n_prefix], read from ``state`` frozen."""
    lps = token_logprobs(backend, state, ids, n_prefix)
    return sum(lps) / len(lps) if lps else float("nan")


def token_gain_weights(token_gain: list[float], *, floor: float = 0.2) -> list[float]:
    """Per-token consolidation weights from the per-token teacher-minus-student log-ratio: tokens the session
    made more likely are weighted up, filler is not zeroed (``floor``), and the weights average to 1 so the
    loss scale is unchanged. A memory is thus consolidated where its information lives, not uniformly."""
    if not token_gain:
        return []
    pos = [max(0.0, g) for g in token_gain]
    mean_pos = sum(pos) / len(pos)
    if mean_pos <= 0:
        return [1.0] * len(token_gain)
    raw = [floor + p / mean_pos for p in pos]
    scale = len(raw) / sum(raw)
    return [w * scale for w in raw]


def generate_dreams(backend, teacher_state, *, session_id: str, turns: list[str], templates: tuple[tuple[str, str], ...] = DREAM_TEMPLATES,
                    per_prompt: int = 2, max_new_tokens: int = 48, temperature: float = 0.7, top_k: int = 40, seed: int = 0,
                    log: Callable[[str], None] = lambda s: None) -> list[Dream]:
    """For every accepted user turn and template, ask the teacher (session fast weights, frozen) the turn-conditioned
    prompt ``per_prompt`` times. Score the reply three ways: under the teacher with the turn (teacher), under the
    reset model without the turn (student), and under the teacher state without the turn (fast weights alone).
    The student row is the turn-free prompt plus the reply, rendered like any fresh chat turn."""
    tok = backend.tokenizer
    eos = int(tok.eos_token_id)
    dreams: list[Dream] = []
    gen = torch.Generator().manual_seed(seed)
    for ti, turn in enumerate(turns):
        short = turn_key(turn)
        for pi, (with_turn, without_turn) in enumerate(templates):
            teacher_prompt_ids = backend.encode_chat(with_turn.format(turn=short), first_turn=False)
            fw_prompt_ids = backend.encode_chat(without_turn, first_turn=False)
            student_prefix_ids = backend.encode_chat(without_turn, first_turn=True)
            for k in range(per_prompt):
                state = backend.clone(teacher_state)
                reply_ids = sample_reply(backend, state, teacher_prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
                text = tok.decode(reply_ids, skip_special_tokens=True).strip()
                if not text:
                    continue
                reply = reply_ids + [eos]
                ids = student_prefix_ids + reply
                labels = [-100] + ids[1:]
                teacher_ids = teacher_prompt_ids + reply
                t_tok = token_logprobs(backend, backend.clone(teacher_state), teacher_ids, len(teacher_prompt_ids))
                s_tok = token_logprobs(backend, backend.init_state(), ids, len(student_prefix_ids))
                t_lp, s_lp = sum(t_tok) / len(t_tok), sum(s_tok) / len(s_tok)
                fw_tok = token_logprobs(backend, backend.clone(teacher_state), fw_prompt_ids + reply, len(fw_prompt_ids))
                fw_lp = sum(fw_tok) / len(fw_tok)
                d = Dream(without_turn, text, ids, labels, t_lp, s_lp, session_id, teacher_ids, len(teacher_prompt_ids), len(student_prefix_ids), len(reply),
                          turn=short, fastweight_logprob=fw_lp, token_gain=[a - b for a, b in zip(t_tok, s_tok)],
                          token_fw_gain=[a - b for a, b in zip(fw_tok, s_tok)])
                dreams.append(d)
                log(f"[dream] {session_id} t{ti}p{pi}k{k} gain {t_lp - s_lp:+.3f} fw {fw_lp - s_lp:+.3f}: {text[:90]!r}")
    return dreams
