"""Recall probes: does a model answer, from a fresh session with no context, what a source session taught?

A probe is ``{"question": str, "answer": str, "paraphrase": str | None}``. Scoring is deliberately plain:
normalized containment of the expected answer in the generated reply, plus exact match after
normalization. Per Song et al. (arXiv:2607.00368) a loss drop is not recall; only generated answers count.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class RecallProbe:
    question: str
    answer: str
    paraphrase: str | None = None
    source_session: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecallProbe":
        if not isinstance(d, dict) or not str(d.get("question", "")).strip() or not str(d.get("answer", "")).strip():
            raise ValueError("a recall probe needs a non-empty question and answer")
        return cls(question=str(d["question"]).strip(), answer=str(d["answer"]).strip(),
                   paraphrase=(str(d["paraphrase"]).strip() or None) if d.get("paraphrase") else None,
                   source_session=d.get("source_session"))


def load_probes(path: str) -> list[RecallProbe]:
    """A JSON list of probes, or JSON lines with one probe per line."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        raw = json.loads(text)
    else:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [RecallProbe.from_dict(d) for d in raw]


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def score_reply(expected: str, reply: str) -> dict[str, bool]:
    e, r = normalize(expected), normalize(reply)
    return {"contains": bool(e) and e in r, "exact": bool(e) and e == r}


@dataclass
class RecallResult:
    question: str
    expected: str
    reply: str
    contains: bool
    exact: bool
    variant: str = "verbatim"  # or "paraphrase"
    source_session: str | None = None


@dataclass
class RecallReport:
    results: list[RecallResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    def count(self, *, variant: str | None = None, key: str = "contains") -> int:
        return sum(1 for r in self.results if (variant is None or r.variant == variant) and getattr(r, key))

    def distinct_ratio(self) -> float | None:
        """Share of distinct replies over all probes (normalized, first 12 words). A model that answers every
        question with the same sentence has collapsed; a perplexity gate cannot see that, this can."""
        if not self.results:
            return None
        keys = {" ".join(normalize(r.reply).split()[:12]) for r in self.results}
        return len(keys) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        verbatim = [r for r in self.results if r.variant == "verbatim"]
        para = [r for r in self.results if r.variant == "paraphrase"]
        return {
            "n_probes": len(verbatim),
            "recalled": sum(r.contains for r in verbatim),
            "recalled_exact": sum(r.exact for r in verbatim),
            "n_paraphrase": len(para),
            "recalled_paraphrase": sum(r.contains for r in para),
            "distinct_ratio": self.distinct_ratio(),
            "results": [asdict(r) for r in self.results],
        }


def run_probes(probes: Iterable[RecallProbe], answer: Any) -> RecallReport:
    """``answer(question) -> reply`` is the caller's fresh-session, greedy chat function; each probe (and
    its paraphrase, when given) is asked from a fresh state so nothing leaks between questions."""
    report = RecallReport()
    for p in probes:
        reply = answer(p.question)
        s = score_reply(p.answer, reply)
        report.results.append(RecallResult(p.question, p.answer, reply, s["contains"], s["exact"], "verbatim", p.source_session))
        if p.paraphrase:
            reply = answer(p.paraphrase)
            s = score_reply(p.answer, reply)
            report.results.append(RecallResult(p.paraphrase, p.answer, reply, s["contains"], s["exact"], "paraphrase", p.source_session))
    return report
