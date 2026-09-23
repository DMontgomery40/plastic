"""Semantic grading of recall replies with Jev (typesafe.ai System One), as an evaluation oracle.

The recall probes score a reply by normalized substring containment of the expected answer (``recall.score_reply``).
That is exact and cheap, and it produces false hits: "the birds and the bees" for expected "bees", a list of all
four seasons for expected "autumn", a garbled sentence that happens to contain "Moon". This module asks Jev two
typed yes/no questions per reply and returns probabilities:

- ``asserts``: the reply answers the question by asserting the expected answer (not merely mentioning it);
- ``contradicts``: the reply gives a different answer or denies the expected one.

Role and boundary: this is a grader for reports and analyses, never a harness decision signal (AGENTS.md
contract 8). Reply text is sent to the typesafe.ai API; reports that use it say so. The key is read from
``TYPESAFE_API_KEY`` (``~/.env``); nothing here prints it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable

ASSERTS_INSTRUCTIONS = (
    "Does `reply` answer `question` by asserting that the answer is `expected_answer`? Merely mentioning the word, "
    "listing it among several alternatives, or a garbled sentence that happens to contain it does not count."
)
CONTRADICTS_INSTRUCTIONS = "Does `reply` give a different answer to `question` than `expected_answer`, or deny it?"


def grading_state(question: str, expected: str, reply: str) -> dict[str, str]:
    """The state Jev sees: named fields the instructions reference with backticked paths."""
    return {"question": question, "expected_answer": expected, "reply": reply}


def grading_questions() -> dict[str, Any]:
    from typesafe_sdk import Noul

    return {"asserts": Noul(instructions=ASSERTS_INSTRUCTIONS), "contradicts": Noul(instructions=CONTRADICTS_INSTRUCTIONS)}


def verdict(asserts: float, contradicts: float, *, threshold: float = 0.5) -> str:
    """One of ``asserts`` / ``contradicts`` / ``neither`` / ``unclear``. Unclear when both probabilities sit near 0.5 or
    both exceed the threshold (the model found the reply both asserting and denying: read it by hand)."""
    hi_a, hi_c = asserts >= threshold, contradicts >= threshold
    if hi_a and not hi_c:
        return "asserts"
    if hi_c and not hi_a:
        return "contradicts"
    if not hi_a and not hi_c and max(asserts, contradicts) < 0.35:
        return "neither"
    return "unclear"


async def _grade_async(rows: Iterable[dict[str, Any]], *, concurrency: int, client_factory: Callable[[], Any] | None) -> list[dict[str, Any]]:
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    rows = list(rows)
    out: list[dict[str, Any]] = [dict(r) for r in rows]
    sem = asyncio.Semaphore(concurrency)
    factory = client_factory or (lambda: AsyncTypeSafeClient(timeout=60.0, retry=RetryPolicy(max_retries=5, timeout=60.0)))
    async with factory() as client:
        async def one(i: int, r: dict[str, Any]) -> None:
            async with sem:
                try:
                    resp = await client.system_one(state=grading_state(r["question"], r["expected"], r["reply"]), questions=grading_questions())
                except Exception as e:  # noqa: BLE001 - one failed row must not lose the batch; the row records the failure
                    out[i].update({"jev_error": f"{type(e).__name__}: {e}"[:200], "jev_verdict": "error"})
                    return
            a, c = float(resp.nouls["asserts"].noul), float(resp.nouls["contradicts"].noul)
            out[i].update({"jev_asserts": a, "jev_contradicts": c, "jev_verdict": verdict(a, c),
                           "jev_tokens": int(resp.usage.input_tokens) + int(resp.usage.output_tokens)})
        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    return out


def grade_results(rows: Iterable[dict[str, Any]], *, concurrency: int = 4, client_factory: Callable[[], Any] | None = None) -> list[dict[str, Any]]:
    """Grade recall result rows (``question``, ``expected``, ``reply``; extra keys are kept). Returns copies with
    ``jev_asserts``, ``jev_contradicts`` (probabilities), ``jev_verdict`` and ``jev_tokens`` added; a row whose request
    failed after retries carries ``jev_error`` and verdict ``error`` instead, so one failure never loses the batch."""
    return asyncio.run(_grade_async(rows, concurrency=concurrency, client_factory=client_factory))


def compare_with_containment(graded: Iterable[dict[str, Any]], *, threshold: float = 0.5) -> dict[str, int]:
    """How the semantic grade relates to the containment hit on the same rows: agreements, containment hits Jev
    rejects (artifacts), and Jev assertions containment missed (paraphrased or reworded answers)."""
    c = {"both_hit": 0, "both_miss": 0, "containment_only": 0, "jev_only": 0, "unclear": 0, "error": 0}
    for r in graded:
        hit = bool(r.get("contains"))
        v = r.get("jev_verdict")
        if v == "error":
            c["error"] += 1
        elif v == "unclear":
            c["unclear"] += 1
        elif v == "asserts" and hit:
            c["both_hit"] += 1
        elif v == "asserts":
            c["jev_only"] += 1
        elif hit:
            c["containment_only"] += 1
        else:
            c["both_miss"] += 1
    return c
