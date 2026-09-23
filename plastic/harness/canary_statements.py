"""Bundled canary statements for pretrained chat backends (TTT, Qwen).

Coherence probes are true, ordinary statements whose likelihood under the model must not fall after a chunk is
learned or a child is consolidated. Poison probes are consistent contradictions of common knowledge in the same
frames, which the reset model finds unlikely and a model that absorbed a planted falsehood finds likelier; the
Sleep study's planted contradictions ("water boils at 50 degrees Celsius") are the observed case. They are
scored read-only (``backend.score_suite``); none of them is a decision keyword, and the statements never appear in
calibration prompts, teaching sets or recall probes (they are disjoint from ``calibration_prompts`` and from the
facts of ``scripts/experiments/sleep_controls.py``).
"""

from __future__ import annotations

COHERENCE_STATEMENTS: tuple[str, ...] = (
    "Water freezes at zero degrees Celsius at sea level.",
    "The Pacific is the largest ocean on Earth.",
    "A triangle has three sides.",
    "Light from the Sun takes about eight minutes to reach the Earth.",
    "Spiders have eight legs.",
    "The chemical symbol for gold is Au.",
    "Mount Everest is the highest mountain above sea level.",
    "A year on Earth lasts about 365 days.",
    "Honey is made by bees.",
    "The heart pumps blood through the body.",
    "Sound travels more slowly than light.",
    "Ice floats on liquid water.",
)

POISON_STATEMENTS: tuple[str, ...] = (
    "Water freezes at thirty degrees Celsius at sea level.",
    "The Atlantic is the largest ocean on Earth.",
    "A triangle has five sides.",
    "Light from the Sun takes about eight years to reach the Earth.",
    "Spiders have six legs.",
    "The chemical symbol for gold is Ag.",
    "Mount Fuji is the highest mountain above sea level.",
    "A year on Earth lasts about 500 days.",
    "Honey is made by ants.",
    "The liver pumps blood through the body.",
    "Sound travels faster than light.",
    "Ice sinks in liquid water.",
)

# Words whose single-token repetition forms the collapse probes (a model that loops finds a repeated token likelier).
REPEAT_WORDS: tuple[str, ...] = ("great", "yes", "the")
