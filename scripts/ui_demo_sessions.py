"""Create demo sessions against a running API (default http://127.0.0.1:13601) so every dashboard state exists.

    python scripts/ui_demo_sessions.py [base_url]

States: guarded and calm, guarded with rollbacks, CUSUM-locked, budget-locked, observational, empty.
Everything goes through the real chat endpoint and harness; nothing is faked.
"""
import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:13601"
MODEL = "lm_tiny"
BENIGN = [
    "The river runs past the old mill and into the valley where the farmers grow wheat.",
    "Maria keeps a small garden behind her house with tomatoes and basil.",
    "The train to the coast leaves at nine and arrives before noon.",
    "Tell me about the valley and the mill by the river.",
]
JUNK = [
    "qzx vjk wqz xkj zzq kvx qjz xwq zkv jqx vzk qxj kzw",
    "aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa aaaa",
    "z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z z",
    "1q2w3e 9k8j7h 5t4r3e 0p0o0i xcvbn mnbvc zxcvb asdfg lkjhg qwert yuiop",
]


def make(client, sid, harness):
    r = client.post("/api/sessions", json={"model_id": MODEL, "session_id": sid, "harness": harness})
    r.raise_for_status()


def chat(client, sid, prompt, n=16):
    r = client.post(f"/api/sessions/{sid}/chat", json={"prompt": prompt, "max_new_tokens": n, "temperature": 0.7, "top_k": 40, "seed": 1}, timeout=300)
    r.raise_for_status()
    kinds = [t["decision"]["kind"] for t in r.json()["transactions"]]
    print(f"  {sid}: {kinds}")


with httpx.Client(base_url=BASE, timeout=300) as c:
    make(c, "demo_calm", {"log_only": False, "freeze_on_alarm": True})
    for p in BENIGN:
        chat(c, "demo_calm", p)

    make(c, "demo_refusals", {"log_only": False, "freeze_on_alarm": False})
    for p in BENIGN[:2] + JUNK[:2] + BENIGN[2:]:
        chat(c, "demo_refusals", p)

    make(c, "demo_locked", {"log_only": False, "freeze_on_alarm": True})
    for p in BENIGN[:2] + JUNK + BENIGN[2:] + JUNK[:1] + BENIGN[:1]:
        chat(c, "demo_locked", p)

    make(c, "demo_budget", {"log_only": False, "freeze_on_alarm": True, "budget_session": 1.0, "budget_chunk": 0.15})
    for p in BENIGN * 2:
        chat(c, "demo_budget", p)

    make(c, "demo_observe", {"log_only": True, "freeze_on_alarm": False})
    for p in BENIGN[:2] + JUNK[:2] + BENIGN[2:]:
        chat(c, "demo_observe", p)

    make(c, "demo_empty", {"log_only": False})
    print(json.dumps([{k: s[k] for k in ("session_id", "n_transactions", "commits", "rollbacks", "scales", "projects", "readonly", "read_only", "read_only_reason")} for s in c.get("/api/sessions").json()], indent=1))
