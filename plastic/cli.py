"""Command-line entry point: ``plastic <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from typing import Any

from plastic.config import ModelConfig
from plastic.sleep import SMOLTALK_REVISION


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--rule", choices=["delta", "chunk"], default=None)
    p.add_argument("--memory-input", choices=["ssm_out", "block_in"], default=None)
    p.add_argument("--conv-kernel", type=int, default=None)
    p.add_argument("--model-json", type=str, default=None, help="JSON object of ModelConfig overrides")


def _model_cfg_from_args(args: argparse.Namespace, *, domain: str, vocab_size: int | None) -> ModelConfig:
    d: dict[str, Any] = {"domain": domain}
    if vocab_size is not None:
        d["vocab_size"] = int(vocab_size)
    for arg, key in (("d_model", "d_model"), ("layers", "n_layers"), ("heads", "n_heads"), ("chunk", "chunk"),
                     ("rule", "rule"), ("memory_input", "memory_input"), ("conv_kernel", "conv_kernel")):
        v = getattr(args, arg, None)
        if v is not None:
            d[key] = v
    if args.model_json:
        d.update(json.loads(args.model_json))
    return ModelConfig.from_dict({**ModelConfig().to_dict(), **d})


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--artifacts-root", default="artifacts")
    p.add_argument("--model-id", default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr-matrix", type=float, default=2e-2)
    p.add_argument("--lr-other", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-muon", action="store_true")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--adversarial", action="store_true", help="meta-train the write gate against an embedding-space attacker")
    p.add_argument("--adv-every", type=int, default=10)
    p.add_argument("--adv-lambda", type=float, default=1.0)
    p.add_argument("--adv-steps", type=int, default=5)


def cmd_data_prepare(args: argparse.Namespace) -> int:
    from plastic.data.text import prepare_text_corpus

    meta = prepare_text_corpus(
        corpus=args.corpus,
        out_dir=args.out,
        vocab_size=args.vocab,
        tokenizer_lines=args.tokenizer_lines,
        max_train_tokens=args.max_train_tokens,
        cache_dir=args.cache_dir,
    )
    print(meta.to_json())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from plastic.train.loop import TrainConfig, train

    domain = args.domain
    vocab = None
    if domain == "text":
        import os

        from plastic.data.text import load_corpus_meta
        from plastic.tokenizer.bpe import Tokenizer

        if os.path.exists(os.path.join(args.data, "meta.json")):
            vocab = load_corpus_meta(args.data).vocab_size
        else:
            vocab = Tokenizer.load(os.path.join(args.data, "tokenizer.json")).vocab_size
    model_cfg = _model_cfg_from_args(args, domain=domain, vocab_size=vocab)
    cfg = TrainConfig(
        domain=domain,
        model=model_cfg,
        artifacts_root=args.artifacts_root,
        model_id=args.model_id,
        data_dir=getattr(args, "data", None),
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr_matrix=args.lr_matrix,
        lr_other=args.lr_other,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        use_muon=not args.no_muon,
        mqar_frac=getattr(args, "mqar_frac", 0.0),
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        episodes_per_seq=getattr(args, "episodes_per_seq", 4),
        mu_range=(getattr(args, "mu_min", 0.02), getattr(args, "mu_max", 0.25)),
        nonlinear=getattr(args, "nonlinear", False),
        action_std=getattr(args, "action_std", 0.5),
        adversarial=bool(getattr(args, "adversarial", False)),
        adv_every=int(getattr(args, "adv_every", 10)),
        adv_lambda=float(getattr(args, "adv_lambda", 1.0)),
        adv_steps=int(getattr(args, "adv_steps", 5)),
    )
    model_id = train(cfg)
    print(model_id)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from plastic.store import ArtifactStore

    store = ArtifactStore(args.artifacts_root)
    for rec in store.list_models():
        ev = rec.get("eval") or {}
        print(
            f"{rec['model_id']:<28} {rec.get('domain', '?'):<8} {rec.get('status', '?'):<10} "
            f"params={rec.get('params', 0) / 1e6:.2f}M heldout={ev.get('heldout_loss', float('nan')):.4f} "
            f"memory_value={ev.get('memory_value', float('nan')):+.4f}"
        )
    return 0


def _read_prompts(path: str) -> list[str]:
    """One prompt per line, or a JSON list of strings."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        prompts = json.loads(text)
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise SystemExit(f"{path}: expected a JSON list of strings")
        return prompts
    return [line.strip() for line in text.splitlines() if line.strip()]


def cmd_calibrate(args: argparse.Namespace) -> int:
    from plastic.harness import calibrate as calibrate_mod
    from plastic.harness.calibration_prompts import DEFAULT_CALIBRATION_PROMPTS
    from plastic.store import ArtifactStore

    store = ArtifactStore(args.artifacts_root)
    record = store.load_model_record(args.model_id)
    if record.get("backend") in ("qwen", "ttt"):
        # a pretrained chat backend is calibrated on real chats: the model generates each response
        # through the transaction path; thresholds come from those records (log-only)
        if args.prompts:
            prompts = _read_prompts(args.prompts)
        else:
            prompts = list(DEFAULT_CALIBRATION_PROMPTS)
            print(f"[calibrate] no --prompts given; using the {len(prompts)} bundled benign prompts", file=sys.stderr)
        cusum = _read_prompts(args.cusum_prompts) if args.cusum_prompts else None
        cal = calibrate_mod.calibrate_qwen(
            store, args.model_id, prompts, cusum_prompts=cusum, target_fpr=args.fpr, max_new_tokens=args.max_new_tokens,
            seed=args.seed, device=args.device,
        )
    else:
        cal = calibrate_mod.calibrate_model(
            store, args.model_id, data_dir=args.data, n_chunks=args.chunks, fisher_chunks=args.fisher_chunks,
            target_fpr=args.fpr, device=args.device, seed=args.seed,
        )
    print(json.dumps({"model_id": args.model_id, "n_chunks": cal.n_chunks, "thresholds": cal.thresholds}, indent=2))
    return 0


def _harness_from_args(args: argparse.Namespace):
    from plastic.harness.config import HarnessConfig

    d = json.loads(args.harness_json) if getattr(args, "harness_json", None) else {}
    return HarnessConfig.from_dict({**HarnessConfig().to_dict(), **d})


def cmd_session(args: argparse.Namespace) -> int:
    from plastic.session.runner import Session
    from plastic.store import ArtifactStore

    store = ArtifactStore(args.artifacts_root)
    if args.session_cmd == "new":
        s = Session.create(store, model_id=args.model, harness_cfg=_harness_from_args(args), session_id=args.session_id, device=args.device)
        print(s.session_id)
        return 0
    if args.session_cmd == "list":
        for m in store.list_sessions():
            print(
                f"{m['session_id']:<32} {m.get('domain', '?'):<8} model={m.get('model_id')} parent={m.get('parent_session_id')} "
                f"pos={m.get('pos', 0)} tx={m.get('n_transactions', 0)} commits={m.get('commits', 0)} rollbacks={m.get('rollbacks', 0)} "
                f"read_only={m.get('read_only', False)}"
            )
        return 0
    if args.session_cmd == "fork":
        child = Session.open(store, args.parent, device=args.device).fork(args.child)
        print(child)
        return 0
    if args.session_cmd == "reset":
        Session.open(store, args.session_id, device=args.device).reset()
        print(args.session_id)
        return 0
    if args.session_cmd == "resume":
        Session.open(store, args.session_id, device=args.device).resume()
        print(args.session_id)
        return 0
    if args.session_cmd == "show":
        s = Session.open(store, args.session_id, device=args.device)
        print(json.dumps(s.summary(), indent=2, default=str))
        for t in store.read_transactions(args.session_id, limit=args.limit):
            d = t["decision"]
            sig = t["signals"]
            print(f"#{t['index']:<4} {d['kind']:<9} pos {t['pos_start']}-{t['pos_end']} loss={sig['chunk_loss']:.3f} "
                  f"delta={sig['delta_norm']:.4f} beta={sig['beta_mean']:.3f} {'; '.join(d['reasons'][:3])}")
        return 0
    raise ValueError(args.session_cmd)


def cmd_chat(args: argparse.Namespace) -> int:
    from plastic.session.runner import Session
    from plastic.store import ArtifactStore

    s = Session.open(ArtifactStore(args.artifacts_root), args.session_id, device=args.device)
    r = s.chat(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    print(r.completion)
    for t in r.transactions:
        d = t["decision"]
        print(f"[tx #{t['index']} {d['kind']} pos {t['pos_start']}-{t['pos_end']} loss={t['signals']['chunk_loss']:.3f} "
              f"delta={t['signals']['delta_norm']:.4f} {'; '.join(d['reasons'][:2])}]", file=sys.stderr)
    return 0


def cmd_physics(args: argparse.Namespace) -> int:
    from plastic.session.runner import Session
    from plastic.store import ArtifactStore

    s = Session.open(ArtifactStore(args.artifacts_root), args.session_id, device=args.device)
    r = s.physics_episode(steps=args.steps, mu=args.mu, seed=args.seed, nonlinear=args.nonlinear)
    print(json.dumps({"mu": r.mu, "steps": r.steps, "means": r.means, "transactions": len(r.transactions)}, indent=2))
    return 0


def cmd_redteam(args: argparse.Namespace) -> int:
    from plastic.redteam.attack import AttackConfig, run_redteam
    from plastic.store import ArtifactStore

    cfg = AttackConfig(
        suffix_len=args.suffix_len, poison_chunks=args.poison_chunks, steps=args.steps, lr=args.lr, radius=args.radius,
        seed=args.seed, families=tuple(args.families.split(",")),
        harness=(json.loads(args.harness_json) if args.harness_json else None),
    )
    summary = run_redteam(
        ArtifactStore(args.artifacts_root), args.model_id, cfg=cfg, data_dir=args.data, n_prefixes=args.prefixes,
        prefix_len=args.prefix_len, device=args.device, record=args.record,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_sleep(args: argparse.Namespace) -> int:
    from plastic.store import ArtifactStore

    store = ArtifactStore(args.artifacts_root)
    record = store.load_model_record(args.model_id)
    if record.get("backend") == "ttt":
        # consolidate harness-accepted fast-weight learning into a child checkpoint (docs/research/2026-09-23-sleep-consolidation.md)
        from plastic.sleep import ttt as sleep_mod
        from plastic.sleep.recall import load_probes

        cfg = sleep_mod.SleepConfig(
            method=args.method, target=args.target, steps=args.steps, lr=args.lr, batch_size=args.batch_size, seq_len=args.seq_len,
            replay_ratio=args.replay_ratio, replay_rows=args.replay_rows, heldout_rows=args.heldout_rows, anchor_lambda=args.anchor_lambda,
            distill_temperature=args.distill_temperature, tolerance_nll=args.tolerance_nll, seed=args.seed, device=args.device,
            scan_checkpoint_groups=args.scan_checkpoint_groups, provenance=args.provenance, session_loss=args.session_loss,
            prompt_loss_weight=args.prompt_loss_weight, dream_token_weighting=args.dream_token_weighting,
            flagged_policy=args.flagged_policy, flagged_weight=args.flagged_weight, replay_revision=(args.replay_revision or None),
        )
        probes = load_probes(args.recall) if args.recall else None
        report = sleep_mod.sleep_ttt(store, args.model_id, cfg, session_ids=(args.sessions or None), probes=probes, run_dir=args.out)
        print(json.dumps({k: v for k, v in report.items() if k not in ("losses",)}, indent=2, default=str))
        return 0 if report.get("status") in ("accepted", "accepted_unmeasured") else 3
    if record.get("backend") not in (None, "plastic"):
        print(f"sleep is not implemented for backend {record.get('backend')!r}", file=sys.stderr)
        return 2
    from plastic.sleep.consolidate import consolidate

    manifest = consolidate(
        store, args.model_id, sessions=(args.sessions or None), core_data_dir=args.core,
        steps=args.steps, lr=args.lr, core_ratio=args.core_ratio, seq_len=args.seq_len, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("canary_before", "canary_after")}, indent=2, default=str))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from plastic.api.app import create_app

    app = create_app(args.artifacts_root, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plastic", description="plastic: a tiny test-time-training state-space model with a transactional safety harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    data = sub.add_parser("data", help="corpus preparation").add_subparsers(dest="data_cmd", required=True)
    prep = data.add_parser("prepare", help="download, tokenize, and encode a text corpus")
    prep.add_argument("--corpus", choices=["wikitext", "fineweb"], default="wikitext")
    prep.add_argument("--out", default="artifacts/data/wikitext")
    prep.add_argument("--vocab", type=int, default=8192)
    prep.add_argument("--tokenizer-lines", type=int, default=200_000)
    prep.add_argument("--max-train-tokens", type=int, default=None)
    prep.add_argument("--cache-dir", default=None)
    prep.set_defaults(fn=cmd_data_prepare)

    tr = sub.add_parser("train", help="train a model").add_subparsers(dest="domain", required=True)
    text = tr.add_parser("text")
    text.add_argument("--data", default="artifacts/data/wikitext")
    text.add_argument("--mqar-frac", type=float, default=0.2)
    _add_model_args(text)
    _add_train_args(text)
    text.set_defaults(fn=cmd_train)
    phys = tr.add_parser("physics")
    phys.add_argument("--episodes-per-seq", type=int, default=4)
    phys.add_argument("--mu-min", type=float, default=0.02)
    phys.add_argument("--mu-max", type=float, default=0.25)
    phys.add_argument("--nonlinear", action="store_true")
    phys.add_argument("--action-std", type=float, default=0.5)
    _add_model_args(phys)
    _add_train_args(phys)
    phys.set_defaults(fn=cmd_train, seq_len_default=512)

    models = sub.add_parser("models", help="list trained models")
    models.add_argument("--artifacts-root", default="artifacts")
    models.set_defaults(fn=cmd_models)


    cal = sub.add_parser("calibrate", help="calibrate harness thresholds, Fisher, and canaries for a model")
    cal.add_argument("model_id")
    cal.add_argument("--artifacts-root", default="artifacts")
    cal.add_argument("--data", default=None, help="text corpus dir with validation.bin")
    cal.add_argument("--chunks", type=int, default=512)
    cal.add_argument("--fisher-chunks", type=int, default=64)
    cal.add_argument("--prompts", default=None, help="pretrained chat backends: file of calibration prompts (one per line or a JSON list); default: the bundled benign set")
    cal.add_argument("--cusum-prompts", default=None, help="pretrained chat backends: prompts for the continuous CUSUM reference (default: --prompts)")
    cal.add_argument("--max-new-tokens", type=int, default=64, help="pretrained chat backends: generated tokens per calibration chat")
    cal.add_argument("--fpr", type=float, default=0.01)
    cal.add_argument("--device", default="cpu")
    cal.add_argument("--seed", type=int, default=0)
    cal.set_defaults(fn=cmd_calibrate)

    sess = sub.add_parser("session", help="create, list, fork, reset, show sessions").add_subparsers(dest="session_cmd", required=True)
    new_ = sess.add_parser("new")
    new_.add_argument("--model", required=True)
    new_.add_argument("--session-id", default=None)
    new_.add_argument("--harness-json", default=None, help="JSON object of HarnessConfig overrides")
    for sp_ in (new_,):
        sp_.add_argument("--artifacts-root", default="artifacts")
        sp_.add_argument("--device", default="cpu")
    sp_.set_defaults(fn=cmd_session)
    ls = sess.add_parser("list")
    ls.add_argument("--artifacts-root", default="artifacts")
    ls.add_argument("--device", default="cpu")
    ls.set_defaults(fn=cmd_session)
    fk = sess.add_parser("fork")
    fk.add_argument("parent")
    fk.add_argument("child", nargs="?", default=None)
    fk.add_argument("--artifacts-root", default="artifacts")
    fk.add_argument("--device", default="cpu")
    fk.set_defaults(fn=cmd_session)
    for name in ("reset", "resume", "show"):
        sp2 = sess.add_parser(name)
        sp2.add_argument("session_id")
        sp2.add_argument("--artifacts-root", default="artifacts")
        sp2.add_argument("--device", default="cpu")
        if name == "show":
            sp2.add_argument("--limit", type=int, default=20)
        sp2.set_defaults(fn=cmd_session)

    chat = sub.add_parser("chat", help="chat in a text session (prompt tokens are learned through transactions)")
    chat.add_argument("session_id")
    chat.add_argument("prompt")
    chat.add_argument("--artifacts-root", default="artifacts")
    chat.add_argument("--device", default="cpu")
    chat.add_argument("--max-new-tokens", type=int, default=128)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-k", type=int, default=50)
    chat.add_argument("--seed", type=int, default=None)
    chat.set_defaults(fn=cmd_chat)

    phys = sub.add_parser("physics", help="run a hidden-mu episode in a physics session")
    phys.add_argument("session_id")
    phys.add_argument("--artifacts-root", default="artifacts")
    phys.add_argument("--device", default="cpu")
    phys.add_argument("--steps", type=int, default=256)
    phys.add_argument("--mu", type=float, default=0.12)
    phys.add_argument("--seed", type=int, default=0)
    phys.add_argument("--nonlinear", action="store_true")
    phys.set_defaults(fn=cmd_physics)

    rt = sub.add_parser("redteam", help="attack a model through the token path and the harness")
    rt.add_argument("model_id")
    rt.add_argument("--artifacts-root", default="artifacts")
    rt.add_argument("--data", default="artifacts/data/wikitext")
    rt.add_argument("--device", default="cpu")
    rt.add_argument("--prefixes", type=int, default=8)
    rt.add_argument("--prefix-len", type=int, default=128)
    rt.add_argument("--suffix-len", type=int, default=64)
    rt.add_argument("--poison-chunks", type=int, default=8, help="coherence_poison payload length in chunks")
    rt.add_argument("--steps", type=int, default=50)
    rt.add_argument("--lr", type=float, default=0.05)
    rt.add_argument("--radius", type=float, default=1.0)
    rt.add_argument("--seed", type=int, default=0)
    rt.add_argument("--families", default="pgd,random,repeat,shuffle,topic_switch")
    rt.add_argument("--harness-json", default=None)
    rt.add_argument("--record", action="store_true", help="append the strongest payloads to the model's poison canaries")
    rt.set_defaults(fn=cmd_redteam)

    sl = sub.add_parser("sleep", help="consolidate accepted session learning into the slow weights, gated by locality checks")
    sl.add_argument("model_id")
    sl.add_argument("--artifacts-root", default="artifacts")
    sl.add_argument("--sessions", nargs="*", default=None, help="source sessions (default: every session of the model)")
    sl.add_argument("--steps", type=int, default=40)
    sl.add_argument("--lr", type=float, default=1e-4)
    sl.add_argument("--seq-len", type=int, default=512)
    sl.add_argument("--batch-size", type=int, default=2)
    sl.add_argument("--device", default="cpu")
    sl.add_argument("--seed", type=int, default=0)
    # TTT backend (chat models)
    sl.add_argument("--method", default="replay", choices=["replay", "distill", "anchor", "dream"], help="ttt: consolidation method")
    sl.add_argument("--target", default="w0", choices=["w0", "all"], help="ttt: which slow parameters change")
    sl.add_argument("--replay-ratio", type=float, default=0.5, help="ttt: share of each batch from the SFT replay corpus")
    sl.add_argument("--replay-rows", type=int, default=64)
    sl.add_argument("--heldout-rows", type=int, default=24, help="ttt: conversations for the locality measurement")
    sl.add_argument("--anchor-lambda", type=float, default=0.5)
    sl.add_argument("--distill-temperature", type=float, default=1.0)
    sl.add_argument("--tolerance-nll", type=float, default=0.05, help="ttt: allowed rise in mean held-out assistant NLL")
    sl.add_argument("--scan-checkpoint-groups", type=int, default=4)
    sl.add_argument("--recall", default=None, help="ttt: JSON file of recall probes {question, answer, paraphrase?}")
    sl.add_argument("--session-loss", default="all", choices=["all", "assistant"],
                    help="ttt: supervise every token of an accepted turn (default) or only the assistant's reply")
    sl.add_argument("--prompt-loss-weight", type=float, default=1.0, help="ttt: weight of the user's tokens vs the assistant's in a session turn (0-1)")
    sl.add_argument("--flagged-policy", default="exclude", choices=["exclude", "downweight", "include"],
                    help="ttt: accepted turns the policy flagged (scaled/projected/would-have-intervened): exclude from sleep, downweight, or include")
    sl.add_argument("--flagged-weight", type=float, default=0.25, help="ttt: row weight for flagged turns under --flagged-policy downweight")
    sl.add_argument("--replay-revision", default=SMOLTALK_REVISION,
                    help="ttt: HuggingFaceTB/smoltalk dataset revision for the replay and held-out samples (an empty string follows the Hub's main)")
    sl.add_argument("--dream-token-weighting", default="uniform", choices=["uniform", "gain", "fw_gain"], help="ttt dream: weight reply tokens by their information gain (turn+fast weights) or by the fast-weight part alone")
    sl.add_argument("--provenance", default="accepted", choices=["accepted", "all"],
                    help="ttt: 'all' consumes rolled-back turns too (experiment control only; never the product rule)")
    sl.add_argument("--out", default=None, help="ttt: run directory for the report and log (default: <artifacts>/sleep/<run>)")
    # toy plastic models
    sl.add_argument("--core", default="artifacts/data/wikitext", help="plastic: core corpus dir")
    sl.add_argument("--core-ratio", type=float, default=0.8, help="plastic: share of core corpus in each batch")
    sl.set_defaults(fn=cmd_sleep)

    srv = sub.add_parser("serve", help="run the API over an artifact store")
    srv.add_argument("--artifacts-root", default="artifacts")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=13579)
    srv.add_argument("--device", default="cpu")
    srv.add_argument("--log-level", default="info")
    srv.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
