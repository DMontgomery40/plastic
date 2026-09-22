# M2 Data and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the first real `plastic` models: a text model on wikitext-103-raw-v1 (HF Jobs, L4) and a physics model on hidden-mu episodes (local MPS), with a tokenizer that works, a data pipeline, an outer loop that logs and evaluates the three checkpoint numbers, and a model registry.

**Architecture:** `plastic data prepare` trains a byte-level BPE (HF `tokenizers`, vocab 8192) and encodes wikitext splits to uint16 token files; `plastic/train/loop.py` samples T = 1024 windows, mixes MQAR batches, runs the Muon/AdamW outer loop with warmup and cosine, evaluates held-out loss with and without the memory plus MQAR accuracy and the β histogram, and registers checkpoints in `artifacts/models/`. The physics domain generates episodes on the fly. One PEP 723 script launches the same loop on Hugging Face Jobs.

**Tech Stack:** torch 2.14, tokenizers, datasets, huggingface_hub 1.32 (`hf jobs uv run`), numpy memmap, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-plastic-design.md` sections 4.2, 4.3, 5, 12 (item 13).

## Global Constraints

- Package `plastic`, `uv run` for everything, tests on CPU with MPS parametrization.
- Text vocabulary 8192; sequences T = 1024 built by concatenating documents with `<eos>`; held-out evaluation only on wikitext validation/test.
- Checkpoints ship `eval.json` with: held-out loss, held-out loss with `beta_scale=0`, MQAR accuracy by pair count, β histogram.
- No "legacy"/"v1" language, no schema-version fields; `model_signature = sha256(config JSON || checkpoint bytes)`.
- Every task ends with `uv run pytest -q` green and a commit on `fuse`. Never push.

---

### Task 1: Tokenizer (`plastic/tokenizer/bpe.py`)

**Files:** Create `plastic/tokenizer/__init__.py`, `plastic/tokenizer/bpe.py`; Test `tests/test_tokenizer.py`.

**Interfaces:**
- `Tokenizer` wrapper around `tokenizers.Tokenizer` (byte-level BPE, `ByteLevel` pretokenizer with whitespace preserved, special tokens `<pad>=0, <bos>=1, <eos>=2`).
  - `Tokenizer.train(files: list[str] | Iterable[str], *, vocab_size: int = 8192, max_lines: int | None = None) -> Tokenizer` (class method; trains from an iterator of lines).
  - `encode(text, *, add_bos=False, add_eos=False) -> list[int]`, `decode(ids, *, skip_special=True) -> str`, `save(path)`, `Tokenizer.load(path)`, properties `vocab_size, pad_id, bos_id, eos_id`.

**Tests:** whitespace and punctuation round-trip (`"hello  world\tfoo\nbar"` decodes back exactly), `vocab_size == 8192` after training on a small text with `vocab_size=300` yielding exactly 300, special ids fixed, `encode` of two different words gives different ids, saved/loaded tokenizer encodes identically.

---

### Task 2: Text corpus pipeline (`plastic/data/text.py`)

**Files:** Create `plastic/data/__init__.py`, `plastic/data/text.py`; Test `tests/test_data_text.py`.

**Interfaces:**
- `prepare_text_corpus(*, corpus: Literal["wikitext","fineweb"], out_dir: str, vocab_size: int = 8192, tokenizer_lines: int = 200_000, max_train_tokens: int | None = None, cache_dir: str | None = None) -> CorpusMeta` downloads via `datasets` (`Salesforce/wikitext`, `wikitext-103-raw-v1`; or `HuggingFaceFW/fineweb-edu` `sample-10BT` streamed), trains the tokenizer on the first `tokenizer_lines` train lines, encodes each split to `<out_dir>/<split>.bin` (uint16 little-endian, documents joined by `<eos>`), writes `<out_dir>/tokenizer.json` and `<out_dir>/meta.json` (`CorpusMeta`: corpus, vocab_size, splits with token counts, created_at).
- `encode_documents_to_bin(tok: Tokenizer, docs: Iterable[str], path: str, *, max_tokens: int | None = None) -> int` (shared by both corpora; used directly by tests with synthetic docs).
- `TokenWindows(path: str, seq_len: int)`: memmap sampler with `sample(batch: int, rng: torch.Generator) -> Tensor (B, seq_len+1)` and `sequential(batch: int, seq_len: int) -> Iterator[Tensor]` for evaluation.
- wikitext document boundaries: a new document starts at a line matching a level-1 heading (` = Title = `, exactly one `=` on each side); blank lines are dropped.

**Tests:** `encode_documents_to_bin` writes uint16 with `<eos>` between docs and honors `max_tokens`; `TokenWindows.sample` returns in-range ids with shape (B, T+1); `sequential` covers the file once without overlap; wikitext heading detection on a synthetic sample yields the right document count.

---

### Task 3: MQAR and physics data (`plastic/data/mqar.py`, `plastic/data/physics.py`)

**Interfaces:**
- `mqar_batch(batch: int, *, n_pairs: int, seq_len: int, vocab_size: int, key_range: tuple[int,int], value_range: tuple[int,int], rng: torch.Generator) -> tuple[Tensor tokens (B, seq_len), Tensor answer_mask (B, seq_len)]`: `n_pairs` distinct keys, values, then queries of every key in random order; `answer_mask` marks positions whose next token is a queried value; pads with `<pad>` to `seq_len` (padded targets are masked in the loss).
- `mqar_accuracy(logits, tokens, answer_mask) -> float`.
- `PhysicsEnv(mu: float, *, nonlinear: bool = False, dt: float = 1.0, noise_std: float = 0.0)` with `reset() -> obs (4,)`, `step(action (2,)) -> obs (4,)` (exact port of the friction dynamics).
- `physics_batch(batch: int, *, seq_len: int, episodes_per_seq: int, mu_range: tuple[float,float], nonlinear: bool, action_std: float, rng: torch.Generator) -> PhysicsBatch(inputs (B, T, 7), target_delta (B, T, 4), mu (B, episodes), reset_flag (B, T))` vectorized in torch; `reset_flag = 1` on the first step of each episode; targets are `obs_{t+1} − obs_t`.

**Tests:** MQAR answer positions are exactly the value tokens after each query key; accuracy of a perfect oracle is 1.0; `PhysicsEnv` matches the closed form `vel <- (1-mu) vel + a; pos += vel` for a few steps; batch reset flags count equals `episodes_per_seq`; targets equal finite differences of the trajectory.

---

### Task 4: Model registry (`plastic/store.py`)

**Interfaces:**
- `ArtifactStore(root: str)` with `models_dir`, `model_dir(model_id)`, `new_model_id(prefix)`, `register_model(model_id, record)`, `list_models() -> list[dict]`, `load_model_record(model_id)`, `save_checkpoint(model_id, cfg: ModelConfig, model: nn.Module, *, step: int, extra: dict) -> str`, `load_checkpoint(model_id, device) -> tuple[ModelConfig, nn.Module, dict]`, `model_signature(model_id) -> str` (sha256 of `cfg.signature_material()` and `checkpoint.pt` bytes), `write_eval(model_id, eval: dict)`, `tokenizer_path(model_id)`.
- Index at `<root>/models/index.json` `{ "models": { id: record } }`.

**Tests:** register, list sorted by created time, checkpoint round-trip restores equal outputs, signature changes when the checkpoint changes, eval written and readable.

---

### Task 5: Outer loop (`plastic/train/loop.py`, `plastic/train/schedule.py`)

**Interfaces:**
- `TrainConfig` dataclass: `domain`, `model: ModelConfig`, `data_dir` (text), `steps`, `batch_size`, `seq_len`, `lr_matrix`, `lr_other`, `weight_decay`, `warmup_steps`, `min_lr_ratio`, `grad_clip`, `mqar_frac`, `mqar_pairs`, `eval_every`, `eval_batches`, `save_every`, `seed`, `device`, `log_every`, physics fields `episodes_per_seq, mu_range, nonlinear, action_std`, `model_id` (optional), `artifacts_root`.
- `lr_scale(step, *, warmup, total, min_ratio) -> float` (linear warmup then cosine to `min_ratio`).
- `train(cfg: TrainConfig, *, log=print) -> str` returns `model_id`; writes `train_log.jsonl` (step, loss, grad_norm, lr_scale, tokens, seconds, tok_per_s), `eval.json` at every eval and at the end (`heldout_loss`, `heldout_loss_beta0`, `memory_value = heldout_loss_beta0 − heldout_loss`, `mqar_accuracy: {pairs: acc}`, `beta_hist: {edges, counts}`, `alpha_mean`), checkpoints via the store, and registers the model with status `running|completed|failed`.
- `evaluate(model, *, cfg, data, device) -> dict`.

**Tests:** `lr_scale` shape; 6-step text run on a synthetic `.bin` corpus writes a checkpoint, log, eval with all keys, and a `completed` record; 6-step physics run likewise; deterministic given seed (two runs give identical step-1 loss).

---

### Task 6: CLI and HF Jobs launcher

**Files:** `plastic/cli.py` (`plastic data prepare`, `plastic train text|physics`, `plastic eval`), `scripts/hf_jobs/train_text.py` (PEP 723 script: clones the repo at a given ref, installs, runs `prepare` if the bucket lacks the corpus, runs `train`, syncs `artifacts/models/<id>` to the bucket), `scripts/hf_jobs/launch_text.sh` (the `hf jobs uv run --flavor l4x1 ... --timeout 2h --secrets HF_TOKEN -v hf://buckets/DMontgomery40/plastic-runs:/out` command).

**Tests:** `plastic --help` lists subcommands; `plastic train physics --steps 3` completes on CPU.

---

### Task 7: First runs

1. Local: `uv run plastic train physics --steps 3000 --device mps` (about 10 minutes); record the three numbers and the mu-switch behavior in `docs/research/2026-09-22-first-runs.md`.
2. Cloud: create bucket `plastic-runs`; launch `train_text` on `l4x1` (wikitext, vocab 8192, 6.8M params, T = 1024, B = 32, steps sized to about 60M tokens); follow logs; sync the model back; record held-out loss, memory value, MQAR accuracy, β histogram.

## Self-review

- Spec coverage: 4.2 tokenizer/vocab (T1), 4.3 registry (T4), 5 data/optimizer/compute/launch (T2, T3, T5, T6, T7), checkpoint numbers (T5). Calibration (6.5) belongs to M3.
- Type consistency: `Tokenizer`, `TokenWindows`, `PhysicsBatch`, `TrainConfig`, `ArtifactStore` names are used identically across tasks.
