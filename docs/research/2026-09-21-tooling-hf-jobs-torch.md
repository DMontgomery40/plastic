# Tooling research: HF Jobs, PyTorch 2.12 (Muon / torch.func / MPS), TTT reference implementations

Date: 2026-09-21. Local machine: Apple Silicon, macOS (Darwin 27), `python3.12` = 3.12.7, torch 2.12.0 (MPS available), huggingface_hub 1.15.0 (`hf` CLI 1.15.0). Every "verified locally" claim below was run on this machine today; scripts are in the scratchpad (`mps_checks.py`, `fla_titans_check.py`, `fla_naive_standalone/`).

---

## 1. Hugging Face Jobs (as of Sept 2026)

### 1.1 Version gap you should close first

The installed CLI is huggingface_hub **1.15.0** (released 2026-05-15). PyPI latest is **1.32.0**. Several documented Jobs features do not exist in 1.15.0 (confirmed with `hf jobs <cmd> --help` locally):

| Feature | Added in | In 1.15.0? |
|---|---|---|
| `hf jobs ls` (alias of `ps`), server-side filters | 1.21.0 | no (`ps` only) |
| `hf jobs wait <id>` / `wait_for_job()` | 1.20.0 | no |
| `--ssh` + `hf jobs ssh <id>` | 1.20.0 | no |
| `--expose <port>` | 1.19.0 | no |
| `--name` on run | 1.24.0 | no |
| Local-directory volumes `-v ./dir:/mnt[:rw]`, `sync_job_volume()` | 1.22.0 | no (only `hf://` sources) |
| `[tool.hf-jobs]` PEP 723 launch config, `--dry-run` | 1.32.0 | no |

Upgrade: `python3.12 -m pip install -U "huggingface_hub>=1.32"` (or `uv tool install -U huggingface_hub`). Sources: [releases](https://github.com/huggingface/huggingface_hub/releases), [releases page 2](https://github.com/huggingface/huggingface_hub/releases?page=2).

### 1.2 Hardware and prices (local `hf jobs hardware` output, verbatim)

```
NAME            PRETTY NAME        CPU      RAM     ACCELERATOR  COST/MIN COST/HOUR
cpu-basic       CPU Basic          2 vCPU   16 GB                $0.0002  $0.01
cpu-upgrade     CPU Upgrade        8 vCPU   32 GB                $0.0005  $0.03
cpu-performance CPU Performance    32 vCPU  256 GB               $0.0317  $1.90
cpu-xl          CPU XL             16 vCPU  124 GB               $0.0167  $1.00
t4-small        Nvidia T4 - small  4 vCPU   15 GB   1x T4 (16GB) $0.0067  $0.40
t4-medium       Nvidia T4 - medium 8 vCPU   30 GB   1x T4 (16GB) $0.0100  $0.60
a10g-small      Nvidia A10G - s... 4 vCPU   15 GB   1x A10G (24) $0.0167  $1.00
a10g-large      Nvidia A10G - l... 12 vCPU  46 GB   1x A10G (24) $0.0250  $1.50
a10g-largex2    2x Nvidia A10G ... 24 vCPU  92 GB   2x A10G      $0.0500  $3.00
a10g-largex4    4x Nvidia A10G ... 48 vCPU  184 GB  4x A10G      $0.0833  $5.00
a100-large      Nvidia A100 - l... 12 vCPU  142 GB  1x A100 (80) $0.0417  $2.50
a100x4          4x Nvidia A100     48 vCPU  568 GB  4x A100      $0.1667  $10.00
a100x8          8x Nvidia A100     96 vCPU  1136 GB 8x A100      $0.3333  $20.00
h200            Nvidia H200        23 vCPU  256 GB  1x H200(141) $0.0833  $5.00
h200x2/x4/x8                                                     $10 / $20 / $40 per hour
rtx-pro-6000    RTX PRO 6000 (96GB) 23 vCPU 256 GB               $0.0458  $2.75
rtx-pro-6000x2/x4/x8                                             $5.50 / $11 / $22 per hour
l4x1            1x Nvidia L4       8 vCPU   30 GB   1x L4 (24GB) $0.0133  $0.80
l4x4            4x Nvidia L4       48 vCPU  186 GB  4x L4        $0.0633  $3.80
l40sx1          1x Nvidia L40S     8 vCPU   62 GB   1x L40S(48)  $0.0300  $1.80
l40sx4 / l40sx8                                                  $8.30 / $23.50 per hour
```

Billing is per minute, only while Starting/Running; default **timeout is 30 minutes** unless you pass `--timeout`. Jobs require a positive credit balance; PRO/Team/Enterprise monthly compute credits count toward it. H100 flavors were removed in Dec 2025. Source: [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing).

**ZeroGPU / `zero-a10g`:** the `--flavor` choice list in the CLI includes `zero-a10g` (and `sprx8`, `inf2x6`), but that is the shared `SpaceHardware` enum leaking through. It is *not* listed by `hf jobs hardware` and ZeroGPU is documented as Spaces-only (Gradio SDK, `@spaces.GPU` decorator, 60 s default call duration, no `torch.compile`, now backed by RTX PRO 6000 slices with 5 min/day free and 40 min/day PRO quota, $1 per 10 min over quota). There is no free GPU tier for Jobs; the cheapest Jobs compute is `cpu-basic` at $0.01/h. Source: [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).

### 1.3 Running a script on a GPU flavor

Two entry points, both verified from local `--help` output:

```
hf jobs run [OPTIONS] IMAGE COMMAND...           # docker-run style
hf jobs uv run [OPTIONS] SCRIPT [SCRIPT_ARGS]... # PEP 723 script, local file or URL
```

Common options on both: `--flavor`, `-e/--env`, `-s/--secrets` (`--secrets HF_TOKEN` reads your local token), `-v/--volume`, `--env-file`, `--secrets-file`, `--timeout` (`30m`, `2h`, `1d`, or seconds), `-d/--detach`, `--namespace`, `-l/--label`. `uv run` adds `--image`, `--with PKG`, `-p/--python`.

Docker-image route (image has torch+CUDA preinstalled, so startup is fast):

```bash
hf jobs run --flavor l4x1 --timeout 2h --secrets HF_TOKEN --detach \
  -v hf://buckets/DMontgomery40/ttt-runs:/out \
  pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel \
  bash -lc 'git clone https://github.com/DMontgomery40/plastic.git /w && cd /w && pip install -q -e . \
            && python -m plastic.cli train text --data /out/data/wikitext --artifacts-root /out/artifacts --device cuda --steps 2000'
```

uv-script route (default image `ghcr.io/astral-sh/uv:python3.12-bookworm`; the local file is uploaded to a temporary repo at submit time and shipped into the container):

```bash
hf jobs uv run --flavor a10g-small --timeout 90m --secrets HF_TOKEN --detach \
  -v hf://buckets/DMontgomery40/ttt-runs:/out \
  train_ttt.py -- --steps 2000 --out /out
```

Streaming logs and managing:

```bash
hf jobs logs -f <job_id>          # follow until completion (non-blocking without -f)
hf jobs logs --tail 50 <job_id>
hf jobs ps                        # 1.15.0 name; `hf jobs ls` from 1.21
hf jobs inspect <job_id>
hf jobs stats <job_id>            # cpu/mem/gpu usage
hf jobs cancel <job_id>
hf jobs wait <job_id> && ...      # needs >= 1.20
```

Non-detached `hf jobs run` exits non-zero if the job fails, so `hf jobs run ... && next` chains correctly. Built-in env vars inside the container: `JOB_ID`, `ACCELERATOR`, `CPU_CORES`, `MEMORY`. Sources: [Jobs guide](https://huggingface.co/docs/huggingface_hub/guides/jobs), [Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration).

### 1.4 Getting code in

- **Single script:** `hf jobs uv run local.py` uploads the file; a `https://...` URL to a raw file also works. It does **not** take a directory or a git URL.
- **A directory:** `-v ./dir:/mnt` (>= 1.22) syncs the directory into your private `jobs-artifacts` bucket and mounts it read-only (`:rw` to write back; the CLI prints the `hf buckets sync` command to pull results). Re-syncs only upload changed files.
- **A git repo:** `git clone` inside the command. Use `--secrets GITHUB_TOKEN` and `https://${GITHUB_TOKEN}@github.com/...` for private repos. Note the `pytorch/pytorch:*-runtime` images do **not** ship `git` or `curl` (the official Dockerfile installs only ca-certificates/libjpeg/libpng/python in the runtime stage); use the `-devel` tag, the default uv image (`python:3.12-bookworm` base, has git), or `apt-get install -y git` first.
- **Launch config in the script (>= 1.32):** a `[tool.hf-jobs]` table in the PEP 723 header sets `image`, `flavor`, `python`, `timeout`, `name`, `namespace`, `env`, `secrets`, `labels`, `volumes`; CLI flags override. Read by the `hf` CLI only, ignored by `run_uv_job()`.

Docker image choice for this project: `pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel` (tags verified on Docker Hub, updated 2026-06-18; cu13.0 and cu13.2 variants also exist). The PyTorch 2.12 default PyPI wheel is CUDA 13.0, which needs driver >= 580; the HF docs still show `pytorch/pytorch:2.6.0-cuda12.4` as the canonical example, so cu126 is the conservative pick. If you go the uv route, pin the index in the script header so the *same file* runs on MPS locally and on CUDA in the job:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.12.*", "huggingface_hub>=1.32", "numpy"]
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'linux'" }]
# [tool.hf-jobs]
# flavor = "l4x1"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# volumes = ["hf://buckets/DMontgomery40/ttt-runs:/out"]
# ///
```

Source: [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/), [Jobs images](https://huggingface.co/docs/hub/jobs-images). To reuse an image's preinstalled torch with uv extras, the documented pattern is `hf jobs uv run --image huggingface/trl --python /opt/conda/bin/python3 -e PYTHONPATH=/opt/conda/lib/python3.11/site-packages ...`.

### 1.5 Getting results out

**Option A: Hub repo from inside the job** (versioned, good for final checkpoints). `HF_TOKEN` is picked up automatically from the secret:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("DMontgomery40/ttt-ssm-runs", repo_type="model", exist_ok=True)
api.upload_folder(folder_path="/out", repo_id="DMontgomery40/ttt-ssm-runs",
                  path_in_repo=f"runs/{os.environ['JOB_ID']}", commit_message="job results")
# single file: api.upload_file(path_or_fileobj="/out/metrics.json", path_in_repo="metrics.json", repo_id=..., repo_type="dataset")
# many/large files: api.upload_large_folder(repo_id=..., folder_path="/out", repo_type="model")
```

**Option B: Storage Bucket volume** (mutable, no git history, overwrite-in-place; right for rolling checkpoints and logs). Buckets are a GA repo type "available to all users and organizations", S3-like, Xet-backed, with a free allowance and then $12/TB/month public or $18/TB/month private (volume discounts from 50 TB). Mounted buckets are read-write by default and fetched lazily.

```bash
hf buckets create ttt-runs --private                 # or https://huggingface.co/new-bucket
hf jobs run -v hf://buckets/DMontgomery40/ttt-runs:/out ... python train.py --out /out/run1
hf buckets list DMontgomery40/ttt-runs -R -h          # browse
hf buckets sync hf://buckets/DMontgomery40/ttt-runs/run1 ./artifacts/run1   # pull back (alias: hf sync)
hf buckets cp hf://buckets/DMontgomery40/ttt-runs/run1/metrics.json - | jq .
```

Python: `create_bucket`, `sync_bucket`, `batch_bucket_files`, `download_bucket_files` (all present in 1.15.0). Model/dataset repos can also be mounted read-only (`-v hf://datasets/org/ds:/data`). Sources: [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets), [hf.co/storage pricing](https://huggingface.co/storage).

### 1.6 Scheduled jobs

```
hf jobs scheduled run  SCHEDULE IMAGE COMMAND...      # SCHEDULE: annually|monthly|weekly|daily|hourly or cron "0 9 * * 1"
hf jobs scheduled uv run SCHEDULE SCRIPT [ARGS]...
hf jobs scheduled ps | inspect <id> | suspend <id> | resume <id> | delete <id>
```

Same `--flavor/--secrets/--timeout/-v` flags plus `--suspend/--no-suspend` and `--concurrency/--no-concurrency` (allow overlapping instances). Python: `create_scheduled_job`, `create_scheduled_uv_job`, `trigger_scheduled_job`. SSH is not available on scheduled jobs. Example: `hf jobs scheduled uv run "@daily" --flavor t4-small --secrets HF_TOKEN nightly_eval.py`.

---

## 2. PyTorch 2.12 on this machine

### 2.1 `torch.optim.Muon` exists (verified locally)

```
Muon.__init__(self, params, lr: float = 0.001, weight_decay: float = 0.1, momentum: float = 0.95,
              nesterov: bool = True, ns_coefficients: tuple = (3.4445, -4.775, 2.0315),
              eps: float = 1e-07, ns_steps: int = 5, adjust_lr_fn: str | None = None)
```

Docstring facts: momentum buffer `B_t = mu*B_{t-1} + g_t`, Nesterov `g_t + mu*B_t`, Newton-Schulz orthogonalization `O_t = NS_k^{(a,b,c)}(B_t; eps)`, decoupled weight decay `theta -= lr*wd*theta`, then `AdjustLR(lr, shape)`, then `theta -= lr*O_t`. `adjust_lr_fn` accepts `"original"` (Keller: `sqrt(max(1, A/B))`) or `"match_rms_adamw"` (Moonshot: `0.2*lr*sqrt(max(A,B))`); `None` means original. `"spectral_unclamped"` appears in the 2.14 docs but **raises `ValueError` in 2.12** (tested). Muon is for **2D hidden-layer params only**; biases/embeddings go to AdamW. Gotchas: default `weight_decay=0.1` is non-zero, and `lr=1e-3` is far below the usual Muon 0.02. Muon `.step()` runs on MPS (tested). Source module: `torch/optim/_muon.py`, which also exposes reusable functional helpers `_zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps)` and `_adjust_lr(lr, adjust_lr_fn, param_shape)`. Docs: [torch.optim.Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html).

Relevance: `torch.optim.Muon` is a stateful in-place optimizer, so you cannot differentiate through it for meta-learning. For a Muon-style *inner* loop (LaCT does this), apply `_zeropower_via_newtonschulz` to the `torch.func.grad` output and update functionally; that stays differentiable.

### 2.2 `torch.func` and MPS (all verified locally, torch 2.12.0)

`torch.func` exports: `functional_call, grad, grad_and_value, vmap, vjp, jvp, jacrev, jacfwd, hessian, linearize, functionalize, stack_module_state`.

| Check | CPU | MPS |
|---|---|---|
| `cumsum` + `cumprod` backward (incl. cumprod with a zero) | OK | OK (correct grads, 1.5-2.3 s first call = shader warm-up) |
| `torch.func.grad` matches autograd | diff 0.0 | diff 0.0 |
| `vmap(grad(functional_call))` per-chunk grads over 6 chunks, 2-layer MLP | OK | OK |
| 2nd-order meta-gradient through 4 inner `torch.func.grad` SGD steps (grad wrt init params and inner lr) | OK | OK |
| `vmap(grad)` over cumsum/cumprod | OK | OK |
| `torch.optim.Muon.step` | OK | OK |
| `torch.compile` default backend, fwd+bwd on MLP | OK (6 s compile) | OK, diff 6e-8 |
| `torch.compile` of cumprod/cumsum | OK | OK |
| `torch.compile(backend="aot_eager")` | OK | OK |
| `torch._higher_order_ops.associative_scan` (generic combine) | OK | OK |
| bfloat16 matmul | OK | OK |

`torch.compile` on MPS is real Inductor codegen, not a CPU fallback: with `TORCH_LOGS=output_code` the generated code shows `async_compile.metal(...)` kernels using `metal::precise::exp/tanh`, and `torch._inductor.codegen.mps` provides `MetalKernel`/`MetalScheduling`. PyTorch 2.12 (released 2026-05-13) also ships Apple Silicon wheels with ahead-of-time compiled Metal-4 shaders and puts all MPS tensors in unified memory. Caveat from the MPS backend generally: unsupported ops fall back to CPU silently rather than erroring, so profile before assuming speed. Source: [PyTorch 2.12 release blog](https://pytorch.org/blog/pytorch-2-12-release-blog/).

Bottom line: the meta-learning-through-TTT-inner-loop pattern (`functional_call` + `grad` + `vmap`, optionally second-order) works on MPS today with no workaround.

---

## 3. Reference TTT implementations to learn from

| Repo | Inner-loop formulation | Pure PyTorch on a Mac? |
|---|---|---|
| [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) (`ttt.py`, MIT-ish research code) | TTT-Linear (`W1,b1` per head) and TTT-MLP (`W1,b1,W2,b2`, GELU). Self-supervised reconstruction target with LayerNorm; the LN + L2 backward is hand-derived (`ln_fwd`, `ln_fused_l2_bwd`), no autograd in the inner loop. Mini-batch **dual form**: `Attn1 = tril(XQ @ X1^T)`, `b1_bar = b1 - tril(eta) @ grad`, so updates are batched matmuls with causal masks. Token-dependent learnable inner lr: `eta = sigmoid(X @ lr_weight + lr_bias) * token_idx`. Iterates mini-batches with a JAX-style `scan()` plus optional gradient checkpointing; no `vmap`. | Yes. Only `torch` + `transformers`; optional `causal_conv1d` with an `nn.Conv1d` fallback. Best single-file read for the dual form. |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 0.5.2 | Has **naive pure-torch references**: `fla/ops/ttt/naive.py` (`chunk_ttt_linear_ref(q,k,v,w,b,eta,scale,eps,mini_batch_size,initial_state,initial_state_bias,output_final_state)`: per mini-batch `kh = k@h + hb`, LN, `g = w*kh_hat + b - (v - k)`, `h -= (eta*k)^T @ v_new`, returns `(o, h, hb)`) and `fla/ops/titans/naive.py` (`chunk_titans_linear_ref(q,k,v,w,b,theta,alpha,eta,...)`: linear memory `M_t=(1-alpha_t)M_{t-1}+S_t`, surprise `S_t=eta_t S_{t-1} - 2 theta_t k^T v_new`, chunked via `cumprod` of `1-alpha` and `eta`). `fla/layers` has `DeltaNet`, `GatedDeltaNet`, `GatedDeltaNet2`, `MesaNet`, `MomentumDeltaNet`, `Comba`, `PaTHAttention`, `Mamba2/3`, `RWKV7`, `KimiDeltaAttention`, etc. There is **no** `fla.layers.ttt`, `titans`, or `lact` layer; TTT/Titans exist only as ops (`chunk.py`, `fused_chunk.py`, `log_impl.py`). | **Not directly.** `pip install flash-linear-attention` installs on macOS (tested), but `import fla.ops.ttt.naive` fails with `ModuleNotFoundError: triton` because `fla/ops/__init__.py` imports Triton kernels. Copying `ttt/naive.py` (130 lines, imports only torch/F) and `titans/naive.py` + `titans/log_impl.py` out of site-packages runs fine: fwd+bwd verified on CPU and MPS (TTT ref: 10 ms CPU / 69 ms MPS; Titans ref: 17 ms / 510 ms). Note the TTT ref does in-place ops, so pass non-leaf tensors. |
| [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) 0.5.5 | `NeuralMemory`: memory is an arbitrary `nn.Module` (default MLP); per-chunk gradients via `torch.func`: `grad_fn = grad(forward_and_loss, has_aux=True); per_sample_grad_fn = vmap(grad_fn, in_dims=(0,0,0,0))` over `functional_call(memory_model, params, inputs)` with MSE loss. Surprise = `-grads`; momentum and adaptive forgetting via associative scans (`assoc_scan(1-decay, update, prev=...)`), optional `spectral_norm_surprises` applies `newtonschulz5` (Muon-style). `MemoryAsContextTransformer` wraps it with segment attention + persistent/long-term memory tokens. | Yes. Depends on `assoc-scan` (default path is a pure-torch recursive divide-and-conquer scan; `use_accelerated=True` needs the Triton `accelerated-scan` package) plus `x-transformers`, `einops`. Verified: `NeuralMemory` and `MemoryAsContextTransformer` fwd+bwd run on CPU and MPS (MAC 43 ms CPU / 4 s MPS first call). This is the closest existing code to the "meta-learn through the inner loop" pattern. |
| [a1600012888/LaCT](https://github.com/a1600012888/LaCT) (MIT) | `minimal_implementations/causal_lact_with_sliding_window_attn.py` and `bidirectional_lact_layer.py`. Fast weight is a SwiGLU MLP `f(x)=w1 @ (silu(w0 x) * (w2 x))`; **large chunks** (2K to 1M tokens) with hand-derived closed-form gradients (`silu_backprop`, `dw1 = bmm(v, (hidden^T * lr1))`), per-chunk learned lr `softplus(lr + base_lr_inv)`, optional momentum, and `use_muon=True` orthogonalizing each chunk's update with `zeropower_via_newtonschulz5`. Sliding-window attention runs in parallel and is summed with the TTT output. | Mostly. The TTT core is torch + einops only; the attention branch imports `flash_attn` (wrapped in try/except with a warning), so on a Mac swap it for `F.scaled_dot_product_attention` with a window mask. Optional Triton kernels exist for the fused layer but are not required. |
| [Yingfa Chen, "Implementing TTT, Part 1" (2025)](https://chen-yingfa.github.io/research_posts/2025-ttt-implementation/) | ~50-line forward pass deriving the dual form for a 2-layer MLP learner by hand ("backward matmuls" using matrix associativity), explicitly avoiding `torch.func`. Code lives inline in the post. | Yes, pure torch. |

Other pointers: `kyegomez/TTL` (unofficial TTT-Linear, lower quality), `nikitadurasov/torch-ttt` (test-time *adaptation* framework, different problem), `LeapLabTHU/Transformer-to-TTT` (ICML 2026, vision), `zeyun-zhong/E2-TTT`. I did not find a well-known "TTT in 100 lines" gist; the fla `ttt/naive.py` (130 lines) and the LaCT minimal file are the closest.

---

## Recommended setup

1. **Upgrade the CLI** to huggingface_hub 1.32 before touching Jobs. 1.15.0 lacks `wait`, `ls`, `ssh`, local-dir volumes and the `[tool.hf-jobs]` header, and those are what make the workflow pleasant.
2. **Develop on the Mac with MPS.** Everything the project needs (`functional_call`/`grad`/`vmap`, second-order meta-gradients, cumsum/cumprod backward for the SSM, `torch.compile` Metal kernels, `torch.optim.Muon`) works on torch 2.12 MPS today. Warm-up latency on first call is 0.5 to 4 s; steady state is fine.
3. **For the inner loop, write it functionally.** Follow titans-pytorch's pattern (`vmap(grad(loss_of_functional_call))` per chunk) so the outer loop can meta-learn inner lr, gates, or the adapter init. For a Muon-flavoured inner step, reuse `torch.optim._muon._zeropower_via_newtonschulz` on the functional gradient; keep `torch.optim.Muon` for the *outer* optimizer only, with `weight_decay=0` set explicitly and 2D params separated from AdamW params.
4. **Reference code:** read `ttt-lm-pytorch/ttt.py` for the dual form, copy `fla/ops/ttt/naive.py` and `fla/ops/titans/naive.py` into the repo as pure-torch test oracles (do not import `fla` on macOS), and look at LaCT's minimal file for the large-chunk + Newton-Schulz variant. Skip fla's Triton layers unless a CUDA job needs throughput.
5. **Cloud runs on HF Jobs.** For this toy scale (64-dim, 8K vocab), `cpu-upgrade` ($0.03/h) or `t4-small` ($0.40/h) is enough for sweeps; use `l4x1` ($0.80/h, 24 GB) or `a10g-small` ($1.00/h) when the meta-learning inner loop grows. Always pass `--timeout` (default is 30 min) and `--detach`, then `hf jobs logs -f`.
6. **Code in, results out:** one PEP 723 script with the cu126 index pinned under `sys_platform == 'linux'` so it runs unchanged on MPS locally and on CUDA in the job; `git clone` the repo in the command if the script needs the package (use a `-devel` PyTorch tag or the default uv image, since `-runtime` images lack git). Create one private bucket (`hf buckets create ttt-runs --private`), mount it at `/out` for checkpoints and JSONL metrics, `hf sync` it back, and promote only final artifacts to a versioned model/dataset repo with `upload_folder`.
7. **Scheduled nightly eval** once the pipeline is stable: `hf jobs scheduled uv run "@daily" --flavor t4-small --secrets HF_TOKEN -v hf://buckets/DMontgomery40/ttt-runs:/out nightly_eval.py`.

Sources consulted: [HF Jobs guide](https://huggingface.co/docs/huggingface_hub/guides/jobs), [Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration), [Jobs images](https://huggingface.co/docs/hub/jobs-images), [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing), [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli), [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets), [hf.co/storage](https://huggingface.co/storage), [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), [huggingface_hub releases](https://github.com/huggingface/huggingface_hub/releases), [torch.optim.Muon docs](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html), [PyTorch 2.12 blog](https://pytorch.org/blog/pytorch-2-12-release-blog/), [pytorch/pytorch Docker tags](https://hub.docker.com/r/pytorch/pytorch/tags), [pytorch Dockerfile](https://github.com/pytorch/pytorch/blob/main/Dockerfile), [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/), [ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [titans-pytorch](https://github.com/lucidrains/titans-pytorch), [assoc-scan](https://github.com/lucidrains/assoc-scan), [LaCT](https://github.com/a1600012888/LaCT), [Implementing TTT Part 1](https://chen-yingfa.github.io/research_posts/2025-ttt-implementation/).
