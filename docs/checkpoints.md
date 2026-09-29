# Download the weights

The [Hugging Face model repository](https://huggingface.co/dmontgomery40/plastic/tree/main)
hosts the trained checkpoint files. GitHub carries the source; its ignored local
`artifacts/` directory is not the checkpoint distribution mechanism.

| Checkpoint | Weight file | Purpose |
| --- | --- | --- |
| TTT-MLP 760M chat, final SFT step 250 | [text-chat/model.safetensors](https://huggingface.co/dmontgomery40/plastic/blob/main/text-chat/model.safetensors) | Final chat evaluation, text-rule experiments and final-checkpoint Sleep results |
| PlasticCore WikiText | [text/checkpoint.pt](https://huggingface.co/dmontgomery40/plastic/blob/main/text/checkpoint.pt) | From-scratch text baseline |
| PlasticCore physics | [physics/checkpoint.pt](https://huggingface.co/dmontgomery40/plastic/blob/main/physics/checkpoint.pt) | Hidden-friction benchmark |
| Qwen base and abliterated comparison | [qwen/](https://huggingface.co/dmontgomery40/plastic/tree/main/qwen) | Complete local Qwen bundles, including the hosted derivative |
| TTT bases, intermediate SFT and research checkpoints | [checkpoints/](https://huggingface.co/dmontgomery40/plastic/tree/main/checkpoints) | 760M/1.3B bases, steps 50/100, smoke, legacy and coordinate-ablation weights |

The [complete weight inventory](https://huggingface.co/dmontgomery40/plastic/blob/main/WEIGHTS.md)
links every bundle; `weights-manifest.json` records their verified sizes and hashes.

## Final TTT chat checkpoint

Download the complete bundle, including its tokenizer and configuration:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "dmontgomery40/plastic",
    revision="ttt-chat-v1-step250",
    allow_patterns=["text-chat/*"],
    local_dir="artifacts/checkpoints/plastic",
)
PY
```

The checkpoint directory is `artifacts/checkpoints/plastic/text-chat`. It contains
`model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `sft_meta.json`, the upstream license and `manifest.json`.
The manifest records individual file sizes/SHA-256 values and the backend digest:

```text
29e0f8558d157cbfeaa9670334a38766448c9bdfe2ba651f286db22300aaf8c0
```

After installing the project's `pretrained` dependencies, load it through Plastic's
TTT implementation:

```python
from plastic.backends.ttt_lm.backend import TTTBackend
backend = TTTBackend.load("artifacts/checkpoints/plastic/text-chat", device="cpu")
```

This is the final save of the 250-step chat fine-tuning run on 8,000 SmolTalk
conversations, based on
[RetentionLabs/TTT-MLP-760M-Base-Pile-8k](https://huggingface.co/RetentionLabs/TTT-MLP-760M-Base-Pile-8k).
The [chat evaluation](research/results/chat-eval-2026-09-23/README.md) records its
limitations and exact evaluation conditions. A downloadable checkpoint does not
change the model selected in the live Space. Earlier reports use the separately
published `checkpoints/ttt-chat-step50/` and `checkpoints/ttt-chat-step100/` bundles;
this final checkpoint does not reproduce those earlier tables.

## Other checkpoints

Download `text/*` or `physics/*` from the same model repository to obtain the full
PlasticCore bundles, including their configuration, saved evaluation and manifest.
The base 760M and 1.3B TTT weights remain available from their upstream
[760M](https://huggingface.co/RetentionLabs/TTT-MLP-760M-Base-Pile-8k) and
[1.3B](https://huggingface.co/RetentionLabs/TTT-MLP-1.3B-Base-Pile-8k) repositories.
The hosted Qwen comparison uses the pinned upstream checkpoint described in the
[deployment guide](../deploy/huggingface/README.md).

The source publication workflow disables mirror deletions and verifies existing
`text/`, `physics/`, `text-chat/`, `qwen/` and `checkpoints/` file identities after sync.
Removed source files require explicit targeted cleanup rather than whole-repo deletion.
Private session state and incidental local experiment artifacts are not included.
