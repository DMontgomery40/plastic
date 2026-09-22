"""Pluggable text backends the transaction harness can drive.

The native `plastic` model is one backend; a pretrained model (Qwen3.5-0.8B, which is itself
gated-delta recurrent) is another, wrapped so its recurrent state can be snapshotted, committed,
rolled back, and probed exactly as the harness already does for the native fast-weight memory.
Transformers is an optional dependency, imported lazily only when a pretrained backend is used.
"""
