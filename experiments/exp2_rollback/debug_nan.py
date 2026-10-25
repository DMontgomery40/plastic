"""Find the first non-finite step of the repo's text training loop (same RNG order as plastic.train.loop)."""
import sys, torch, os
from plastic.config import ModelConfig
from plastic.data.mqar import mqar_batch
from plastic.data.text import TokenWindows
from plastic.model.lm import build_model
from plastic.train.optim import build_optimizer
from plastic.train.schedule import lr_scale
from plastic.train.loop import text_loss
torch.set_num_threads(3)
lr_m, lr_o, steps = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
cfg = ModelConfig(d_model=128, n_layers=4, n_heads=4, vocab_size=4096)
torch.manual_seed(0); g = torch.Generator().manual_seed(0)
tw = TokenWindows("artifacts/data/wikitext/train.bin", 256)
model = build_model(cfg)
opt = build_optimizer(model, lr_matrix=lr_m, lr_other=lr_o, weight_decay=0.1, use_muon=True)
for step in range(1, steps + 1):
    opt.set_lr_scale(lr_scale(step, warmup=300, total=2000, min_ratio=0.1))
    use_mqar = float(torch.rand(1, generator=g)) < 0.2
    if use_mqar:
        pairs = (4, 8, 16)[int(torch.randint(0, 3, (1,), generator=g))]
        toks, _ = mqar_batch(16, n_pairs=pairs, seq_len=256, vocab_size=4096, key_range=(3, 2048), value_range=(2048, 4096), rng=g)
    else:
        toks = tw.sample(16, g)
    loss, _ = text_loss(model, toks)
    opt.zero_grad(set_to_none=True); loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    pbad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if step % 10 == 0 or not torch.isfinite(loss) or bad:
        mx = max(float(p.abs().max()) for p in model.parameters())
        print(step, "mqar" if use_mqar else "text", float(loss), gn, "maxparam", mx, "badgrad", bad[:4], "badparam", pbad[:4], flush=True)
    if not torch.isfinite(loss) or bad:
        torch.save(toks, "/tmp/bad_batch.pt"); break
    opt.step()
