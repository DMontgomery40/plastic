# Learning contract on `phys_mps_3k`

Checkpoint digest `5689582b652cd62050b5699156601c6c9abf8c75f1492368b82b6cf372a84edd`, execution commit `3726ec21ffb829b693f92eb94a83ce5541c2a94f`, device cpu, contract 2026-09-23.2, seed 0.

## Before any stream (identical for every mode)

| measurement | with adaptation | writes disabled | elements |
|---|---|---|---|
| held-out combos, impulse | 0.3645 | 0.5300 | 2048 |
| held-out combos, hold | 0.0683 | 0.3603 | 2048 |
| held-out combos, release | 0.1099 | 0.1648 | 2048 |
| training distribution | 0.2495 | 0.4741 | 2048 |
| adaptation speed (held-out; adapting error as a fraction of writes-disabled error, by step) | mean 0.392; below one half at step 2 | | 8 episodes |

## After the stream, per mode

MSE deltas are after minus before on identical inputs; negative is improvement. `poison harm (vs clean)` is transfer MSE after the poisoned stream minus after the clean stream (damage plus the forgone clean gain); `poison harm (vs start)` is minus the poison arm's own pre-stream start (damage alone); `corr. residual` is after the corrective stream minus after clean. Continued training uses Adam. Acceptance is a pair of rates; n/a means no decision was recorded, not zero.

| mode | transfer Δ (impulse) | transfer Δ (hold) | transfer Δ (release) | forgetting Δ | poison harm (vs clean) | poison harm (vs start) | corr. residual | revert ok | accepted-good | refused-bad | tokens consumed | tokens measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | yes | n/a (n=0) | n/a (n=0) | 3072 | 25600 |
| continued | -0.1017 | +0.2214 | -0.0285 | -0.0765 | -0.0111 | +0.0193 | +0.0044 | yes | +1.0000 (n=2) | +0.0000 (n=1) | 3072 | 25600 |
| in_context | -0.0002 | -0.0006 | -0.0006 | -0.0003 | +0.0000 | -0.0005 | +0.0009 | yes | +1.0000 (n=2) | +0.0000 (n=1) | 3072 | 353280 |

Every measurement is taken from a fresh state with the stream removed, one episode per row. Adaptation window: {'update_period': 1, 'boundaries_per_episode': 63, 'checked': True}. Spec: docs/superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md
