# Learning contract on `phys_mps_3k`

Checkpoint digest `5689582b652cd62050b5699156601c6c9abf8c75f1492368b82b6cf372a84edd`, execution commit `59e0f29add4d74425866e548083fe62efea1cb08`, device cpu, contract 2026-09-23.1, seed 0.

## Before any stream (identical for every mode)

| measurement | with adaptation | writes disabled | elements |
|---|---|---|---|
| held-out combos, impulse | 0.2963 | 0.4373 | 2048 |
| held-out combos, hold | 0.0974 | 0.3553 | 2048 |
| held-out combos, release | 0.1042 | 0.2414 | 2048 |
| training distribution | 0.2123 | 0.4084 | 2048 |
| adaptation speed (held-out; adapting error as a fraction of writes-disabled error, by step) | mean 0.584; below one half at step 2 | | 32 episodes |

## After the stream, per mode

MSE deltas are after minus before on identical inputs; negative is improvement. `poison harm` is transfer MSE after the poisoned stream minus after the clean stream; `corr. residual` is the same after the corrective stream. Acceptance is a pair of rates; n/a means no decision was recorded, not zero.

| mode | transfer Δ (impulse) | transfer Δ (hold) | transfer Δ (release) | forgetting Δ | poison harm | corr. residual | revert ok | accepted-good | refused-bad | tokens consumed | tokens measured |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | yes | n/a (n=0) | n/a (n=0) | 768 | 25600 |
| continued | -0.0616 | +0.1587 | -0.0087 | -0.0068 | +0.0189 | +0.0107 | yes | +1.0000 (n=2) | +0.0000 (n=1) | 768 | 25600 |
| in_context | +0.0001 | -0.0006 | +0.0001 | -0.0004 | +0.0000 | +0.0001 | yes | +1.0000 (n=2) | +0.0000 (n=1) | 768 | 107520 |

Every measurement is taken from a fresh state with the stream removed. Spec: docs/superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md
