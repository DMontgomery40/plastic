import torch

from plastic.harness.stats import Cusum, SignalHistory, quantile_threshold, robust_z


def test_robust_z_basics():
    ref = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.0, 1.0]
    assert abs(robust_z(1.0, ref)) < 1e-9
    assert robust_z(5.0, ref) > 20
    assert robust_z(1.0, ref[:5]) is None


def test_cusum_silent_on_noise_and_alarms_on_shift():
    g = torch.Generator().manual_seed(0)
    # in-control average run length for a two-sided CUSUM with k=0.5, h=5 is roughly 230,
    # so 2000 standard-normal draws give about 9 false alarms; a +1 sigma shift is
    # caught in about 10 steps on average
    noise = torch.randn(2000, generator=g).tolist()
    c = Cusum(k=0.5, h=5.0)
    alarms = sum(c.update(z) for z in noise)
    assert alarms <= 25, alarms
    delays = []
    for _ in range(20):
        c = Cusum(k=0.5, h=5.0)
        shifted = (torch.randn(200, generator=g) + 1.0).tolist()
        first = next((i for i, z in enumerate(shifted) if c.update(z)), None)
        assert first is not None
        delays.append(first)
    assert sum(delays) / len(delays) < 25


def test_cusum_state_roundtrip():
    c = Cusum(k=0.4, h=3.0)
    for z in (1.0, 1.0, 1.0):
        c.update(z)
    d = c.state()
    back = Cusum.from_state(d)
    assert back.s_hi == c.s_hi and back.k == 0.4 and back.h == 3.0


def test_quantile_threshold_controls_false_positives():
    g = torch.Generator().manual_seed(1)
    benign = torch.randn(1000, generator=g).tolist()
    thr = quantile_threshold(benign, 0.01)
    fresh = torch.randn(10000, generator=g)
    fpr = float((fresh > thr).float().mean())
    assert 0.003 < fpr < 0.03, fpr
    shifted = torch.randn(1000, generator=g) + 5.0
    assert float((shifted > thr).float().mean()) > 0.99
    low = quantile_threshold(benign, 0.01, side="lower")
    assert low < thr


def test_signal_history_bounded():
    h = SignalHistory(maxlen=3)
    for x in range(5):
        h.append(x)
    assert h.values() == [2.0, 3.0, 4.0] and len(h) == 3
