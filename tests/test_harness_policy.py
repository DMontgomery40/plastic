import pytest

from plastic.harness.config import HarnessConfig
from plastic.harness.policy import decide
from plastic.harness.signals import ChunkSignals


def _sig(**kw) -> ChunkSignals:
    base = dict(
        pos_start=0, pos_end=8, n_tokens=8, chunk_loss=3.0, surprise_mean=1.0, surprise_max=2.0,
        beta_mean=0.5, alpha_mean=0.98, write_norm_sum=1.0, delta_norm=0.5,
    )
    base.update(kw)
    return ChunkSignals(**base)


def test_config_roundtrip():
    cfg = HarnessConfig(budget_chunk=1.5, log_only=True)
    assert HarnessConfig.from_json(cfg.to_json()) == cfg
    with pytest.raises(ValueError):
        HarnessConfig.from_dict({"nope": 1})


@pytest.mark.parametrize(
    "kw,thresholds,expected,reason_fragment",
    [
        ({}, None, "commit", None),
        ({"canary_delta_coherence": 0.9}, None, "rollback", "canary_coherence"),
        ({"canary_delta_poison": -0.9}, None, "rollback", "canary_poison"),
        ({"chunk_loss": 9.0}, {"chunk_loss": 5.0}, "rollback", "chunk_loss(9>5)"),
        ({"z": {"surprise_mean": 7.0}}, None, "rollback", "surprise_mean_z"),
        ({"z": {"surprise_mean": 4.0}}, None, "scale", "surprise_mean_z"),
        ({"cusum_alarm": True}, None, "rollback", "cusum_alarm"),
        ({"fisher_drift": 10.0}, None, "commit", None),
        ({"canary_alignment": 0.3}, None, "project", "canary_alignment"),
        ({"delta_norm": 4.0}, None, "scale", "budget_chunk"),
    ],
)
def test_decision_matrix(kw, thresholds, expected, reason_fragment):
    cfg = HarnessConfig(budget_chunk=2.0)
    d = decide(_sig(**kw), cfg, thresholds, read_only=False)
    assert d.kind == expected, d
    if reason_fragment:
        assert any(reason_fragment in r for r in d.reasons), d.reasons


def test_budget_scale_factor_and_combination():
    cfg = HarnessConfig(budget_chunk=1.0)
    d = decide(_sig(delta_norm=4.0), cfg, None, read_only=False)
    assert d.kind == "scale" and abs(d.scale - 0.25) < 1e-9
    d = decide(_sig(delta_norm=4.0, z={"chunk_loss": 4.0}), cfg, None, read_only=False)
    assert d.kind == "scale" and d.scale == min(0.25, cfg.scale_factor)


def test_read_only_dominates_everything():
    cfg = HarnessConfig()
    d = decide(_sig(canary_delta_coherence=5.0, cusum_alarm=True), cfg, None, read_only=True)
    assert d.kind == "readonly"


def test_log_only_never_blocks_but_records():
    cfg = HarnessConfig(log_only=True)
    d = decide(_sig(canary_delta_coherence=5.0, canary_alignment=0.5, z={"chunk_loss": 4.0}), cfg, None, read_only=False)
    assert d.kind == "commit"
    joined = " ".join(d.reasons)
    assert "would_rollback:canary_coherence" in joined and "would_project" in joined and "would_scale" in joined


def test_fisher_drift_cap():
    cfg = HarnessConfig(fisher_drift_max=1.0)
    d = decide(_sig(fisher_drift=2.0), cfg, None, read_only=False)
    assert d.kind == "rollback" and any("fisher_drift" in r for r in d.reasons)


def test_disabled_layers():
    cfg = HarnessConfig(enable_rollback=False, enable_stats=False, enable_projection=False, enable_budget=False)
    d = decide(_sig(canary_delta_coherence=5.0, cusum_alarm=True, canary_alignment=0.9, delta_norm=100.0), cfg, None, read_only=False)
    assert d.kind == "commit"


def test_decide_and_the_final_candidate_recheck_share_one_constraint_function():
    """The runner rechecks a scaled or projected candidate with the same limits decide() applied to the proposal
    (external review of 6cf4457, finding 2); both go through constraint_violations, calibrated thresholds first."""
    from plastic.harness.policy import constraint_violations

    cfg = HarnessConfig(canary_delta_max=0.5, poison_delta_min=-0.5, fisher_drift_max=2.0)
    assert constraint_violations(0.4, -0.4, 1.0, cfg, None) == []
    v = constraint_violations(0.6, -0.6, 3.0, cfg, None)
    assert [x.split("(")[0] for x in v] == ["canary_coherence", "canary_poison", "fisher_drift"]
    assert constraint_violations(0.4, 0.0, None, cfg, {"canary_delta_coherence": 0.3})[0].startswith("canary_coherence")
    assert constraint_violations(None, None, None, cfg, None) == []
