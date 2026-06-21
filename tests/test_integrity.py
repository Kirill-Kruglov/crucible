from crucible.learning.gate import champion_gate
from crucible.learning.integrity import (
    gamed_rate,
    is_gamed,
    reproduce_rate,
    routed_reproduce_rate,
)


def _eps():
    # family "a": strategy "good" reproduces, strategy "hack" passes oracle but never reproduces
    out = []
    for _ in range(4):
        out.append({"task_type": "a", "strategy_used": "good", "reward": 0.6,
                    "oracle_pass": True, "repro_pass": True})
    for _ in range(4):
        out.append({"task_type": "a", "strategy_used": "hack", "reward": 0.7,
                    "oracle_pass": True, "repro_pass": False})
    return out


def test_gamed_episode_detection():
    assert is_gamed({"oracle_pass": True, "repro_pass": False})
    assert not is_gamed({"oracle_pass": True, "repro_pass": True})
    assert not is_gamed({"oracle_pass": False, "repro_pass": False})


def test_reproduce_and_gamed_rate_complement():
    eps = _eps()
    rr = reproduce_rate(eps)
    assert abs(rr - 0.5) < 1e-9  # 4 of 8 oracle-passes reproduced
    assert abs(gamed_rate(eps) - 0.5) < 1e-9


def test_no_credit_invented_when_unverifiable():
    eps = [{"task_type": "a", "strategy_used": "x", "reward": 0.5}]  # no oracle/repro fields
    assert reproduce_rate(eps) is None
    assert routed_reproduce_rate(eps, {"a": "x"}) is None


def test_routing_to_hack_strategy_lowers_reproduce_rate():
    eps = _eps()
    assert routed_reproduce_rate(eps, {"a": "good"}) == 1.0
    assert routed_reproduce_rate(eps, {"a": "hack"}) == 0.0


def test_integrity_gate_rejects_reward_gain_that_erodes_reproducibility():
    # reward improves, no family reward regression, but reproducibility collapses
    base = {"overall": 0.6, "per_family": {"a": 0.6}}
    cand = {"overall": 0.7, "per_family": {"a": 0.7}}
    d = champion_gate(base, cand, baseline_integrity=1.0, candidate_integrity=0.0)
    assert not d.passed
    assert "integrity" in d.reason
    # the same candidate passes a reward-only gate (no integrity supplied)
    assert champion_gate(base, cand).passed
