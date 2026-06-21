from pathlib import Path

from crucible.learning.routing import (
    baseline_metrics,
    family_strategy_table,
    load_episodes,
    propose_policy,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "replay_sample.jsonl"


def _episodes():
    return [
        {"task_type": "a", "strategy_used": "x", "reward": 0.2, "features": [], "success": False},
        {"task_type": "a", "strategy_used": "x", "reward": 0.3, "features": [], "success": False},
        {"task_type": "a", "strategy_used": "x", "reward": 0.1, "features": [], "success": False},
        {"task_type": "a", "strategy_used": "y", "reward": 0.8, "features": [], "success": True},
        {"task_type": "a", "strategy_used": "y", "reward": 0.9, "features": [], "success": True},
        {"task_type": "a", "strategy_used": "y", "reward": 0.7, "features": [], "success": True},
    ]


def test_baseline_is_mean_reward():
    eps = _episodes()
    base = baseline_metrics(eps)
    assert abs(base["overall"] - 0.5) < 1e-9
    assert "a" in base["per_family"]


def test_best_policy_beats_baseline_and_worst_is_below():
    eps = _episodes()
    base = baseline_metrics(eps)
    _, best = propose_policy(eps, min_support=3, pick="best")
    _, worst = propose_policy(eps, min_support=3, pick="worst")
    assert best["overall"] > base["overall"]
    assert worst["overall"] < base["overall"]


def test_low_support_cell_keeps_status_quo():
    eps = _episodes() + [{"task_type": "b", "strategy_used": "z", "reward": 0.99, "features": [], "success": True}]
    # family "b" has only 1 sample -> below min_support -> no confident proposal
    routing, _ = propose_policy(eps, min_support=3, pick="best")
    assert "b" not in routing


def test_table_filters_by_support():
    eps = _episodes()
    table = family_strategy_table(eps, min_support=3)
    assert set(table["a"].keys()) == {"x", "y"}


def test_curated_sample_loads_and_is_well_formed():
    eps = load_episodes(SAMPLE)
    assert len(eps) > 50
    for e in eps[:20]:
        assert {"features", "reward", "strategy_used", "task_type"} <= e.keys()
        assert len(e["features"]) == 193
