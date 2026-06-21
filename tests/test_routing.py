from pathlib import Path

from crucible.learning.routing import (
    baseline_metrics,
    evaluate_routing,
    family_strategy_table,
    load_episodes,
    propose_policy,
    propose_routing,
    split_by_family,
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


def test_split_is_stratified_and_disjoint():
    eps = _episodes() + [
        {"task_type": "b", "strategy_used": "z", "reward": 0.5, "features": [], "success": True}
        for _ in range(10)
    ]
    train, test = split_by_family(eps, test_frac=0.3, seed=1)
    assert len(train) + len(test) == len(eps)
    # both families represented in the test split (stratification)
    assert {e["task_type"] for e in test} == {"a", "b"}


def test_heldout_falls_back_when_routing_unrepresented():
    # routing chooses strategy "y", but the eval set only has strategy "x"
    eval_set = [
        {"task_type": "a", "strategy_used": "x", "reward": 0.2, "features": [], "success": False},
        {"task_type": "a", "strategy_used": "x", "reward": 0.4, "features": [], "success": False},
    ]
    metrics = evaluate_routing(eval_set, {"a": "y"})
    # no credit invented: falls back to the family's status-quo mean (0.3)
    assert abs(metrics["per_family"]["a"] - 0.3) < 1e-9


def test_heldout_estimate_is_not_more_optimistic_than_in_sample_on_average():
    eps = load_episodes(SAMPLE)
    train, test = split_by_family(eps, test_frac=0.3, seed=0)
    routing = propose_routing(train, min_support=3, pick="best")
    in_sample = evaluate_routing(train, routing)["overall"]
    held_out = evaluate_routing(test, routing)["overall"]
    # the whole point of held-out eval: it should not flatter the candidate
    assert held_out <= in_sample + 1e-6


def test_curated_sample_loads_and_is_well_formed():
    eps = load_episodes(SAMPLE)
    assert len(eps) > 50
    for e in eps[:20]:
        assert {"features", "reward", "strategy_used", "task_type"} <= e.keys()
        assert len(e["features"]) == 193
