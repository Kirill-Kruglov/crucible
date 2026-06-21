"""One turn of the loop, and the multi-seed version that quantifies its variance.

A single train/test split gives one verdict. Whether that verdict is *stable* is
a separate question — a candidate that wins on one split and loses on the next
has not really earned adoption. `multiseed` runs the whole propose→evaluate→gate
cycle over several stratified splits and reports the distribution, so the gate's
decision comes with a confidence, not a single lucky number.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from crucible.learning.gate import GateDecision, champion_gate
from crucible.learning.integrity import reproduce_rate, routed_reproduce_rate
from crucible.learning.routing import (
    baseline_metrics,
    evaluate_routing,
    propose_routing,
    split_by_family,
)


@dataclass
class GateRun:
    seed: int
    reward_baseline: float
    reward_in_sample: float
    reward_candidate: float
    integrity_baseline: float | None
    integrity_candidate: float | None
    decision: GateDecision


def run_gate(
    episodes: list[dict],
    *,
    seed: int = 0,
    test_frac: float = 0.3,
    min_support: int = 3,
    pick: str = "best",
) -> GateRun:
    """Propose a routing on a train split; judge it on held-out reward AND integrity."""
    train, test = split_by_family(episodes, test_frac=test_frac, seed=seed)
    routing = propose_routing(train, min_support=min_support, pick=pick)

    base = baseline_metrics(test)
    cand = evaluate_routing(test, routing)
    integ_base = reproduce_rate(test)
    integ_cand = routed_reproduce_rate(test, routing)

    decision = champion_gate(
        base,
        cand,
        baseline_integrity=integ_base,
        candidate_integrity=integ_cand,
    )
    return GateRun(
        seed=seed,
        reward_baseline=base["overall"],
        reward_in_sample=evaluate_routing(train, routing)["overall"],
        reward_candidate=cand["overall"],
        integrity_baseline=integ_base,
        integrity_candidate=integ_cand,
        decision=decision,
    )


def multiseed(
    episodes: list[dict],
    *,
    seeds: list[int] | None = None,
    test_frac: float = 0.3,
    min_support: int = 3,
    pick: str = "best",
) -> dict:
    """Run the gate over several splits; summarize the distribution of outcomes."""
    seeds = seeds if seeds is not None else list(range(5))
    runs = [
        run_gate(episodes, seed=s, test_frac=test_frac, min_support=min_support, pick=pick)
        for s in seeds
    ]
    reward_deltas = [r.reward_candidate - r.reward_baseline for r in runs]
    integ_deltas = [
        r.integrity_candidate - r.integrity_baseline
        for r in runs
        if r.integrity_candidate is not None and r.integrity_baseline is not None
    ]
    adopted = [r for r in runs if r.decision.passed]
    return {
        "runs": runs,
        "n": len(runs),
        "adopt_rate": len(adopted) / len(runs),
        "reward_delta_mean": mean(reward_deltas),
        "reward_delta_std": pstdev(reward_deltas) if len(reward_deltas) > 1 else 0.0,
        "integrity_delta_mean": mean(integ_deltas) if integ_deltas else None,
        "integrity_delta_std": (pstdev(integ_deltas) if len(integ_deltas) > 1 else 0.0)
        if integ_deltas
        else None,
    }
