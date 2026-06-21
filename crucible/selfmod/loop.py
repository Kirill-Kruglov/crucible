"""Self-modification under the integrity gate.

A proposer suggests edits to the agent's policy. Each edit is evaluated on the
task suite and judged by the *same* champion gate used elsewhere in `crucible` —
on reward AND reproduce rate. An edit is adopted only if it improves reward
without eroding reproducibility; otherwise it is reverted. This is the bounded,
honest version of "an agent rewrites itself to get better": it can only keep
changes that survive an integrity check it cannot game.
"""

from __future__ import annotations

from crucible.learning.gate import GateDecision, champion_gate
from crucible.learning.integrity import reproduce_rate
from crucible.learning.routing import baseline_metrics
from crucible.selfmod.world import (
    DEFAULT_POLICY,
    TASKS,
    Edit,
    oracle,
    repro,
    reward,
    solve,
)

# The scripted proposer: the first edit is a genuine win, the second games the oracle.
EDITS: list[Edit] = [
    Edit("trim the explanation", {"verbosity": 1}),
    Edit("answer from memory (skip the derivation)", {"method": "lookup", "verbosity": 0}),
]


def evaluate_policy(policy: dict, tasks: list[dict] | None = None) -> tuple[dict, float | None]:
    """Run the suite under `policy`; return (reward metrics, reproduce rate).

    Reward metrics reuse `routing.baseline_metrics`; the integrity signal reuses
    `integrity.reproduce_rate` — the toy is judged by the same machinery as the
    log-based loop, not a bespoke scorer.
    """
    episodes = []
    for t in tasks or TASKS:
        out = solve(t, policy)
        episodes.append(
            {
                "task_type": t["kind"],
                "reward": reward(t, out),
                "oracle_pass": oracle(t, out),
                "repro_pass": repro(t, out),
            }
        )
    return baseline_metrics(episodes), reproduce_rate(episodes)


def try_edit(policy: dict, edit: Edit) -> tuple[GateDecision, dict]:
    """Evaluate `edit` against `policy`; adopt it only if the gate passes."""
    candidate = {**policy, **edit.delta}
    base_r, base_i = evaluate_policy(policy)
    cand_r, cand_i = evaluate_policy(candidate)
    decision = champion_gate(
        base_r,
        cand_r,
        baseline_integrity=base_i,
        candidate_integrity=cand_i,
    )
    return decision, (candidate if decision.passed else policy)


def run(edits: list[Edit] | None = None, start: dict | None = None):
    """Apply edits sequentially, keeping only those the gate accepts.

    Yields one record per edit:
        (edit, decision, candidate_metrics, policy_in_effect_after)
    where candidate_metrics = (reward_metrics, reproduce_rate) for what the edit
    *would* produce — so callers can show what was proposed and whether it stuck.
    """
    policy = dict(start or DEFAULT_POLICY)
    for edit in edits if edits is not None else EDITS:
        candidate = {**policy, **edit.delta}
        cand_metrics = evaluate_policy(candidate)
        decision, policy = try_edit(policy, edit)
        yield edit, decision, cand_metrics, policy
