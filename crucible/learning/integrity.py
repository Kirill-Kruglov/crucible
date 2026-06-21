"""An integrity signal that is *orthogonal to reward* — the project's edge.

Reward can be gamed. An agent can satisfy a task's oracle (the automated checker
that grants reward) without actually producing a result that holds up. In the
logs this leaves a signature: ``oracle_pass = True`` but ``repro_pass = False`` —
the check passed, yet re-running the produced solution did not reproduce the
result. Reward still pays out for these episodes.

A loop that optimizes reward alone will happily amplify whatever earns reward,
including these non-reproducible "wins". So `crucible` tracks a second signal —
the **reproduce rate** — and the champion gate refuses to adopt a policy that
raises reward while eroding reproducibility. This is the anti-Goodhart guard:
be suspicious of reward when the process behind it does not hold up.

(See docs/LIMITATIONS.md and the parent project's GOODHART_SELF_METRICS notes.)
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean


def is_checkable(e: dict) -> bool:
    """True when the episode carries both an oracle verdict and a repro check."""
    return e.get("oracle_pass") is not None and e.get("repro_pass") is not None


def is_gamed(e: dict) -> bool:
    """Oracle granted reward, but the result did not reproduce."""
    return bool(e.get("oracle_pass")) and e.get("repro_pass") is False


def reproduce_rate(episodes: list[dict]) -> float | None:
    """Fraction of *checkable* episodes whose oracle-pass also reproduced.

    Returns None when no episode is checkable — we never fabricate a verdict.
    """
    checkable = [e for e in episodes if is_checkable(e)]
    passed = [e for e in checkable if e.get("oracle_pass")]
    if not passed:
        return None
    reproduced = [e for e in passed if e.get("repro_pass")]
    return len(reproduced) / len(passed)


def gamed_rate(episodes: list[dict]) -> float | None:
    """Fraction of oracle-passing episodes that did NOT reproduce (1 - reproduce_rate)."""
    rr = reproduce_rate(episodes)
    return None if rr is None else 1.0 - rr


def routed_reproduce_rate(episodes: list[dict], routing: dict[str, str]) -> float | None:
    """Reproduce rate a routing would inherit, estimated per family then pooled.

    For each family routed to strategy ``s`` we look only at that family's
    episodes that used ``s`` (mirroring how reward is evaluated). Families with
    no checkable, oracle-passing episodes under the routing contribute nothing —
    no integrity credit is invented where the data can't support it.
    """
    by_fam_strat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        by_fam_strat[(e["task_type"], e["strategy_used"])].append(e)
        by_fam[e["task_type"]].append(e)

    passed = reproduced = 0
    for fam in by_fam:
        chosen = routing.get(fam)
        pool = by_fam_strat.get((fam, chosen), []) if chosen else by_fam[fam]
        for e in pool:
            if is_checkable(e) and e.get("oracle_pass"):
                passed += 1
                reproduced += int(bool(e.get("repro_pass")))
    return reproduced / passed if passed else None


def reward_hack_routing(
    episodes: list[dict], *, min_support: int = 3
) -> dict[str, str]:
    """An adversarial routing that chases reward into non-reproducible territory.

    Per family, among strategies with enough support, pick the one that maximizes
    reward while having the *lowest* reproduce rate — i.e. a policy that learns to
    earn reward via wins that don't hold up. This is a controlled probe: a
    reward-only gate would happily adopt it; the integrity-aware gate should not.
    """
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in episodes:
        by_cell[(e["task_type"], e["strategy_used"])].append(e)

    fam_strats: dict[str, list[str]] = defaultdict(list)
    for (fam, strat), rows in by_cell.items():
        if len(rows) >= min_support:
            fam_strats[fam].append(strat)

    routing: dict[str, str] = {}
    for fam, strats in fam_strats.items():
        def score(s: str) -> tuple[float, float]:
            rows = by_cell[(fam, s)]
            rr = reproduce_rate(rows)
            # high reward first, then low reproducibility (tie-break toward hacking)
            return (mean(r["reward"] for r in rows), -(rr if rr is not None else 1.0))
        routing[fam] = max(strats, key=score)
    return routing
