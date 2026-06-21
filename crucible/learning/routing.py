"""Offline policy improvement from logged episodes.

Each logged episode records the strategy that was actually used and the reward
it earned. From those logs we estimate, per task family, which strategy has the
highest mean reward, and propose routing each family to its empirical best.

This is a deliberately simple, honest off-policy estimate. Its limitation is
stated up front: we only observe the reward of the strategy that *was* chosen,
never the counterfactual reward of the others on the same episode. So the
estimate is confounded by whatever drove strategy selection originally, and is
only trustworthy with enough support per (family, strategy) cell. The champion
gate (see gate.py) exists precisely because this estimate can be wrong.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_episodes(path: str | Path) -> list[dict]:
    """Load curated replay sample (one JSON object per line)."""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def baseline_metrics(episodes: list[dict]) -> dict:
    """Status-quo policy: the reward the logged behavior actually achieved."""
    per_family: dict[str, float] = {}
    by_fam: dict[str, list[float]] = defaultdict(list)
    for e in episodes:
        by_fam[e["task_type"]].append(e["reward"])
    for fam, rewards in by_fam.items():
        per_family[fam] = mean(rewards)
    overall = mean([e["reward"] for e in episodes])
    return {"overall": overall, "per_family": per_family}


def family_strategy_table(
    episodes: list[dict], *, min_support: int = 3
) -> dict[str, dict[str, tuple[float, int]]]:
    """{family: {strategy: (mean_reward, n)}} for cells with enough support."""
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        cells[e["task_type"]][e["strategy_used"]].append(e["reward"])
    table: dict[str, dict[str, tuple[float, int]]] = {}
    for fam, strats in cells.items():
        table[fam] = {
            s: (mean(rs), len(rs)) for s, rs in strats.items() if len(rs) >= min_support
        }
    return table


def propose_routing(
    episodes: list[dict], *, min_support: int = 3, pick: str = "best"
) -> dict[str, str]:
    """Choose one strategy per family from these episodes.

    pick="best"  -> each family's highest-mean-reward strategy
    pick="worst" -> lowest (to demonstrate the gate rejecting a regression)

    Families without enough support are omitted (no confident proposal), so the
    routing never invents signal where the data is too thin.
    """
    table = family_strategy_table(episodes, min_support=min_support)
    routing: dict[str, str] = {}
    for fam, cells in table.items():
        if not cells:
            continue
        ranked = sorted(cells.items(), key=lambda kv: kv[1][0])
        chosen, _ = ranked[0] if pick == "worst" else ranked[-1]
        routing[fam] = chosen
    return routing


def evaluate_routing(episodes: list[dict], routing: dict[str, str]) -> dict:
    """Estimate the reward of a fixed routing on a (possibly held-out) set.

    For each family routed to strategy `s`, the estimate is the mean reward of
    the episodes in that family that actually used `s`. When the eval set has no
    such episodes (the routing isn't represented here), we fall back to the
    family's status-quo mean — we never fabricate a number. This is what makes
    held-out evaluation honest: routings that don't generalize get no credit.
    """
    by_fam: dict[str, list[float]] = defaultdict(list)
    by_fam_strat: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in episodes:
        by_fam[e["task_type"]].append(e["reward"])
        by_fam_strat[(e["task_type"], e["strategy_used"])].append(e["reward"])

    per_family: dict[str, float] = {}
    for fam, rewards in by_fam.items():
        chosen = routing.get(fam)
        cell = by_fam_strat.get((fam, chosen)) if chosen else None
        per_family[fam] = mean(cell) if cell else mean(rewards)

    total = sum(len(v) for v in by_fam.values())
    overall = sum(per_family[f] * len(by_fam[f]) / total for f in per_family)
    return {"overall": overall, "per_family": per_family}


def propose_policy(
    episodes: list[dict], *, min_support: int = 3, pick: str = "best"
) -> tuple[dict[str, str], dict]:
    """In-sample convenience: propose a routing and score it on the same set.

    Kept for simple callers; prefer split_by_family + propose_routing +
    evaluate_routing for an honest (held-out) estimate.
    """
    routing = propose_routing(episodes, min_support=min_support, pick=pick)
    return routing, evaluate_routing(episodes, routing)


def split_by_family(
    episodes: list[dict], *, test_frac: float = 0.3, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Stratified train/test split that keeps each family's proportion."""
    rng = random.Random(seed)
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        by_fam[e["task_type"]].append(e)
    train: list[dict] = []
    test: list[dict] = []
    for rows in by_fam.values():
        rows = rows[:]
        rng.shuffle(rows)
        n_test = int(round(len(rows) * test_frac))
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
    return train, test
