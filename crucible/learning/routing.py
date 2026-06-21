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


def propose_policy(
    episodes: list[dict], *, min_support: int = 3, pick: str = "best"
) -> tuple[dict[str, str], dict]:
    """Propose a per-family strategy routing and estimate its metrics.

    pick="best"  -> route each family to its highest-mean-reward strategy
    pick="worst" -> route to lowest (used to demonstrate the gate rejecting a regression)

    Returns (routing, metrics). Families without enough support keep the
    baseline (no confident proposal), so the estimate never invents signal.
    """
    table = family_strategy_table(episodes, min_support=min_support)
    base = baseline_metrics(episodes)
    routing: dict[str, str] = {}
    per_family: dict[str, float] = {}

    for fam, base_r in base["per_family"].items():
        cells = table.get(fam, {})
        if not cells:
            per_family[fam] = base_r  # no confident proposal -> keep status quo
            continue
        ranked = sorted(cells.items(), key=lambda kv: kv[1][0])
        chosen, (chosen_r, _n) = ranked[0] if pick == "worst" else ranked[-1]
        routing[fam] = chosen
        per_family[fam] = chosen_r

    # family weights = share of episodes in that family
    counts: dict[str, int] = defaultdict(int)
    for e in episodes:
        counts[e["task_type"]] += 1
    total = sum(counts.values())
    overall = sum(per_family[f] * counts[f] / total for f in per_family)

    return routing, {"overall": overall, "per_family": per_family}
