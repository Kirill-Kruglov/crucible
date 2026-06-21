"""Offline self-improvement loop — runnable in seconds, no LLM / GPU / network.

What makes this more than a tutorial: the loop does not trust reward. It checks
every candidate policy against a second signal that reward cannot fake — the
**reproduce rate** (did oracle-passing episodes actually hold up on re-run?).

    logged episodes ─► split ─► propose on train ─► held-out reward + integrity ─► gate

The demo shows, in order:
  1. the raw reward-hacking signature already present in the logs;
  2. an honest champion judged on held-out reward AND integrity;
  3. a controlled reward-hacking policy that a naive reward-only gate would adopt
     but the integrity-aware gate rejects;
  4. multi-seed stability — the verdict with a variance, not a lucky single split.

Run:
    python -m demo.self_improve
    python -m demo.self_improve --chart   # also write a PNG (needs matplotlib)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crucible.learning.gate import champion_gate
from crucible.learning.integrity import (
    gamed_rate,
    reproduce_rate,
    reward_hack_routing,
    routed_reproduce_rate,
)
from crucible.learning.loop import multiseed, run_gate
from crucible.learning.routing import (
    baseline_metrics,
    evaluate_routing,
    load_episodes,
    split_by_family,
)

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "replay_sample.jsonl"


def _bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 22) -> str:
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    fill = round(frac * width)
    return "█" * fill + "·" * (width - fill)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline self-improvement loop demo")
    ap.add_argument("--sample", type=Path, default=SAMPLE)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=7, help="number of seeds for multi-seed run")
    ap.add_argument("--chart", type=Path, nargs="?",
                    const=REPO / "docs" / "assets" / "before_after.png",
                    help="write summary PNG (needs matplotlib)")
    args = ap.parse_args()

    episodes = load_episodes(args.sample)

    # 1) the phenomenon: reward pays out for non-reproducible wins
    gr = gamed_rate(episodes)
    print(f"[1/4] in the logs: {_pct(gr)} of oracle-passing episodes did NOT reproduce")
    print(f"      (reward still paid out — optimizing reward alone amplifies this)")

    # 2) honest champion: judged on held-out reward AND integrity
    run = run_gate(episodes, seed=args.seed, test_frac=args.test_frac, min_support=args.min_support)
    print(f"[2/4] champion (seed {args.seed}):")
    print(f"      reward    {run.reward_baseline:.3f} -> {run.reward_candidate:.3f} held-out"
          f"  (in-sample {run.reward_in_sample:.3f})")
    print(f"      integrity {_pct(run.integrity_baseline)} -> {_pct(run.integrity_candidate)} reproduce rate")
    print(f"      {run.decision.render()}")

    # 3) controlled reward-hacking policy: reward-only gate vs integrity-aware gate
    train, test = split_by_family(episodes, test_frac=args.test_frac, seed=args.seed)
    hack = reward_hack_routing(train, min_support=args.min_support)
    base = baseline_metrics(test)
    hack_reward = evaluate_routing(test, hack)
    hack_integ = routed_reproduce_rate(test, hack)
    naive = champion_gate(base, hack_reward)  # reward-only
    aware = champion_gate(base, hack_reward,
                          baseline_integrity=reproduce_rate(test),
                          candidate_integrity=hack_integ)
    print(f"[3/4] reward-hacking probe (chases reward into non-reproducible wins):")
    print(f"      reward-only gate : {'ADOPT' if naive.passed else 'reject'}  ({naive.reason})")
    print(f"      integrity gate   : {'adopt' if aware.passed else 'REJECT'}  ({aware.reason})")

    # 4) multi-seed: the verdict with a variance
    ms = multiseed(episodes, seeds=list(range(args.seeds)),
                   test_frac=args.test_frac, min_support=args.min_support)
    print(f"[4/4] multi-seed ({ms['n']} splits): adopted {ms['adopt_rate'] * 100:.0f}% of the time")
    print(f"      reward Δ    {ms['reward_delta_mean']:+.3f} ± {ms['reward_delta_std']:.3f}")
    if ms["integrity_delta_mean"] is not None:
        print(f"      integrity Δ {ms['integrity_delta_mean']:+.3f} ± {ms['integrity_delta_std']:.3f}")

    lo = min(base["overall"], hack_reward["overall"]) - 0.05
    hi = max(run.reward_candidate, base["overall"]) + 0.05
    print()
    print("  reward (held-out)                          integrity (reproduce rate)")
    print(f"    baseline    {_bar(run.reward_baseline, lo, hi)}  {run.reward_baseline:.3f}"
          f"      baseline  {_pct(run.integrity_baseline)}")
    print(f"    champion    {_bar(run.reward_candidate, lo, hi)}  {run.reward_candidate:.3f}"
          f"      champion  {_pct(run.integrity_candidate)}  -> {'ADOPTED' if run.decision.passed else 'REJECTED'}")
    print(f"    reward-hack {_bar(hack_reward['overall'], lo, hi)}  {hack_reward['overall']:.3f}"
          f"      hack      {_pct(hack_integ)}  -> REJECTED on integrity ✓")

    if args.chart:
        _write_chart(run, hack_reward, hack_integ, reproduce_rate(test), ms, args.chart)


def _write_chart(run, hack_reward, hack_integ, base_integ, ms, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("      (chart skipped: matplotlib not installed — pip install '.[charts]')")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))

    labels = ["baseline", "champion\n(adopted)" if run.decision.passed else "champion",
              "reward-hack\n(rejected)"]
    rewards = [run.reward_baseline, run.reward_candidate, hack_reward["overall"]]
    ax1.bar(labels, rewards, color=["#888", "#2a9d8f", "#e76f51"])
    ax1.axhline(run.reward_baseline, color="#888", lw=1, ls="--", zorder=0)
    ax1.set_title(f"Reward (held-out)   adopted {ms['adopt_rate']*100:.0f}% over {ms['n']} seeds")
    ax1.set_ylabel("mean reward")
    for i, v in enumerate(rewards):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    integ = [base_integ or 0, run.integrity_candidate or 0, hack_integ or 0]
    ax2.bar(labels, integ, color=["#888", "#2a9d8f", "#e76f51"])
    ax2.set_title("Integrity (reproduce rate) — what reward can't fake")
    ax2.set_ylabel("reproduce rate")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(integ):
        ax2.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"      chart -> {path}")


if __name__ == "__main__":
    main()
