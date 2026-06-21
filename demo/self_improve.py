"""Offline self-improvement loop — runnable in seconds, no LLM / GPU / network.

Story this demo tells:

    logged episodes  ->  estimate a better per-family routing  ->  champion gate

The headline is NOT "the agent got smarter on its own". It is the discipline:
a candidate policy is proposed from real logs, then a conservative gate decides
whether to adopt it. The same gate *rejects* a deliberately-regressed candidate,
which is the point — an honest loop must be able to say "no, this is worse".

Run:
    python -m demo.self_improve
    python -m demo.self_improve --chart   # also write a PNG (needs matplotlib)

Everything except the optional --chart and the optional learned-router section
runs on the Python standard library alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crucible.learning.routing import (
    baseline_metrics,
    evaluate_routing,
    load_episodes,
    propose_routing,
    split_by_family,
)
from crucible.learning.gate import champion_gate

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "replay_sample.jsonl"


def _bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 24) -> str:
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    fill = round(frac * width)
    return "█" * fill + "·" * (width - fill)


def _learned_router_accuracy(episodes: list[dict]) -> str | None:
    """Train the project's real PolicyNetwork as a learned router (optional).

    Reward-weighted classification of (episode features -> strategy used), the
    same objective the full system uses. Reported as train accuracy only, with
    an explicit off-policy caveat — it is a sanity signal, not a guarantee.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    from crucible.learning.policy import PolicyNetwork

    labels = sorted({e["strategy_used"] for e in episodes})
    idx = {s: i for i, s in enumerate(labels)}
    X = torch.tensor([e["features"] for e in episodes], dtype=torch.float32)
    y = torch.tensor([idx[e["strategy_used"]] for e in episodes], dtype=torch.long)
    r = torch.tensor([e["reward"] for e in episodes], dtype=torch.float32)
    w = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else torch.ones_like(r)

    model = PolicyNetwork(X.shape[1], len(labels), labels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    for _ in range(120):
        logits = model(X)
        loss = (loss_fn(logits, y) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (model(X).argmax(1) == y).float().mean().item()
    return f"{acc:.2f} over {len(labels)} strategies (train fit; off-policy — not a counterfactual guarantee)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline self-improvement loop demo")
    ap.add_argument("--sample", type=Path, default=SAMPLE)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chart", type=Path, nargs="?", const=REPO / "docs" / "assets" / "before_after.png",
                    help="write before/after PNG (needs matplotlib)")
    args = ap.parse_args()

    episodes = load_episodes(args.sample)
    train, test = split_by_family(episodes, test_frac=args.test_frac, seed=args.seed)
    base = baseline_metrics(test)  # status quo, measured on held-out episodes
    print(f"[1/5] loaded {len(episodes)} episodes -> train {len(train)} / held-out {len(test)}")
    print(f"[2/5] baseline mean reward (held-out): {base['overall']:.3f}")

    # --- propose on TRAIN, then estimate honestly on the held-out set ---
    routing = propose_routing(train, min_support=args.min_support, pick="best")
    in_sample = evaluate_routing(train, routing)   # optimistic (fit the data it learned from)
    champ = evaluate_routing(test, routing)        # what the gate actually trusts
    decision = champion_gate(base, champ)
    print(f"[3/5] proposed champion:  in-sample {in_sample['overall']:.3f} (optimistic)"
          f"  ->  held-out {champ['overall']:.3f}")
    print(f"      {decision.render()}")
    if routing:
        shown = ", ".join(f"{f}->{s}" for f, s in sorted(routing.items())[:4])
        print(f"      routing (sample): {shown}{' ...' if len(routing) > 4 else ''}")

    # --- regression control: the gate must reject a worse candidate ---
    bad_routing = propose_routing(train, min_support=args.min_support, pick="worst")
    regressed = evaluate_routing(test, bad_routing)
    bad_decision = champion_gate(base, regressed)
    print(f"[4/5] regression control candidate (held-out): {regressed['overall']:.3f}")
    print(f"      {bad_decision.render()}")

    # --- optional: train the real PolicyNetwork as a learned router ---
    learned = _learned_router_accuracy(train)
    line = learned if learned else "skipped (torch not installed)"
    print(f"[5/5] learned router accuracy: {line}")

    lo = min(base["overall"], regressed["overall"]) - 0.05
    hi = max(in_sample["overall"], champ["overall"], base["overall"]) + 0.05
    print()
    print("  reward (mean) — gate judges the held-out estimate")
    print(f"    baseline (held-out)   {_bar(base['overall'], lo, hi)}  {base['overall']:.3f}")
    print(f"    champion in-sample    {_bar(in_sample['overall'], lo, hi)}  {in_sample['overall']:.3f}  (optimistic)")
    print(f"    champion held-out     {_bar(champ['overall'], lo, hi)}  {champ['overall']:.3f}"
          f"  ({decision.overall_delta:+.3f})  {'ADOPTED' if decision.passed else 'REJECTED'}")
    print(f"    regressed (held-out)  {_bar(regressed['overall'], lo, hi)}  {regressed['overall']:.3f}"
          f"  REJECTED by gate ✓")

    if args.chart:
        _write_chart(base, in_sample, champ, regressed, decision.passed, args.chart)


def _write_chart(base, in_sample, champ, regressed, adopted, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("      (chart skipped: matplotlib not installed — pip install '.[charts]')")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "baseline\n(held-out)",
        "champion\nin-sample",
        "champion\nheld-out\n" + ("(adopted)" if adopted else "(rejected)"),
        "regressed\n(rejected)",
    ]
    vals = [base["overall"], in_sample["overall"], champ["overall"], regressed["overall"]]
    colors = ["#888", "#cdb4db", "#2a9d8f" if adopted else "#bbb", "#e76f51"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(labels, vals, color=colors)
    ax.axhline(base["overall"], color="#888", lw=1, ls="--", zorder=0)
    ax.set_ylabel("mean reward")
    ax.set_title("Champion gate: judged on held-out, not in-sample")
    ax.set_ylim(0, max(vals) * 1.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"      chart -> {path}")


if __name__ == "__main__":
    main()
