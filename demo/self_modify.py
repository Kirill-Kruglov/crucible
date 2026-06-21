"""Self-modification under an integrity gate — bounded, honest, zero-dependency.

An agent proposes edits to its own policy. Each edit is judged by the same gate
used across `crucible`: it is kept only if it raises reward *without* eroding the
reproduce rate. The demo shows a genuine improvement being adopted, and a
reward-hacking edit (right answers, no reproducible work) being reverted — even
though a reward-only gate would have kept it.

This is the bounded version of "an agent rewrites itself to get better": it can
only keep changes that survive an integrity check it cannot game.

    python -m demo.self_modify
"""

from __future__ import annotations

from crucible.learning.gate import champion_gate
from crucible.selfmod.loop import EDITS, evaluate_policy, run
from crucible.selfmod.world import DEFAULT_POLICY


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def main() -> None:
    base_r, base_i = evaluate_policy(DEFAULT_POLICY)
    print(f"start policy {DEFAULT_POLICY}")
    print(f"      reward {base_r['overall']:.3f}   reproduce {_pct(base_i)}\n")

    prev_r, prev_i = base_r, base_i
    for i, (edit, decision, (cand_r, cand_i), policy) in enumerate(run(), 1):
        verb = "ADOPT" if decision.passed else "REVERT"
        print(f"[edit {i}] {edit.name}  ->  {edit.delta}")
        print(f"      reward {prev_r['overall']:.3f} -> {cand_r['overall']:.3f}"
              f"     reproduce {_pct(prev_i)} -> {_pct(cand_i)}")
        print(f"      [gate] {verb}: {decision.reason}")

        if not decision.passed:
            # contrast: a reward-only gate would have accepted this very edit
            naive = champion_gate(prev_r, cand_r)
            if naive.passed:
                print(f"      (a reward-only gate would have ADOPTED this — reward Δ={naive.overall_delta:+.3f})")
        print()
        # what is actually in effect after the gate's decision:
        prev_r, prev_i = evaluate_policy(policy)

    print(f"final policy in effect: {policy}")
    print(f"      reward {prev_r['overall']:.3f}   reproduce {_pct(prev_i)}")
    print("\nThe reward-hacking edit raised reward but was reverted: integrity held.")


if __name__ == "__main__":
    main()
