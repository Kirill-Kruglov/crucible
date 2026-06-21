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

import argparse

from crucible.learning.gate import champion_gate
from crucible.selfmod.loop import evaluate_policy, run, run_with_proposer
from crucible.selfmod.world import DEFAULT_POLICY


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _llm_runs(url: str, model: str, rounds: int):
    """Use a local OpenAI-compatible model as the proposer."""
    from openai import OpenAI

    from crucible.selfmod import llm_proposer

    client = OpenAI(base_url=url, api_key="not-needed")

    def propose_fn(policy, reward, reproduce, history):
        avoid = [e.delta for e, passed in history if not passed]
        return llm_proposer.propose(client, model, policy, reward, reproduce, avoid)

    print(f"proposer: LLM @ {url} ({model})\n")
    return run_with_proposer(propose_fn, rounds=rounds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-modification under the integrity gate")
    ap.add_argument("--llm", metavar="URL", nargs="?", const="http://127.0.0.1:8080/v1",
                    help="use a local OpenAI-compatible model as the proposer")
    ap.add_argument("--model", default="local")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    runs = _llm_runs(args.llm, args.model, args.rounds) if args.llm else run()

    base_r, base_i = evaluate_policy(DEFAULT_POLICY)
    print(f"start policy {DEFAULT_POLICY}")
    print(f"      reward {base_r['overall']:.3f}   reproduce {_pct(base_i)}\n")

    prev_r, prev_i = base_r, base_i
    policy = dict(DEFAULT_POLICY)
    adopted = reverted = caught = 0
    for i, (edit, decision, (cand_r, cand_i), policy) in enumerate(runs, 1):
        verb = "ADOPT" if decision.passed else "REVERT"
        print(f"[edit {i}] {edit.name}  ->  {edit.delta}")
        print(f"      reward {prev_r['overall']:.3f} -> {cand_r['overall']:.3f}"
              f"     reproduce {_pct(prev_i)} -> {_pct(cand_i)}")
        print(f"      [gate] {verb}: {decision.reason}")

        if decision.passed:
            adopted += 1
        else:
            reverted += 1
            # contrast: would a reward-only gate have accepted this very edit?
            naive = champion_gate(prev_r, cand_r)
            if naive.passed:
                caught += 1
                print(f"      (a reward-only gate would have ADOPTED this — reward Δ={naive.overall_delta:+.3f})")
        print()
        prev_r, prev_i = evaluate_policy(policy)  # what is actually in effect now

    print(f"final policy in effect: {policy}")
    print(f"      reward {prev_r['overall']:.3f}   reproduce {_pct(prev_i)}")
    print(f"\nadopted {adopted}, reverted {reverted}; "
          f"{caught} reward-positive edit(s) caught by the integrity gate.")


if __name__ == "__main__":
    main()
