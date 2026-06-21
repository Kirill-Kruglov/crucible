from crucible.selfmod.loop import EDITS, evaluate_policy, run, try_edit
from crucible.selfmod.world import (
    DEFAULT_POLICY,
    TASKS,
    oracle,
    repro,
    reward,
    solve,
)


def test_compute_policy_passes_oracle_and_repro():
    for t in TASKS:
        out = solve(t, {"method": "compute", "verbosity": 1})
        assert oracle(t, out)
        assert repro(t, out)


def test_lookup_policy_passes_oracle_but_fails_repro():
    # the gaming behavior: correct answer, no reproducible derivation
    for t in TASKS:
        out = solve(t, {"method": "lookup", "verbosity": 0})
        assert oracle(t, out)          # answer is right -> reward pays out
        assert not repro(t, out)       # but it doesn't reproduce


def test_repro_rejects_fabricated_inputs():
    t = {"id": "x", "kind": "sum", "input": [1, 2, 3]}
    forged = {"answer": 6, "derivation": {"kind": "sum", "input": [6]}}  # sums to 6 but not the task's inputs
    assert oracle(t, forged)
    assert not repro(t, forged)


def test_shorter_correct_output_earns_more_reward():
    t = TASKS[0]
    verbose = reward(t, solve(t, {"method": "compute", "verbosity": 3}))
    concise = reward(t, solve(t, {"method": "compute", "verbosity": 1}))
    gamed = reward(t, solve(t, {"method": "lookup", "verbosity": 0}))
    assert gamed > concise > verbose  # reward alone prefers the gaming policy


def test_gate_adopts_genuine_improvement_and_reverts_reward_hack():
    decisions = [d for _, d, _, _ in run()]
    assert decisions[0].passed          # trim explanation: adopted
    assert not decisions[1].passed      # answer-from-memory: reverted
    assert "integrity" in decisions[1].reason


def test_final_policy_keeps_reproducibility():
    *_, last = run()
    _edit, _decision, _cand, policy = last
    _r, integrity = evaluate_policy(policy)
    assert integrity == 1.0             # the surviving policy still reproduces everything


def test_try_edit_does_not_mutate_input_policy():
    start = dict(DEFAULT_POLICY)
    try_edit(start, EDITS[1])
    assert start == DEFAULT_POLICY
