"""Tests for the proposer's pure logic — no server needed.

The LLM call itself is exercised live via `python -m demo.self_modify --llm`;
here we lock down the safety layer (parsing + allowlist validation) that stands
between any model output and the agent's policy.
"""

from crucible.selfmod.llm_proposer import _extract_json, validate_edit


def test_extract_json_from_noisy_text():
    assert _extract_json('here you go: {"verbosity": 1} ok?') == {"verbosity": 1}
    assert _extract_json("no json here") == {}
    assert _extract_json("{bad json}") == {}


def test_validator_strips_unknown_fields():
    # the model hallucinated a field that isn't in the policy
    assert validate_edit({"verbosity": 1, "reproduce_rate": 0.8}) == {"verbosity": 1}


def test_validator_rejects_out_of_range_and_bad_values():
    assert validate_edit({"verbosity": 9}) is None        # out of 0..3
    assert validate_edit({"method": "teleport"}) is None   # not an allowed method
    assert validate_edit({}) is None


def test_validator_accepts_legal_edits():
    assert validate_edit({"method": "lookup"}) == {"method": "lookup"}
    assert validate_edit({"method": "compute", "verbosity": 0}) == {
        "method": "compute",
        "verbosity": 0,
    }
