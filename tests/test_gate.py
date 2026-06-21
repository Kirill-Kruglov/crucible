from crucible.learning.gate import champion_gate


def test_accepts_clear_improvement():
    base = {"overall": 0.40, "per_family": {"a": 0.4, "b": 0.4}}
    cand = {"overall": 0.50, "per_family": {"a": 0.5, "b": 0.5}}
    d = champion_gate(base, cand)
    assert d.passed
    assert d.overall_delta > 0


def test_rejects_when_a_family_regresses_even_if_average_improves():
    # average goes up, but family "b" collapses -> must be rejected
    base = {"overall": 0.40, "per_family": {"a": 0.4, "b": 0.4}}
    cand = {"overall": 0.45, "per_family": {"a": 0.8, "b": 0.1}}
    d = champion_gate(base, cand)
    assert not d.passed
    assert "b" in d.regressed_families


def test_rejects_insufficient_gain():
    base = {"overall": 0.40, "per_family": {"a": 0.4}}
    cand = {"overall": 0.405, "per_family": {"a": 0.405}}
    d = champion_gate(base, cand, min_margin=0.01)
    assert not d.passed
    assert "insufficient" in d.reason


def test_small_family_dip_within_tolerance_is_allowed():
    base = {"overall": 0.40, "per_family": {"a": 0.4, "b": 0.4}}
    cand = {"overall": 0.45, "per_family": {"a": 0.51, "b": 0.39}}  # b dips 0.01 < tol
    d = champion_gate(base, cand, family_tolerance=0.02)
    assert d.passed
