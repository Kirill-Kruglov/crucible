"""Champion gate: accept a candidate policy only if it does not regress.

This is the honest core of the loop. Offline metrics are *estimates*, so the
gate is deliberately conservative: a candidate must beat the incumbent on the
aggregate signal AND must not regress any individual family below tolerance.
A candidate that merely improves the average while quietly hurting one family
is rejected. This is what keeps an offline self-improvement loop from fooling
itself (Goodhart) on a single headline number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateDecision:
    passed: bool
    reason: str
    overall_delta: float
    family_deltas: dict[str, float] = field(default_factory=dict)
    regressed_families: list[str] = field(default_factory=list)
    integrity_delta: float | None = None

    def render(self) -> str:
        verdict = "PASS" if self.passed else "REJECT"
        extra = "" if self.integrity_delta is None else f", integrity Δ={self.integrity_delta:+.3f}"
        return f"[gate] {verdict}: {self.reason} (reward Δ={self.overall_delta:+.3f}{extra})"


def champion_gate(
    baseline: dict,
    candidate: dict,
    *,
    min_margin: float = 0.01,
    family_tolerance: float = 0.02,
    baseline_integrity: float | None = None,
    candidate_integrity: float | None = None,
    integrity_tolerance: float = 0.03,
) -> GateDecision:
    """Compare candidate metrics against the incumbent baseline.

    `baseline` / `candidate` are dicts shaped like:
        {"overall": float, "per_family": {family: float}}

    Accept iff ALL hold:
      1. overall reward improves by at least `min_margin`;
      2. no family regresses by more than `family_tolerance`;
      3. (when integrity is supplied) the reproduce rate does not drop by more
         than `integrity_tolerance` — reward gains bought by eroding
         reproducibility are rejected, even if every reward check passes.
    """
    overall_delta = candidate["overall"] - baseline["overall"]
    integrity_delta = (
        None
        if baseline_integrity is None or candidate_integrity is None
        else candidate_integrity - baseline_integrity
    )

    family_deltas: dict[str, float] = {}
    regressed: list[str] = []
    for fam, base_r in baseline.get("per_family", {}).items():
        cand_r = candidate.get("per_family", {}).get(fam, base_r)
        delta = cand_r - base_r
        family_deltas[fam] = delta
        if delta < -family_tolerance:
            regressed.append(fam)

    def decide(passed: bool, reason: str) -> GateDecision:
        return GateDecision(
            passed=passed,
            reason=reason,
            overall_delta=overall_delta,
            family_deltas=family_deltas,
            regressed_families=regressed,
            integrity_delta=integrity_delta,
        )

    if integrity_delta is not None and integrity_delta < -integrity_tolerance:
        return decide(False, f"integrity regression (reproduce rate {integrity_delta:+.3f})")
    if regressed:
        return decide(False, f"{len(regressed)} family regression(s): {', '.join(sorted(regressed))}")
    if overall_delta < min_margin:
        return decide(False, f"insufficient reward gain (need ≥{min_margin:+.3f})")
    return decide(True, "improves reward without family or integrity regression")
