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

    def render(self) -> str:
        verdict = "PASS" if self.passed else "REJECT"
        return f"[gate] {verdict}: {self.reason} (overall Δ={self.overall_delta:+.3f})"


def champion_gate(
    baseline: dict,
    candidate: dict,
    *,
    min_margin: float = 0.01,
    family_tolerance: float = 0.02,
) -> GateDecision:
    """Compare candidate metrics against the incumbent baseline.

    `baseline` / `candidate` are dicts shaped like:
        {"overall": float, "per_family": {family: float}}

    Accept iff:
      1. overall reward improves by at least `min_margin`, and
      2. no family regresses by more than `family_tolerance` below baseline.
    """
    overall_delta = candidate["overall"] - baseline["overall"]

    family_deltas: dict[str, float] = {}
    regressed: list[str] = []
    for fam, base_r in baseline.get("per_family", {}).items():
        cand_r = candidate.get("per_family", {}).get(fam, base_r)
        delta = cand_r - base_r
        family_deltas[fam] = delta
        if delta < -family_tolerance:
            regressed.append(fam)

    if regressed:
        return GateDecision(
            passed=False,
            reason=f"{len(regressed)} family regression(s): {', '.join(sorted(regressed))}",
            overall_delta=overall_delta,
            family_deltas=family_deltas,
            regressed_families=regressed,
        )
    if overall_delta < min_margin:
        return GateDecision(
            passed=False,
            reason=f"insufficient gain (need ≥{min_margin:+.3f})",
            overall_delta=overall_delta,
            family_deltas=family_deltas,
        )
    return GateDecision(
        passed=True,
        reason="improves aggregate without family regression",
        overall_delta=overall_delta,
        family_deltas=family_deltas,
    )
