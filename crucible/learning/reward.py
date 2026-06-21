"""
Pluggable reward strategies (Q11). config.adapter.reward_strategy: two_stage | percentile | raw_linear.
v1.5.4: compute_reward_with_feedback blends offline reward with user feedback scores.
"""
import bisect
import json
from pathlib import Path
from typing import Any


def compute_reward_two_stage(record: dict) -> float:
    if record.get("success"):
        r = 1.0
        r -= min(record.get("time_sec", 0) / 600, 0.4)
        r -= min(record.get("repeated_errors", 0) * 0.05, 0.3)
        r -= min(record.get("context_trim_events", 0) * 0.02, 0.1)
        return max(r, 0.2)
    r = -0.2
    r -= min(record.get("repeated_errors", 0) * 0.05, 0.4)
    r -= min(record.get("context_trim_events", 0) * 0.02, 0.1)
    return max(r, -1.0)


def compute_reward_percentile(record: dict, buffer: Any) -> float:
    success_records = [r for r in buffer if r.get("success")]
    failure_records = [r for r in buffer if not r.get("success")]
    if record.get("success"):
        if not success_records:
            return 0.5
        times = sorted(r.get("time_sec", 0) for r in success_records)
        errors = sorted(r.get("repeated_errors", 0) for r in success_records)
        time_pct = bisect.bisect_left(times, record.get("time_sec", 0)) / len(times)
        error_pct = bisect.bisect_left(errors, record.get("repeated_errors", 0)) / len(errors)
        return 0.2 + 0.8 * (1.0 - 0.5 * time_pct - 0.5 * error_pct)
    if not failure_records:
        return -0.5
    errors = sorted(r.get("repeated_errors", 0) for r in failure_records)
    error_pct = bisect.bisect_left(errors, record.get("repeated_errors", 0)) / len(errors)
    return -0.2 - 0.8 * error_pct


def compute_reward_raw_linear(record: dict) -> float:
    r = 1.0 if record.get("success") else 0.0
    r -= record.get("time_sec", 0) * 0.1 / 60.0
    r -= record.get("repeated_errors", 0) * 0.05
    r -= record.get("context_trim_events", 0) * 0.02
    return max(r, -1.0)


REWARD_STRATEGIES = {
    "two_stage": compute_reward_two_stage,
    "percentile": compute_reward_percentile,
    "raw_linear": compute_reward_raw_linear,
}


def compute_reward(record: dict, config: dict, buffer: Any = None) -> float:
    strategy = config.get("adapter", {}).get("reward_strategy", "two_stage")
    fn = REWARD_STRATEGIES.get(strategy, compute_reward_two_stage)
    if strategy == "percentile" and buffer is not None:
        return fn(record, buffer)
    return fn(record)


def compute_reward_with_feedback(
    record: dict,
    config: dict,
    buffer: Any = None,
    feedback_path: Path | None = None,
    blend_weight: float = 0.3,
) -> float:
    """Blend offline reward with user feedback (0.7*base + 0.3*normalized_user_score).
    Falls back to base reward if no feedback found for this episode.
    """
    base = compute_reward(record, config, buffer)
    if not feedback_path or not feedback_path.exists():
        return base
    episode_id = record.get("episode_id")
    if not episode_id:
        return base
    all_fb = []
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        try:
            fb = json.loads(line)
            if fb.get("episode_id") == episode_id:
                all_fb.append(fb)
        except json.JSONDecodeError:
            pass
    if not all_fb:
        return base
    avg = sum(f["score"] / f.get("max_score", 5) for f in all_fb) / len(all_fb)
    return (1.0 - blend_weight) * base + blend_weight * avg
