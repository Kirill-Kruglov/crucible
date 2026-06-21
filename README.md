# crucible

**A measurement-first self-improvement loop for LLM agents.**

Most "self-improving agent" projects show a number going up. The hard part is
not making a number go up — it's knowing whether the improvement is *real* and
refusing the changes that aren't. `crucible` is the distilled core of that
discipline: it learns a better policy from logged agent episodes, then puts the
candidate through a **champion gate** that adopts genuine gains and **rejects
regressions** — including the subtle ones that improve the average while quietly
hurting a single task family.

The headline demo runs in seconds, on the Python standard library alone — no
LLM, no GPU, no network.

```text
$ python -m demo.self_improve
[1/5] loaded 200 episodes -> train 139 / held-out 61
[2/5] baseline mean reward (held-out): 0.390
[3/5] proposed champion:  in-sample 0.494 (optimistic)  ->  held-out 0.488
      [gate] PASS: improves aggregate without family regression (overall Δ=+0.098)
[4/5] regression control candidate (held-out): 0.310
      [gate] REJECT: 3 family regression(s): analytics-analyst, kb-analyst, web-analyst
[5/5] learned router accuracy: 0.69 over 5 strategies (off-policy — not a counterfactual guarantee)

  reward (mean) — gate judges the held-out estimate
    baseline (held-out)   ███████████·············  0.390
    champion in-sample    ████████████████████····  0.494  (optimistic)
    champion held-out     ███████████████████·····  0.488  (+0.098)  ADOPTED
    regressed (held-out)  ████····················  0.310  REJECTED by gate ✓
```

![Champion gate judged on held-out, not in-sample](docs/assets/before_after.png)

The policy is **proposed on a training split and judged on held-out episodes**.
Note the gap between the in-sample estimate (0.494, optimistic) and the held-out
estimate (0.488) — the gate trusts only the latter. A candidate that overfits the
training logs would show a large gap and be caught here.

## The loop

```
logged episodes ─► split ─► propose on train ─► estimate held-out ─► champion gate
   (replay)                  (best strategy/family)   (honest score)    (adopt / reject)
```

1. **Replay** — each episode logs the strategy used, the task family, a feature
   vector, and the reward earned (`data/replay_sample.jsonl`, 200 anonymized
   records from real agent runs).
2. **Propose on train** (`crucible/learning/routing.py`) — on a training split,
   per task family, find the strategy with the highest mean reward and propose
   routing each family to its best.
3. **Estimate held-out** — score that routing on episodes it never saw, falling
   back to the status quo where the routing isn't represented (no invented gains).
4. **Gate** (`crucible/learning/gate.py`) — adopt only if the held-out estimate
   beats the incumbent *and* regresses no family beyond tolerance. The same gate
   is shown rejecting a deliberately-worse candidate.
5. *(optional)* **Learn** — train the real `PolicyNetwork` (reward-weighted
   classification of features → strategy) as a learned router. Reported as a
   sanity signal, not a guarantee (see honesty note below).

## Quickstart

```bash
pip install -e .                  # zero runtime deps for the headline demo
python -m demo.self_improve       # the loop, in seconds

pip install -e ".[learn,charts]"  # optional: torch (learned router) + matplotlib
python -m demo.self_improve --chart
pytest                            # 9 deterministic tests, no torch needed
```

## Honesty notes (read this)

This project is deliberately modest about what it claims, because that honesty
*is* the point.

- **This is offline and off-policy.** We only observe the reward of the strategy
  that was actually chosen, never the counterfactual reward of the alternatives
  on the same episode. The per-family estimate is therefore confounded and only
  trustworthy with enough support per cell. **The gate exists precisely because
  this estimate can be wrong.**
- **This is not autonomous self-improvement.** It is a bounded, auditable loop: a
  human runs it, reads the gate's verdict, and decides. No claim is made about an
  agent improving itself unsupervised.
- **The learned router is a sanity signal.** Train-set accuracy over strategies
  is reported as a smoke check, not as evidence that routing generalizes.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the full account.

## Background

`crucible` is the distilled, publishable core of a larger private research
codebase (a self-hosted, multi-strategy LLM agent with Planner→Agent→Reviewer,
file memory, and a DuckDB knowledge base). The components here — `PolicyNetwork`,
the reward functions, the champion-gate logic — are lifted from that system and
wired into one small, fully reproducible demonstration.

## License

MIT — see [`LICENSE`](LICENSE).
