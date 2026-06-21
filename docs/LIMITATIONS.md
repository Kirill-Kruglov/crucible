# Limitations & honest status

This document states what `crucible` does *not* do, on purpose. It is distilled
from the internal evaluation notes of the parent research project.

## 1. Off-policy estimation

The per-family policy proposal compares the mean reward of each strategy *as it
was historically chosen*. We never see the counterfactual: what strategy `y`
would have scored on an episode where strategy `x` was actually run. So:

- estimates are confounded by whatever drove the original strategy selection;
- cells with little support are noisy (we apply a `min_support` floor and fall
  back to the status quo when support is insufficient);
- a positive aggregate delta is **a hypothesis**, not a proven improvement.

The champion gate is the response to this: it is conservative, blocks per-family
regressions, and is meant to be run repeatedly with fresh logs.

**Partly addressed:** the loop now proposes the routing on a training split and
scores it on **held-out** episodes (`split_by_family` + `evaluate_routing`), so
in-sample optimism no longer leaks into the gate's decision. The deeper off-policy
confounding (no counterfactual rewards) remains — see below.

## 2. Not autonomous

There is no loop here that lets an agent rewrite itself unsupervised. The cycle
is: collect logs → propose → gate → a human adopts or rejects. In the parent
project, a *bounded* single-case improvement was demonstrated end-to-end
(a reproduction failure turned into a pass, reward 0.36 → 0.61), but even there
broad autonomous self-improvement was explicitly **not** claimed.

## 3. Metric fragility

In the parent project, naive prompt tweaking was repeatedly shown capable of
*regressing* multi-seed reliability while appearing to help on a single run.
This is the core reason the gate judges against a best-known baseline and across
families rather than trusting one headline number.

## 4. The learned router

The optional `PolicyNetwork` section reports train-set accuracy only. It is a
smoke check that the features carry *some* signal about strategy choice — not
evidence of generalization, and not used to make adoption decisions in this demo.

## What would move this from "interesting" to "strong"

- [x] held-out evaluation baked into the gate (done — train/test split);
- [ ] multi-seed evaluation (average the gate verdict over several splits to
  quantify variance, not just one split);
- [ ] counterfactual / IPS-style off-policy evaluation instead of naive per-cell means;
- [ ] replacing the hash-derived inference features of the parent project with real
  semantic features (length, task type, embeddings, KB-hit signals).
