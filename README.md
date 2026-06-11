# Card-transaction fraud detection

Fraud is about 0.5% of a card stream, so a model that always predicts "no fraud" is
99.5% accurate and completely useless. This project builds the pipeline the way a
fraud team actually builds it: a synthetic transaction stream with a realistic base
rate and four labelled fraud patterns, a strictly time-ordered train/test split,
velocity and amount-deviation features that only ever look backwards, class
weighting instead of naive oversampling, a decision threshold chosen by minimising
expected loss from a cost matrix rather than maximising F1, and an exact per-
transaction attribution that reconstructs the model's log-odds from the trees
themselves.

## What is in the box

| Module | Job |
| --- | --- |
| `src/fraud/config.py` | Typed, validated view of `config.yaml`; every tunable is parsed and range-checked once, at load time |
| `src/fraud/generator.py` | Synthetic cards and transactions: Poisson daily activity, lognormal amounts per merchant category, home countries, and four injected fraud patterns (velocity burst, extreme amount, foreign geo, night purchase) at the configured base rate |
| `src/fraud/features_velocity.py` | Per-card transaction counts and sums over 1h/24h/7d, hours since previous transaction, distinct merchants in 24h, burst flag. Windows are half-open and exclude the row being scored |
| `src/fraud/features_behaviour.py` | Amount deviation from the card's own prior history: z-score with a variance floor, ratio to prior median, percentile, ratio to prior maximum, round-amount flags, cold-start flag |
| `src/fraud/features_context.py` | Cyclic hour-of-week encodings, night/weekend flags, merchant-category risk, transaction-country risk, category frequency (fitted on train), first-seen merchant and country flags |
| `src/fraud/feature_pipeline.py` | Assembles the three blocks, fits imputation and winsorising on the training window only, builds the split masks, and exposes `assert_no_future_leakage` |
| `src/fraud/trainer.py` | `GradientBoostingClassifier` with `scale_pos_weight = negatives / positives`, a temporal holdout for tuning, deterministic training, model persistence |
| `src/fraud/cost_threshold.py` | The cost matrix, expected-loss curve over every distinct score, optimal threshold, break-even probability, threshold comparison table |
| `src/fraud/attribution.py` | Exact additive log-odds attribution: walks each tree's decision path, assigns every split delta to the feature that split on, and folds the init/root constant into the base |
| `src/fraud/evaluation.py` | Precision@k, recall at a fixed false-positive budget, ROC AUC, average precision, per-pattern recall, calibration table, one-screen summary report |
| `src/fraud/rules.py` | The rule baseline a fraud team already has, so the model's contribution is measurable against the same features |
| `src/fraud/train.py`, `src/fraud/evaluate.py` | CLI entry points: generate or load data, tune, train, persist artifacts, re-evaluate a saved bundle |

## Quickstart

```bash
pip install -r requirements.txt

# Generates data/transactions.csv (once), tunes the threshold on the tail of the
# training window, fits on the full training window, evaluates on the future,
# and writes artifacts/model.pkl, artifacts/metrics.json,
# artifacts/feature_manifest.json and artifacts/explanations.json.
python -m src.fraud.train --config config.yaml

# Re-score the saved model against a stream and rebuild the metrics/curves.
python -m src.fraud.evaluate --config config.yaml --model artifacts/model.pkl

# The suite.
make test
```

Training takes several minutes at the default size (2,500 cards over 90 days,
roughly 200k transactions). For a quick loop, shrink `data.n_cards` and
`data.n_days` in `config.yaml`; every value is read from the file, nothing is
hard-coded.

## Design notes

**The split is by time, but that alone does not make the features safe.** A random
split leaks because future transactions describe the past: a card's 24h velocity on
the last training day would be computed using test-period rows. The pipeline handles
this in three layers. First, velocity and behaviour features are functions of
strictly earlier rows only — `(t - window, t)` half-open windows, excluding the row
being scored, with duplicate timestamps resolved in favour of the earlier row, which
is what an online scorer would see. Second, everything that needs fitting (category
frequency, per-column medians for imputation, winsorising bounds) is fitted on the
training window and applied unchanged to the test window. Third,
`assert_no_future_leakage` re-computes the feature frame from truncated prefixes and
compares sampled rows; the test suite also asserts that appending the test rows
leaves every training row's features bit-identical. If either check fails, the build
fails before a model is trained.

**The threshold is a business decision, not an F1 maximum.** A false positive costs
an analyst a few minutes; a missed fraud costs the amount plus chargeback handling.
`cost_threshold.py` turns that into `expected_loss(t) = FP(t) * fp_cost +
sum_over_missed(amount_i * multiplier + loss_fee)` and minimises it over every
distinct score in the tuning window. The report includes the loss at 0.5 and the
saving versus it, the break-even probability `fp / (fp + fn)`, and the alert volume
each threshold implies. The threshold is tuned on the last 20% of the training
window (a temporal holdout), never on the test window.

**Class weighting, not oversampling.** With a 0.5% base rate, duplicating positives
makes training slower and teaches the model a prior that does not exist.
`scale_pos_weight = negatives / positives` plus per-row sample weights keeps every
row once. The resulting probabilities are shifted towards the weighted prior, which
is acceptable because the decision rule is the cost-matrix threshold, not 0.5 —
and the README states that rather than pretending the probabilities are calibrated.

**Attribution is exact, not approximated.** A gradient-boosted tree's raw margin is
`init + lr * sum(trees)`. Inside one tree, the path from root to leaf visits a chain
of splits; the change in leaf value at each split belongs to the feature that split
on. Summing those deltas over the path and then over all trees gives an additive
decomposition that matches `decision_function` to machine precision (the test
asserts < 1e-9, the implementation measures ~1e-15). No SHAP dependency, no sampling,
no background dataset. The trade-off is documented below: this is the path-dependent
variant without interaction terms, so the split of credit between correlated
features depends on tree order.

## Tests

95 tests across 12 files:

| File | Tests | What they assert |
| --- | --- | --- |
| `test_config.py` | 10 | Real config loads; overrides merge per section; missing/invalid sections and out-of-range values raise with the offending key named |
| `test_generator.py` | 11 | Same seed gives identical frames; fraud rate lands near the configured value; all four patterns appear; rows are sorted, unique, positive; CSV round-trip; chronological split respects the gap |
| `test_features_velocity.py` | 13 | Hand-counted window counts/sums on a five-row frame, duplicate-timestamp ordering, distinct-merchant scans, gap hours, burst flag, index alignment under shuffled input, missing-column errors |
| `test_features_behaviour.py` | 7 | Prior-only history statistics on `[100, 100, 300]`: mean/std/median/max/percentile exact, z-score with variance floor, cold-start flag clears at `min_history` |
| `test_features_context.py` | 8 | Cyclic encodings equal trig values at 0/6/12/18h, bucket boundaries, weekend/night flags, risk-table overrides, category frequency fitted and applied with unseen fallback, causal first-sighting flags |
| `test_pipeline.py` | 8 | Appending test rows does not change training features; leakage assertion passes; masks are disjoint and respect the gap; imputer/clipper behave; manifest round-trip |
| `test_trainer.py` | 10 | `scale_pos_weight` is exactly negatives/positives; single-class training raises; temporal holdout is ordered; confusion metrics exact; importances sum to one; save/load round-trip |
| `test_cost_threshold.py` | 8 | Hand-computed expected losses (0/10/150), loss curve thresholds and optimum, break-even probability, tuning report fields, cost matrix from config |
| `test_attribution.py` | 6 | Additivity < 1e-9 against `decision_function`; contributions match per-row; sigmoid matches `predict_proba`; grouped contributions sum to total; explanation JSON is valid |
| `test_evaluation.py` | 8 | Precision@k and recall@k on fixed scores, FP-budget threshold, ROC AUC on a deliberately inverted ranking, confusion counts, per-pattern recall, lift, calibration bins |
| `test_rules.py` | 5 | Rule flags fire only on the intended rows, score equals the weighted sum, disabled-rule paths, threshold tuning improves over the default |
| `test_cli.py` | 1 | Full pipeline on a temp config: artifacts exist, metrics are consistent, `evaluate` re-reads the bundle and writes a report |

Run them with `python -m pytest -q`; nothing needs a network connection, a GPU or a
model download.

## Limitations

- The data is synthetic and the four fraud patterns are hand-designed. Features are
  aligned with the generator's mechanisms, so measured precision/recall are
  optimistic relative to real card data; the pipeline and the failure modes it
  guards against are the reusable part, not the numbers.
- The cost matrix is a constant. It assumes stationary review cost and a fixed
  fraud-loss fee, and it does not model per-transaction chargeback probability.
- Cards are treated as stationary: activity rate, spend scale and preferred
  categories never drift, so concept drift and seasonal fraud migration are absent.
- A single model. There is no ensemble, no per-segment model (B2B vs consumer, new
  card vs established card) and no rule/model hybrid.
- Attribution has no interaction terms. It is exact for the model margin but splits
  credit between correlated features in tree order; it should not be read as a
  causal statement.
- Weighted training shifts predicted probabilities toward the weighted prior. The
  cost threshold absorbs that; the raw probabilities are not calibrated to the true
  base rate.
- Velocity features hold each card's history in memory; there is no incremental
  online state, no drift monitoring, no retraining schedule and no serving layer.
  Input is CSV only.
