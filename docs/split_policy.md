# Split and leakage policy

The code is in `src/risikopegel/splits.py` and `src/risikopegel/leakage.py`. The tests are in `tests/`, are marked `leakage`, and run in CI.

## Splits

| Use | Method | Function |
|---|---|---|
| Final scores, market risk and macro | Walk-forward (rolling origin). Rolling window for VaR, expanding window for macro. Test blocks lie strictly after train, with an optional `gap` | `walk_forward` |
| Final scores, credit | Out-of-time split by application/observation date, with an optional embargo of one performance window | `out_of_time_split` |
| Hyperparameter tuning | Only inside a fold's training window. Purged and embargoed K-fold, or a nested walk-forward | `purged_kfold` |
| Any split | Must pass `assert_no_lookahead`: no shared rows, and the last training time *including its label window* comes before the first test time | `assert_no_lookahead` |

Random splits are not provided. The only exception is a dataset with no time column at all (e.g. Give Me Some Credit). There a stratified split is allowed, but the report has to say so and cannot make any stability or drift claim based on it.

All split sizes are counted in unique time points, so rows from the same date always end up on the same side.

## Leakage checks

1. **Causality** (`assert_causal_features`): the features are recomputed on data truncated at several cutoffs, and every value at or before a cutoff must stay the same. The tests confirm it catches centred windows, leads, full-sample scaling and backfill.
2. **Timestamps** (`assert_feature_timestamps`): a feature's availability time must be at or before the decision time (for example, macro release dates rather than reference periods).
3. **Shuffled labels** (`shuffled_label_check`): the *whole* pipeline is retrained on permuted training labels, and test performance has to fall to chance. The tests confirm it catches target encoding that was fitted on the full sample.

Every model pipeline in modules A–C must have a test that applies checks 1 and 3 to its real feature function.
