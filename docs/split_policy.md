# Split and leakage policy

The code is in `src/risikopegel/splits.py` and `src/risikopegel/leakage.py`. The tests are in `tests/`, are marked `leakage`, and run in CI.

## Splits

| Use | Method | Function |
|---|---|---|
| Final scores, market risk and macro | Walk-forward (rolling origin). Rolling window for VaR, expanding window for macro. Test blocks lie strictly after train, with an optional `gap` | `walk_forward` |
| Final scores, credit | Out-of-time split by application/observation date, with an optional embargo of one performance window | `out_of_time_split` |
| Hyperparameter tuning | Only inside a fold's training window. Purged and embargoed K-fold, or a nested walk-forward | `purged_kfold` |
| Any split | Must pass `assert_no_lookahead`: no shared rows, and the last training time *including its label window* comes before the first test time | `assert_no_lookahead` |

Random splits are not provided. The only exception is a dataset with no time column at all. There a stratified split is allowed, but the report has to say so and cannot make any stability or drift claim based on it. Both credit datasets fall under this exception:
- **Give Me Some Credit:** no time information at all.
- **Home Credit:** only day offsets relative to each application. The R1 analysis found no time proxy: IDs carry no signal (adversarial AUC 0.500). The Kaggle test set is a shifted population (adversarial AUC 0.934), so Module A uses train vs Kaggle test as its **drift scenario** for PSI/CSI and monitoring, not as a labelled out-of-time test.

All split sizes are counted in unique time points, so rows from the same date always end up on the same side.

## Leakage checks

1. **Causality** (`assert_causal_features`): the features are recomputed on data truncated at several cutoffs, and every value at or before a cutoff must stay the same. The tests confirm it catches centred windows, leads, full-sample scaling and backfill.
2. **Timestamps** (`assert_feature_timestamps`): a feature's availability time must be at or before the decision time (for example, macro release dates rather than reference periods).
3. **Shuffled labels** (`shuffled_label_check`): the *whole* pipeline is retrained on permuted training labels, and test performance has to fall to chance. The tests confirm it catches target encoding that was fitted on the full sample.
   - The check passes when |mean - chance| ≤ max(tolerance, 3·SE) over 20 rounds. A fixed tolerance alone doesn't work: on Give Me Some Credit, a scorecard fitted to noise had a test AUC scattered with SD ≈ 0.10–0.14 around 0.5 (single rounds ranged from 0.34 to 0.75). The reason is that it still ranks applicants by a few strong features, with a random sign. A leak instead pushes every round the same way.
   - For a scorecard, the IV filter is switched off during this check. Otherwise no variable survives on noise and the check would test nothing.

Every model pipeline in modules A–C must have a test that applies checks 1 and 3 to its real feature function.
