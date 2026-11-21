# Pre-registered hypotheses

These were committed on 2026-11-20, before any model was run. They are copied word for word from PLAN §9. The verdicts get filled in at the end, from `results/` only.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | A tuned boosted model beats the WoE scorecard on AUC by a small margin; the scorecard wins on stability (PSI) and is close on calibration after recalibration. | **confirmed** (2026-11-21, end of R2) | `reports/validation_hc_pd.md`, `reports/credit_hc_challengers.md`, `reports/credit_hc_tuning.md`, Home Credit test set, all six models tuned. **AUC:** the champion (LightGBM, chosen on valid) beats the tuned scorecard by ΔAUC +0.023 [+0.019, +0.026] (0.781 vs 0.759, Gini 0.56 vs 0.52). The gain holds on 3 re-splits (+0.022 to +0.023) and in all 9 segments tested. It is small in absolute terms but not noise. **Stability:** the scorecard has the lowest score PSI train→Kaggle (0.0006 vs 0.0043), although both are far below any action threshold. **Calibration:** the scorecard is not just close but better. Raw ECE is 0.0020 vs 0.0041. On the grade-level Jeffreys test the scorecard has 0 grades with PD too low, the raw champion 2 (1 at the Bonferroni level) and the isotonic-recalibrated champion 3. So the hypothesis holds, with calibration even more in the scorecard's favour than predicted. Caveat: one period, stratified split, no out-of-time test (validation finding F1) |
| H2 | Filtered historical simulation with GARCH-t passes Kupiec/Christoffersen in more periods than plain historical simulation or normal VaR; the normal model fails in stress. | pending | |
| H3 | ML quantile models do not beat GARCH-based methods on VaR coverage. | pending | |
| H4 | On monthly macro series, ETS/ARIMA/seasonal naive are competitive with foundation models at a tiny fraction of the compute; the foundation models help mainly on series with few observations. | pending | |
| H5 | ECB tone adds little to volatility forecasts beyond lagged volatility. | pending | |
| H6 | LLM explanations without a verifier contain unfaithful statements at a non-trivial rate; the verifier plus a retry reduces it to near zero. | pending | |

Each verdict is one of: confirmed, refuted, inconclusive.
