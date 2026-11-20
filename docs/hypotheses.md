# Pre-registered hypotheses

These were committed on 2026-11-20, before any model was run. They are copied word for word from PLAN §9. The verdicts get filled in at the end, from `results/` only.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | A tuned boosted model beats the WoE scorecard on AUC by a small margin; the scorecard wins on stability (PSI) and is close on calibration after recalibration. | pending, interim evidence (2026-11-20) | `reports/credit_hc_challengers.md`, `reports/model_card_hc_pd.md`, Home Credit test set. **AUC:** every booster beats the scorecard with a bootstrap CI excluding 0. The champion (LightGBM, chosen on valid) gains ΔAUC +0.027 [+0.024, +0.031] (0.780 vs 0.752); whether that counts as "small" is a judgement call. **Calibration:** all models are well calibrated raw (slope 0.92–1.00). The scorecard has the lowest raw ECE (0.0027 vs 0.0049). After isotonic recalibration the champion is slightly better (0.0039 vs 0.0041). **Stability:** the scorecard has the lowest score PSI train→Kaggle (0.0020 vs 0.0037 for the champion), but every model is far below 0.1, so the difference has no practical meaning in this scenario. **Explainability:** the champion's reason codes share the scorecard's top reason for 73% of jointly declined applicants. Final verdict at the end of R2, together with the validation report
| H2 | Filtered historical simulation with GARCH-t passes Kupiec/Christoffersen in more periods than plain historical simulation or normal VaR; the normal model fails in stress. | pending | |
| H3 | ML quantile models do not beat GARCH-based methods on VaR coverage. | pending | |
| H4 | On monthly macro series, ETS/ARIMA/seasonal naive are competitive with foundation models at a tiny fraction of the compute; the foundation models help mainly on series with few observations. | pending | |
| H5 | ECB tone adds little to volatility forecasts beyond lagged volatility. | pending | |
| H6 | LLM explanations without a verifier contain unfaithful statements at a non-trivial rate; the verifier plus a retry reduces it to near zero. | pending | |

Each verdict is one of: confirmed, refuted, inconclusive.
