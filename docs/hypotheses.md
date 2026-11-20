# Pre-registered hypotheses

These were committed on 2026-11-20, before any model was run. They are copied word for word from PLAN §9. The verdicts get filled in at the end, from `results/` only.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | A tuned boosted model beats the WoE scorecard on AUC by a small margin; the scorecard wins on stability (PSI) and is close on calibration after recalibration. | pending | |
| H2 | Filtered historical simulation with GARCH-t passes Kupiec/Christoffersen in more periods than plain historical simulation or normal VaR; the normal model fails in stress. | pending | |
| H3 | ML quantile models do not beat GARCH-based methods on VaR coverage. | pending | |
| H4 | On monthly macro series, ETS/ARIMA/seasonal naive are competitive with foundation models at a tiny fraction of the compute; the foundation models help mainly on series with few observations. | pending | |
| H5 | ECB tone adds little to volatility forecasts beyond lagged volatility. | pending | |
| H6 | LLM explanations without a verifier contain unfaithful statements at a non-trivial rate; the verifier plus a retry reduces it to near zero. | pending | |

Each verdict is one of: confirmed, refuted, inconclusive.
