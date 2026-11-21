# Versions

The exact pins are in `uv.lock`. This table records what was installed and checked at each milestone. Rule 2 of the plan says anything written from memory has to be verified, and this is where the verification is recorded.

## Environment (2026-11-20, R1 start)

| Component | Version | Source / note |
|---|---|---|
| Python | 3.12.14 (uv-managed) | `requires-python >=3.12,<3.13` |
| numpy | 2.5.5 | uv.lock |
| pandas | 3.1.1 | uv.lock (pandas 3: copy-on-write by default) |
| polars | 2.1.0 | uv.lock |
| scipy | 1.18.1 | uv.lock |
| scikit-learn | 1.9.1 | uv.lock |
| ruff / mypy / pytest | 0.17.6 / 2.4.1 / 9.2.0 | dev group |
| lightgbm / catboost / xgboost / interpret | 4.7.0 / 1.2.10 / 3.4.1 / 0.7.8 | uv.lock (R2). SHAP: the libraries' built-in exact TreeSHAP, not the `shap` package. **XGBoost pitfall:** `Booster.predict(pred_contribs=True)` uses all trees unless `iteration_range` is passed, while the sklearn wrapper predicts with the early-stopped prefix. Fixed in `credit/explain.py` and covered by a test |
| optuna | 5.0.0 | uv.lock (R2 tuning). Pinned `>=5.0,<5.1`: 5.1.0 came out 2026-11-16, under two weeks before this choice, so the previous release (2026-09-07) is used. TPE sampler, seeded, SQLite storage so the search can resume |
| fastapi / uvicorn / httpx | 0.162.0 / 0.54.0 / 0.28.1 | `api` dependency group (`httpx` for the test client). FastAPI 0.162.0 is the newest release at least two weeks old (0.163 to 0.165 came out on 2026-11-15 to 18) |
| Docker base images | python:3.12.14-slim, ghcr.io/astral-sh/uv:0.12.21 | match the local toolchain |
| arch / statsmodels | 8.0.0 / 0.15.0 | R3 GARCH/GJR fits and MCS (`arch.bootstrap.MCS`). arch 9.0.0 (2026-11-11) is under two weeks old, so `>=8,<9` |
| statsforecast | **not used** | 2.1.1 requires `pandas<3.0`, and this project runs pandas 3.1. Module C uses statsmodels ETS/ARIMA instead |
| torch (CPU) / chronos-forecasting / timesfm / transformers | 2.14.1+cpu / 2.3.2 / 3.0.2 / 5.22.2 | `fm` group, R4. CPU-only wheel from the PyTorch index (~0.2 GB instead of the CUDA build). Models are pinned by Hugging Face revision; see `docs/data.md` |
| matplotlib / pyarrow | 3.11.2 / see uv.lock | figures / parquet artifacts |
| optbinning | 1.0.0 | uv.lock. Raises pandas-3 deprecation warnings internally (filtered in pytest config). `Scorecard(scaling_method="pdo_odds")` API checked by inspecting the installed version |
| Kaggle CLI | 2.2.5 | `uvx --from kaggle --with six` (2.2.5 is missing its `six` dependency) |
| go-task | 3.54.0 | `~/go/bin/task` |
| DuckDB CLI | 2.0.0 | `~/.local/bin/duckdb` |
| rustc | 1.99.1 | for the optional PyO3 kernel (R3) |
| GPU | RTX 5060 Ti 16 GB, driver 580.178.04 | for Module C/D models |

Libraries for the later milestones (optbinning, lightgbm, catboost, xgboost, shap, interpret, mapie, arch, statsforecast, neuralforecast/darts, evidently/nannyML, dbt-duckdb, mlflow) get added when their milestone starts. Their current versions and APIs will be checked at that point, not taken from the plan.

## Scope decisions

- **Module E added: a simplified Risikotragfähigkeit (RTF) model** (user decision, 2026-11-21). It has an economic perspective: credit, market, interest-rate (IRRBB) and operational risk, aggregated against the risk cover. It has a normative perspective: 3-year capital planning under scenarios, plus a reverse stress test. It ties Modules A to C into one bank-wide view. The foundation-model benchmark in Module C was first dropped to make room; with Azure gone it is back in scope (user approved the downloads, 2026-11-21). Design: `docs/module_e_rtf.md`.
- **Azure dropped** (user decision, 2026-11-21): no Azure resources were ever created. The Terraform scaffold (Container Apps, budget alert) and its CI job were removed again; they remain in git history (commit `98c9440`). The Docker image and read-only API stay, run locally and built in CI.

- **No dependency on ZeilenJoule** (user decision, 2026-11-20). The plan says to reuse the `zj` harness and the ZeilenJoule 9B LLM, but this project does neither. Timing and energy measurement (NVML power sampling and RAPL) will be implemented inside this repo when Module C needs it. Any LLM or foundation model will be downloaded fresh, and only after asking if it is larger than ~1 GB.

## Facts written from memory, still to verify

| Claim (plan section) | Status | Source |
|---|---|---|
| Basel traffic light: 250 days at 99%, green ≤4, yellow 5–9, red ≥10 (§4) | **verified 2026-11-21:** BCBS MAR32 table: green 0–4, amber 5–9, red 10+ for 250 observations. Boundaries are where the binomial CDF reaches 95 % and 99.99 % (MAR99 / SAMA rulebook). The code derives the zones from the binomial CDF, and a test checks them against the table | [BCBS MAR32](https://www.bis.org/committees/bcbs/basel-framework/81326/chapter.pdf), [SAMA rulebook](https://www.rulebook.sama.gov.sa/en/node/4000) |
| 97.5% ES is the FRTB basis (§4) | **verified 2026-11-21:** MAR33.3, ES at a 97.5th percentile, one-tailed confidence level | [BCBS MAR33](https://www.bis.org/committees/bcbs/basel-framework/81327/chapter.pdf) |
| Acerbi–Szekely Z2 zones −0.70 / −1.8 | **partly verified 2026-11-21:** the 0.7 / 1.8 thresholds for Test 2 are cited in arXiv 1709.01337. The original paper's table was not read. They are used as indicative zones only, next to an exceedance-residual bootstrap test | [Acerbi & Szekely 2014 (MSCI)](https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b), [arXiv 1709.01337](https://arxiv.org/pdf/1709.01337) |
| EU AI Act: credit scoring listed as high-risk (§3) | **verified 2026-11-20:** Annex III point 5(b) covers evaluating the creditworthiness of natural persons or establishing their credit score, except systems used to detect financial fraud. Commission guidelines on high-risk classification were still a draft in the sources found | [AI Act Service Desk, Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3); law-firm commentary on the draft guidelines |
| SR 11-7 and ECB/BaFin model-risk guidance names (§7) | **verified 2026-11-21:** SR 11-7 ("Supervisory Guidance on Model Risk Management", Fed/OCC, 4 April 2011) is **superseded** by SR 26-2 (Fed/FDIC/OCC, 17 April 2026; OCC Bulletin 2026-13), which also replaces SR 21-8. Cite SR 26-2 as current and SR 11-7 only as the historical reference. The content of SR 26-2 has not been read here | [SR 11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm), [OCC Bulletin 2026-13](https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html) |
| MaRisk AT 4.3.5 (use of models) | **verified 2026-11-21:** added by the 7th MaRisk amendment (BaFin circular 05/2023, 29 June 2023). It covers model assumptions, suitability before use with regular review, data quality, rules for using model results, and explainability, and explicitly applies to AI/ML models. Models within the CRR scope are excluded. A newer MaRisk version dated 30 June 2026 exists (Bundesbank); whether AT 4.3.5 is unchanged there has not been checked | [BaFin RS 05/2023](https://www.bafin.de/SharedDocs/Veroeffentlichungen/DE/Rundschreiben/2023/rs_05_2023_MaRisk_BA.pdf), [Noerr summary](https://www.noerr.com/de/insights/bafin-veroeffentlicht-siebte-novelle-ihres-rundschreibens-zu-den-mindestanforderungen-an-das-risikomanagement), [Bundesbank 2026-06-30](https://www.bundesbank.de/resource/blob/825336/866a68577eb35afee82fbe60c39a4f87/472B63F073F071307366337C94F8C870/2026-06-30-rundschreiben-data.pdf) |
| Jeffreys test for PD back-testing | **verified 2026-11-21:** used in the ECB "Instructions for reporting the validation results of internal models: IRB Pillar I models for credit risk" (February 2019). One-sided test; H0 is that the grade PD is at least the true default rate; p-value = Beta(D+½, N−D+½) CDF at the PD. The ECB guide to internal models was updated in June 2026, so whether the 2019 instructions still apply has not been checked | [ECB instructions (2019)](https://www.bankingsupervision.europa.eu/activities/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.cs.pdf), [arXiv 2005.03698](https://pdf.arxiv.org/pdf/2005.03698) |
| Risikotragfähigkeit (ICAAP) | **verified 2026-11-21:** BaFin/Bundesbank RTF guideline of 24 May 2018, based on MaRisk AT 4.1 Tz. 2. It has a normative perspective (capital planning over at least 3 years) and an economic perspective (1-year horizon, internal methods, expected and unexpected losses). Whether the 2018 guideline is unchanged has not been checked | [BaFin RTF-Leitfaden 2018](https://www.bafin.de/SharedDocs/Downloads/DE/Anlage/dl_180524_rtf-leitfaden_veroeffentlichung.pdf%3F__blob%3DpublicationFile%26v%3D1) |
| Module E regulatory facts (CRR Art. 154 retail formula without the 1.06 factor, PD/LGD floors, Art. 92 ratios, op-risk BIC; IRRBB EUR shocks 200/250/100 bp, floor, outlier thresholds) | **verified 2026-11-21** against the EUR-Lex PDFs; see the table in `docs/module_e_rtf.md` | EUR-Lex |
| Foundation models current and licensed (Chronos, TimesFM, Moirai) (§5) | open (R4) | |
