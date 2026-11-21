# Risikopegel

[Deutsche Fassung](README.de.md)

A model-risk platform written in the style of bank model documentation: credit risk, market risk, macro forecasting, a bank-wide risk-bearing-capacity (RTF/ICAAP) view and an LLM module whose output is checked deterministically. Everything runs locally. Every number in this README and in `reports/` is generated from `results/` by `task report`.

> **Not investment advice.** This is risk measurement, not a trading system. It makes no return forecasts and no claim of regulatory compliance (Basel, CRR/CRD, IFRS 9, EU AI Act, GDPR). Where it follows a regulatory formula, the formula was checked against the primary text (`docs/versions.md`, `docs/module_e_rtf.md`).

## Headline results

<!-- BEGIN GENERATED: scripts/report_readme.py -->
<!-- END GENERATED -->

Pre-registered hypotheses and their verdicts: [`docs/hypotheses.md`](docs/hypotheses.md).

## Modules and reports

| Module | Question | Report |
|---|---|---|
| A, credit risk | Does gradient boosting beat a WoE scorecard, and what does it cost in calibration, stability and explainability? | [challengers](reports/credit_hc_challengers.md), [tuning](reports/credit_hc_tuning.md), [model card](reports/model_card_hc_pd.md), [independent validation](reports/validation_hc_pd.md), [smoke test](reports/credit_gmsc_smoke.md) |
| A+C, ECL | How do macro scenarios move PDs and expected credit loss? | [macro-conditional PD and scenario ECL](reports/credit_ecl.md) |
| B, market risk | Which VaR/ES method passes backtests in calm and stressed periods, and does ML add anything? | [VaR/ES backtesting](reports/market_var_backtest.md) |
| C, macro forecasting | Do foundation models beat classical baselines, and at what compute cost? | [forecasting benchmark](reports/macro_forecast.md) |
| D, LLM | Can an LLM-written German credit explanation be proven faithful to the model? Does ECB tone help volatility forecasts? | [verified explanations](reports/llm_explanations.md), [tone signal](reports/llm_text_signal.md) |
| E, Risikotragfähigkeit | Does a stylised bank built on modules A–C have enough risk cover (economic view) and capital over 3 years (normative view)? | [RTF](reports/rtf.md) |
| Data | Sources, licenses, periods | [series](reports/data_series.md), [`docs/data.md`](docs/data.md) |

## How it is built

- **Data:** Kaggle Home Credit and Give Me Some Credit (credit), ECB/Bundesbank/Eurostat APIs (FX, yield curves, macro), ECB speeches dataset. Licenses and approvals: [`docs/data.md`](docs/data.md). Raw data is never committed.
- **Discipline:** time-based splits only (`risikopegel.splits`); every model pipeline has a causality test and a shuffled-label test (`pytest -m leakage`); regulatory facts are verified, not recalled ([`docs/versions.md`](docs/versions.md)).
- **Stack:** Python 3.12 with uv; ruff and mypy `--strict`; pytest; dbt + DuckDB feature marts; LightGBM/XGBoost/CatBoost/EBM/optbinning; arch/statsmodels; Chronos-2, Chronos-Bolt and TimesFM 2.5 on CPU; FinBERT; Qwen3 4B via local Ollama; FastAPI + Docker; Streamlit; GitHub Actions.
- **Energy:** CPU energy via Intel RAPL, GPU via `nvidia-smi` sampling, measured in this repo.

## Quick start

```bash
uv sync                 # locked dependencies (CPU-only torch)
task test               # unit + leakage tests (offline)
task lint               # ruff + mypy --strict
task report             # rebuild every report and the README tables from results/
task dashboard          # Streamlit view on results/
docker build -t risikopegel-api . && docker run -p 8000:8000 risikopegel-api   # read-only API incl. GET /rtf
```

Re-running the models needs the data (`task data:homecredit`, `task data:series`) and, for Module D, a local Ollama with `qwen3:4b-instruct-2507-q4_K_M`. The pipeline per milestone is `task r2:tuned`, `task r3`, `task r4`, `task r5:rtf`, `task r5:llm`.

See [`Risikopegel_PLAN.md`](Risikopegel_PLAN.md) for the plan and `docs/` for data, versions, hypotheses and the split policy.
