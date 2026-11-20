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
| go-task | 3.54.0 | `~/go/bin/task` |
| DuckDB CLI | 2.0.0 | `~/.local/bin/duckdb` |
| rustc | 1.99.1 | for the optional PyO3 kernel (R3) |
| GPU | RTX 5060 Ti 16 GB, driver 580.178.04 | for Module C/D models |

Libraries for the later milestones (optbinning, lightgbm, catboost, xgboost, shap, interpret, mapie, arch, statsforecast, neuralforecast/darts, evidently/nannyML, dbt-duckdb, mlflow) get added when their milestone starts. Their current versions and APIs will be checked at that point, not taken from the plan.

## Scope decisions

- **No dependency on ZeilenJoule** (user decision, 2026-11-20). The plan says to reuse the `zj` harness and the ZeilenJoule 9B LLM, but this project does neither. Timing and energy measurement (NVML power sampling and RAPL) will be implemented inside this repo when Module C needs it. Any LLM or foundation model will be downloaded fresh, and only after asking if it is larger than ~1 GB.

## Facts written from memory, still to verify

| Claim (plan section) | Status | Source |
|---|---|---|
| Basel traffic light: 250 days at 99%, green ≤4, yellow 5–9, red ≥10 (§4) | open (R3) | Basel Committee, Supervisory framework for the use of backtesting (1996) / MAR99 |
| 97.5% ES is the FRTB basis (§4) | open (R3) | BCBS MAR33 |
| EU AI Act: credit scoring listed as high-risk (§3) | open (R2) | Regulation (EU) 2024/1689, Annex III |
| SR 11-7 and ECB/BaFin model-risk guidance names (§7) | open (R5) | |
| Foundation models current and licensed (Chronos, TimesFM, Moirai) (§5) | open (R4) | |
