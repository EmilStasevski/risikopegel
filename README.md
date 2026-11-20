# Risikopegel

A model-risk platform written in the style of bank model documentation. It has four modules on one codebase:

- **A, credit risk:** a WoE/logistic PD scorecard compared with gradient-boosting challengers, plus calibration, stability (PSI) monitoring, SHAP reason codes and a simplified IFRS 9-style ECL.
- **B, market risk:** VaR/ES models with a full backtest suite (Kupiec, Christoffersen, Basel traffic light, ES backtests).
- **C, macro forecasting:** classical baselines compared with foundation models, including the compute cost of each.
- **D, LLM:** German credit-decision explanations that a deterministic verifier checks against the model's SHAP values.

> **Status:** setup (milestone R1). No results yet. Every number in this README will be generated from `results/` by `task report`.

> **Not investment advice.** This is risk measurement, not a trading system. It makes no claims about returns or regulatory compliance.

## Quick start

```bash
uv sync
task test     # unit + leakage tests
task lint
```

See `Risikopegel_PLAN.md` for the plan, `docs/` for data, versions, hypotheses and the split policy.
