"""Log every module's results into a local MLflow store (mlflow.db + mlruns/, git-ignored).

One run per module, tagged with the git revision: parameters from the run configs, metrics from results/,
the generated report as an artifact. It reads results/ only, so it never changes a number; it makes runs
comparable across code versions in the MLflow UI.

    uv run --group mlops python scripts/mlflow_log.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

R = Path("results")


def _j(rel: str) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((R / rel).read_text())
    return out


def credit() -> tuple[dict[str, Any], dict[str, float], str]:
    r = _j("credit/hc_challengers/results.json")
    champ = r["champion"]["name"]
    m: dict[str, float] = {}
    for k, v in r["models"].items():
        for met in ("auc", "gini", "brier", "ece", "calibration_slope"):
            m[f"{k}.test_{met}"] = float(v["test"][met])
        m[f"{k}.psi_train_to_kaggle"] = float(v["psi_train_to_kaggle"])
    params = {
        "champion": champ,
        **{f"{champ}.{k}": v for k, v in r["models"][champ]["selected_params"].items()},
    }
    return params, m, "reports/credit_hc_challengers.md"


def market() -> tuple[dict[str, Any], dict[str, float], str]:
    b = _j("market/backtest/backtest.json")
    s = pd.read_csv(R / "market/backtest/summary.csv")
    s = s[(s["portfolio"] == "main") & (s["window"] == b["config"]["backtest"]["window"])]
    m = {f"{r.model}.{r.level}.exception_rate": float(r.rate) for r in s.itertuples()}
    m |= {f"{r.model}.{r.level}.p_cc": float(r.p_cc) for r in s.itertuples()}
    return (
        {"window": b["config"]["backtest"]["window"], "first_forecast": b["data"]["first_forecast"]},
        m,
        ("reports/market_var_backtest.md"),
    )


def macro() -> tuple[dict[str, Any], dict[str, float], str]:
    f = _j("macro/forecast/forecast.json")
    b = pd.read_csv(R / "macro/forecast/by_h.csv")
    m = {f"{mm}.h{h}.mase": float(v) for mm, h, v in zip(b["model"], b["h"], b["mase"], strict=True)}
    m |= {
        f"{k}.joules_above_idle_per_forecast": float(v["joules_above_idle_per_forecast"])
        for k, v in f["compute"].items()
    }
    return f["protocol"], m, "reports/macro_forecast.md"


def rtf() -> tuple[dict[str, Any], dict[str, float], str]:
    r = _j("rtf/rtf.json")
    m = {f"ec.{k}": float(v) for k, v in r["economic"]["ec"].items()}
    m |= {
        "utilisation_sum": r["economic"]["utilisation_sum"],
        "min_cet1_adverse": r["normative"]["min_ratio_adverse"],
        "reverse_pd_multiplier": r["reverse_stress"]["pd_multiplier_breaching_cet1_requirement"],
    }
    return (
        {"cet1": r["config"]["bank"]["cet1"], "lgd": r["config"]["credit_book"]["lgd"]},
        m,
        "reports/rtf.md",
    )


def llm() -> tuple[dict[str, Any], dict[str, float], str]:
    s = _j("llm/explanations/summary.json")
    m = {
        f"{k}.{met}": float(v[met])
        for k, v in s["variants"].items()
        for met in ("pass_rate", "hallucination_rate", "gpu_joules_per_explanation")
    }
    return (
        {"model": s["run"]["model"], "temperature": s["run"]["temperature"]},
        m,
        "reports/llm_explanations.md",
    )


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # file store is deprecated in MLflow 3.18
    mlflow.set_experiment("risikopegel")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()
    for name, fn in (
        ("credit_pd", credit),
        ("market_var", market),
        ("macro_forecast", macro),
        ("rtf", rtf),
        ("llm_explanations", llm),
    ):
        try:
            params, metrics, report = fn()
        except FileNotFoundError:
            continue
        with mlflow.start_run(run_name=f"{name}@{sha}"):
            mlflow.set_tags({"module": name, "git_sha": sha})
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k.replace("%", "pct"): v for k, v in metrics.items()})
            if Path(report).exists():
                mlflow.log_artifact(report)
        print(f"logged {name}: {len(metrics)} metrics")


if __name__ == "__main__":
    main()
