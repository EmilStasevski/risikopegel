"""R5 (Module D part 2, H5): does ECB speech tone add out-of-sample value to volatility forecasts?

Protocol (fixed before the run):
- corpus: ECB speeches dataset (ECB website, monthly snapshot; docs/data.md), English speeches dated within
  the market-data period; texts stay local, only numeric scores are stored
- tone per speech, three measures: (1) dictionary baseline (own short economic lexicon, defined below),
  (2) FinBERT (ProsusAI/finbert, CPU) averaged over 16 evenly spaced sentences, (3) LLM zero-shot (Qwen3-4B-Instruct,
  local Ollama) on the first 6,000 characters, integer -2..+2
- weekly targets (Fri-Fri): (A) log realised volatility of the Module B portfolio, (B) log of the absolute
  weekly change of the 2-year AAA yield (rate-expectation changes); prediction for week w+1
- base model: HAR-type OLS on log RV of week w, mean over weeks w-3..w, mean over weeks w-12..w
  (for B: the same lags of the target); challenger adds tone (mean of speeches in the trailing 4 weeks)
- out-of-sample: expanding window, first forecast in 2013 (>= 4 years of training), refit weekly; MSE,
  Diebold-Mariano (HAC) vs the base model; in-sample standardised tone coefficient with HAC 95% CI

Writes results/llm/text_signal/{signal.json, tone_scores.csv (date, three scores; no text)}.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from risikopegel.llm.ollama import GpuPower, chat, model_digest
from risikopegel.market.backtest import diebold_mariano
from risikopegel.market.data import load_portfolio
from risikopegel.sources import load_catalog, load_raw

RAW = Path("data/raw/ecb_speeches/all_ECB_speeches.csv")
RES = Path("results/llm/text_signal")
CACHE = Path("data/processed/text_signal")
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
FINBERT = ("ProsusAI/finbert", "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031")
FIRST_OOS = pd.Timestamp("2013-01-01")
# CPU budget: 16 evenly spaced sentences per speech, 64 tokens each (the first run with the first 64 sentences
# at 96 tokens was stopped after 9 CPU-hours without finishing)
FINBERT_SENTENCES = 16

# Own minimal lexicon (not a licensed dictionary), fixed before the run. Stems match by prefix.
NEG = [
    "crisis",
    "crises",
    "risk",
    "uncertain",
    "downturn",
    "recession",
    "declin",
    "weak",
    "stress",
    "tension",
    "deteriorat",
    "volatil",
    "turmoil",
    "loss",
    "contraction",
    "slowdown",
    "vulnerab",
    "shock",
    "fragil",
    "default",
    "concern",
    "headwind",
    "adverse",
    "pessimis",
    "unemploy",
]
POS = [
    "growth",
    "recover",
    "improv",
    "strong",
    "robust",
    "stabl",
    "stabil",
    "confiden",
    "expan",
    "resilien",
    "favourab",
    "gain",
    "optimis",
    "progress",
    "solid",
    "upturn",
    "tailwind",
    "benefit",
    "succe",
    "momentum",
]


def load_speeches(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(RAW, sep="|", dtype=str, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= start) & (df["date"] <= end) & (df["contents"].str.len() > 2000)]
    words = df["contents"].str.lower().str.findall(r"[a-z]+")
    eng = words.map(lambda w: sum(x in {"the", "and", "of", "to", "is"} for x in w) / max(len(w), 1))
    return df[eng > 0.08].reset_index(drop=True)


def dictionary_tone(text: str) -> float:
    w = re.findall(r"[a-z]+", text.lower())
    neg = sum(any(x.startswith(s) for s in NEG) for x in w)
    pos = sum(any(x.startswith(s) for s in POS) for x in w)
    return (pos - neg) / max(pos + neg, 1)


def finbert_tone(texts: list[str]) -> list[float]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = snapshot_download(
        FINBERT[0],
        revision=FINBERT[1],
        local_files_only=True,
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
    )
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()
    lab = {v.lower(): int(k) for k, v in mdl.config.id2label.items()}
    torch.set_num_threads(16)
    out = []
    for i, t in enumerate(texts):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if 40 < len(s) < 600]
        if not sents:
            out.append(float("nan"))
            continue
        pick = np.unique(np.linspace(0, len(sents) - 1, min(FINBERT_SENTENCES, len(sents))).astype(int))
        with torch.no_grad():
            enc = tok(
                [sents[j] for j in pick], padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            p = torch.softmax(mdl(**enc).logits, dim=1).numpy()
        out.append(float((p[:, lab["positive"]] - p[:, lab["negative"]]).mean()))
        if i % 250 == 0:
            print(f"finbert {i}/{len(texts)}", flush=True)
    return out


LLM_PROMPT = (
    "Rate the overall tone of the following ECB speech about the economic and financial outlook, from -2 (very "
    "negative: crisis, high risks, weak economy) to +2 (very positive: strong growth, low risks). "
    "Answer with a single integer only.\n\nSpeech excerpt:\n"
)


def llm_tone(texts: list[str]) -> tuple[list[float], dict[str, Any]]:
    out, n_bad = [], 0
    with GpuPower() as gpu:
        for i, t in enumerate(texts):
            r = chat(MODEL, [{"role": "user", "content": LLM_PROMPT + t[:6000]}], seed=i, temperature=0.0)
            m = re.search(r"[-+]?\d", r.text)
            v = float(m.group()) if m else np.nan
            if not m or abs(v) > 2:
                n_bad, v = n_bad + 1, np.nan
            out.append(v)
    return out, {
        "unparseable": n_bad,
        "seconds_per_speech": gpu.seconds / len(texts),
        "gpu_joules_per_speech": gpu.joules / len(texts),
    }


def weekly_targets() -> pd.DataFrame:
    r = load_portfolio()["portfolio"]
    rv = (r**2).resample("W-FRI").sum().pipe(np.sqrt)
    s = next(x for x in load_catalog() if x.id == "yc_aaa_2y")
    y = load_raw(s)
    y2 = pd.Series(y["value"].to_numpy(), index=pd.to_datetime(y["period"])).resample("W-FRI").last()
    out = pd.DataFrame({"rv": np.log(rv[rv > 0]), "dy2": np.log(y2.diff().abs() + 0.005)}).dropna()
    return out


def har(x: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"l1": x, "l4": x.rolling(4).mean(), "l13": x.rolling(13).mean()})


def evaluate(target: pd.Series, tone: pd.Series) -> dict[str, Any]:
    X = har(target).assign(tone=tone.reindex(target.index)).shift(0)
    df = X.assign(y=target.shift(-1)).dropna()  # features at week w, target at w+1
    base_cols, full_cols = ["l1", "l4", "l13"], ["l1", "l4", "l13", "tone"]
    z = df.copy()
    z["tone"] = (z["tone"] - z["tone"].mean()) / z["tone"].std()
    ins = sm.OLS(z["y"], sm.add_constant(z[full_cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    e_base: list[float] = []
    e_full: list[float] = []
    dates: list[pd.Timestamp] = []
    for t in df.index[df.index >= FIRST_OOS]:
        tr, te = df[df.index < t], df.loc[[t]]
        for cols, acc in ((base_cols, e_base), (full_cols, e_full)):
            m = sm.OLS(tr["y"], sm.add_constant(tr[cols])).fit()
            acc.append(
                float(te["y"].iloc[0] - m.predict(sm.add_constant(te[cols], has_constant="add")).iloc[0])
            )
        dates.append(t)
    eb, ef = np.array(e_base) ** 2, np.array(e_full) ** 2
    dm = diebold_mariano(ef, eb)
    ci = ins.conf_int().loc["tone"]
    return {
        "n_insample": int(ins.nobs),
        "tone_coef_std": float(ins.params["tone"]),
        "tone_ci95": [float(ci[0]), float(ci[1])],
        "tone_p": float(ins.pvalues["tone"]),
        "n_oos": len(dates),
        "mse_base": float(eb.mean()),
        "mse_with_tone": float(ef.mean()),
        "mse_ratio": float(ef.mean() / eb.mean()),
        "dm_stat": dm["stat"],
        "dm_p": dm["p_two_sided"],
    }


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tg = weekly_targets()
    sp = load_speeches(tg.index[0] - pd.Timedelta(days=120), tg.index[-1])
    print(f"speeches: {len(sp)}", flush=True)
    cache = CACHE / "tone.parquet"
    if cache.exists():
        scores = pd.read_parquet(cache)
        llm_meta = json.loads((CACHE / "llm_meta.json").read_text())
    else:
        scores = pd.DataFrame({"date": sp["date"], "dictionary": sp["contents"].map(dictionary_tone)})
        fb_cache = CACHE / "finbert.parquet"
        if fb_cache.exists():
            scores["finbert"] = pd.read_parquet(fb_cache)["finbert"].to_numpy()
        else:
            scores["finbert"] = finbert_tone(sp["contents"].tolist())
            scores[["finbert"]].to_parquet(fb_cache)
        print("finbert done", flush=True)
        scores["llm"], llm_meta = llm_tone(sp["contents"].tolist())
        scores.to_parquet(cache)
        (CACHE / "llm_meta.json").write_text(json.dumps(llm_meta))
    scores.to_csv(RES / "tone_scores.csv", index=False)
    out: dict[str, Any] = {
        "run": {
            "script": "scripts/r5_text_signal.py",
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "llm": MODEL,
            "llm_digest": model_digest(MODEL),
            "finbert": f"{FINBERT[0]}@{FINBERT[1]}",
        },
        "corpus": {
            "n_speeches": len(sp),
            "first": str(sp["date"].min().date()),
            "last": str(sp["date"].max().date()),
            "dataset_fetched_utc": Path(RAW.parent / "FETCHED_UTC").read_text().strip(),
        },
        "llm_scoring": llm_meta,
        "score_correlations": scores[["dictionary", "finbert", "llm"]]
        .corr(method="spearman")
        .round(3)
        .to_dict(),
        "weeks": {"first": str(tg.index[0].date()), "last": str(tg.index[-1].date()), "n": len(tg)},
        "results": {},
    }
    for m in ("dictionary", "finbert", "llm"):
        s = scores.set_index("date")[m].dropna()
        weekly = s.resample("W-FRI").mean().reindex(tg.index)
        tone = weekly.rolling(4, min_periods=1).mean().ffill()
        out["results"][m] = {t: evaluate(tg[t], tone) for t in ("rv", "dy2")}
        print(m, json.dumps(out["results"][m]), flush=True)
    (RES / "signal.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
