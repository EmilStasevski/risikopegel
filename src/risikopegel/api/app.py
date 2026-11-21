"""FastAPI app served from the container (local Docker).

Read-only and stateless. It serves only what is committed to the repo: the generated reports (Markdown)
and the aggregate results (JSON). No licensed row-level data and no trained models are in the image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse

from risikopegel.api import rtf as rtf_calc

ROOT = Path(os.environ.get("RISIKOPEGEL_ROOT", Path(__file__).resolve().parents[3]))
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"

app = FastAPI(
    title="Risikopegel",
    description="Model-risk platform: generated reports and results. Research only, not investment advice.",
    version="0.1.0",
)


def _inside(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if not p.is_relative_to(base.resolve()) or not p.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return p


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version, "revision": os.environ.get("GIT_SHA", "unknown")}


@app.get("/")
def index() -> dict[str, Any]:
    return {
        "reports": sorted(p.name for p in REPORTS.glob("*.md")),
        "results": sorted(str(p.relative_to(RESULTS)) for p in RESULTS.rglob("*.json")),
        "note": "Research project. Not investment advice. All numbers are generated from results/.",
    }


@app.get("/reports/{name}", response_class=PlainTextResponse)
def report(name: str) -> str:
    if not name.endswith(".md"):
        raise HTTPException(status_code=404, detail="not found")
    return _inside(REPORTS, name).read_text()


@app.get("/results/{path:path}")
def result(path: str) -> Any:
    if not path.endswith(".json"):
        raise HTTPException(status_code=404, detail="not found")
    return json.loads(_inside(RESULTS, path).read_text())


@app.get("/rtf")
def rtf(  # noqa: PLR0917 (one query parameter each)
    ead_total: float | None = Query(None, gt=0, description="credit book EAD, EUR m"),
    lgd: float | None = Query(None, gt=0, le=1),
    confidence: float = Query(0.999, gt=0.5, lt=1),
    corr_scale: float = Query(1.0, gt=0, le=5, description="multiplier on the CRR retail correlation"),
    risk_cover: float | None = Query(None, gt=0),
    market_ec: float | None = Query(None, ge=0),
    irrbb_ec: float | None = Query(None, ge=0),
    op_ec: float | None = Query(None, ge=0),
) -> dict[str, Any]:
    """Recompute the economic perspective of the Module E stylised bank. Defaults come from the last run
    (results/rtf/rtf.json); every parameter can be overridden."""
    base_path = RESULTS / "rtf" / "rtf.json"
    if not base_path.is_file():
        raise HTTPException(status_code=404, detail="Module E results not built")
    base = json.loads(base_path.read_text())
    eco, cfg = base["economic"], base["config"]
    params = {
        "ead_total": ead_total if ead_total is not None else cfg["credit_book"]["ead_total"],
        "lgd": lgd if lgd is not None else cfg["credit_book"]["lgd"],
        "confidence": confidence,
        "corr_scale": corr_scale,
        "risk_cover": risk_cover if risk_cover is not None else eco["risk_cover"],
        "market_ec": market_ec if market_ec is not None else eco["ec"]["market"],
        "irrbb_ec": irrbb_ec if irrbb_ec is not None else eco["ec"]["irrbb"],
        "op_ec": op_ec if op_ec is not None else eco["ec"]["op"],
    }
    return rtf_calc.economic(params, rtf_calc.load_buckets(RESULTS / "rtf" / "pd_buckets.csv"))
