"""FastAPI app served from the container (local Docker).

Read-only and stateless. It serves only what is committed to the repo: the generated reports (Markdown)
and the aggregate results (JSON). No licensed row-level data and no trained models are in the image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

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
