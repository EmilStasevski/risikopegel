from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from risikopegel.api.app import app
from risikopegel.api.rtf import credit_ul
from risikopegel.rtf.credit import irb_retail_k

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_lists_committed_reports() -> None:
    body = client.get("/").json()
    assert "validation_hc_pd.md" in body["reports"]
    assert "credit/hc_challengers/results.json" in body["results"]


def test_report_and_result() -> None:
    assert client.get("/reports/validation_hc_pd.md").text.startswith("# Independent validation report")
    assert "champion" in client.get("/results/credit/hc_challengers/results.json").json()


def test_no_path_traversal_or_other_files() -> None:
    for url in ("/reports/..%2Fpyproject.toml", "/results/../../pyproject.toml", "/reports/x.txt",
                "/results/credit/hc_challengers/csi_train_to_kaggle.csv"):  # fmt: skip
        assert client.get(url).status_code == 404


def test_rtf_asrf_matches_irb_formula() -> None:
    """The stdlib ASRF credit UL at 99.9% with scale 1 equals the numpy IRB K (same CRR formula)."""
    pds = np.array([0.001, 0.01, 0.05, 0.2])
    shares = np.array([0.4, 0.3, 0.2, 0.1])
    k = float((irb_retail_k(pds, 0.45) * shares * 1000).sum())
    ul = credit_ul(list(zip(pds.tolist(), shares.tolist(), strict=True)), 1000, 0.45, 0.999, 1.0)["ul"]
    assert ul == pytest.approx(k, rel=1e-9)


def test_rtf_endpoint_defaults_and_override() -> None:
    base = client.get("/rtf")
    assert base.status_code == 200
    b = base.json()
    assert b["traffic_light"] in {"green", "amber", "red"}
    worse = client.get("/rtf", params={"lgd": 0.9}).json()
    assert worse["ec"]["credit"] > b["ec"]["credit"]
    assert client.get("/rtf", params={"lgd": 2}).status_code == 422
