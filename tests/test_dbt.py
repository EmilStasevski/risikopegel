"""Runs `dbt build` (models + data tests) on synthetic fixtures. Offline; ~15 s."""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pytest
from dbt.artifacts.schemas.run import RunExecutionResult
from dbt.cli.main import dbtRunner

from tests.dbt_fixtures import write_fixtures

DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"


def _build(raw: Path, db: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[bool, list[str]]:
    monkeypatch.setenv("RISIKOPEGEL_DUCKDB", str(db))
    monkeypatch.chdir(DBT_DIR)
    res = dbtRunner().invoke(
        [
            "build",
            "--profiles-dir", str(DBT_DIR),
            "--project-dir", str(DBT_DIR),
            "--target-path", str(db.parent / "target"),
            "--vars", json.dumps({"raw_dir": str(raw)}),
            "--quiet",
        ]
    )  # fmt: skip
    run = res.result
    assert isinstance(run, RunExecutionResult), type(run)
    failed = [r.node.name for r in run.results if str(r.status) in {"fail", "error"}]
    return res.success, failed


def test_dbt_build_on_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = write_fixtures(tmp_path / "raw")
    db = tmp_path / "rp.duckdb"
    ok, failed = _build(raw, db, monkeypatch)
    assert ok, failed
    con = duckdb.connect(str(db))  # same config as the connection dbt-duckdb keeps open in-process

    def one(sql: str) -> tuple[object, ...]:
        row = con.execute(sql).fetchone()
        assert row is not None
        return tuple(row)

    assert one("select count(*), count(*) filter (where is_train) from mart_hc__applicant_features") == (
        180,
        150,
    )
    # bureau rows updated after the application and bureau_balance month 0 are excluded
    assert one("select count(*) from stg_hc__bureau") == (298,)
    assert one("select max(months_balance) from stg_hc__bureau_balance") == (-1,)
    assert one("select count(*) from mart_gmsc__applicants where dpd_special_code") == (3,)
    assert one("select count(*) > 0 from mart_gmsc__applicants where monthly_income is null") == (True,)


@pytest.mark.leakage
def test_dbt_tests_catch_payment_after_application(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = write_fixtures(tmp_path / "raw", inject_future_payment=True)
    ok, failed = _build(raw, tmp_path / "rp.duckdb", monkeypatch)
    assert not ok
    assert any("days_entry_payment" in name for name in failed), failed


if os.environ.get("RISIKOPEGEL_SKIP_DBT"):
    pytest.skip("dbt tests disabled via RISIKOPEGEL_SKIP_DBT", allow_module_level=True)
