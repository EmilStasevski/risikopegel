from fastapi.testclient import TestClient

from risikopegel.api.app import app

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
