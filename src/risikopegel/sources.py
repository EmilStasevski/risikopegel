"""Fetch statistical series from the ECB Data Portal, the Bundesbank and Eurostat (SDMX REST APIs).

Raw responses are stored unmodified in data/raw/<source>/<id>.csv with a sidecar <id>.meta.json
(URL, UTC fetch time, SHA-256, rows). The ESCB reuse policy requires citing the source and not modifying
the statistics; all transformations happen downstream on the parsed frame.

Parsed form: DataFrame[period (str, as published), value (float, NaN = missing)].
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

USER_AGENT = "risikopegel-research/0.1"  # generic client name, no personal data
RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class Series:
    id: str
    source: str  # ecb | bundesbank | eurostat
    flow: str
    key: str
    desc: str
    module: str


BASE_URLS = {
    "ecb": "https://data-api.ecb.europa.eu/service/data/{flow}/{key}",
    "bundesbank": "https://api.statistiken.bundesbank.de/rest/data/{flow}/{key}",
    "eurostat": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{flow}/{key}",
}
FORMAT_PARAM = {"ecb": "csvdata", "bundesbank": "sdmx_csv", "eurostat": "SDMX-CSV"}
CITATION = {
    "ecb": "Source: ECB statistics (ECB Data Portal)",
    "bundesbank": "Source: Deutsche Bundesbank time series database",
    "eurostat": "Source: Eurostat",
}


def load_catalog(path: Path = Path("config/series.toml")) -> list[Series]:
    raw = tomllib.loads(path.read_text())
    out = [Series(**s) for s in raw["series"]]
    ids = [s.id for s in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate series ids in catalog")
    unknown = {s.source for s in out} - set(BASE_URLS)
    if unknown:
        raise ValueError(f"unknown sources: {unknown}")
    return out


def build_url(s: Series, start: str | None = None, end: str | None = None) -> str:
    params = {"format": FORMAT_PARAM[s.source]}
    if start:
        params["startPeriod"] = start
    if end:
        params["endPeriod"] = end
    return BASE_URLS[s.source].format(flow=s.flow, key=s.key) + "?" + urllib.parse.urlencode(params)


def http_get(url: str, *, retries: int = 3, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body: bytes = resp.read()
                return body
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def parse(source: str, body: bytes) -> pd.DataFrame:
    """Parse a raw response into [period, value]."""
    text = body.decode("utf-8-sig")  # Bundesbank sends a BOM
    if not text.strip():
        raise ValueError("empty response (unknown key or no data in window)")
    if source == "bundesbank":
        df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, keep_default_na=False)
        missing = {".", ""}  # Bundesbank writes '.' for missing observations
    else:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        missing = {"", "NaN"}
    if not {"TIME_PERIOD", "OBS_VALUE"} <= set(df.columns):
        raise ValueError(f"unexpected columns: {list(df.columns)[:8]}")
    vals = df["OBS_VALUE"].where(~df["OBS_VALUE"].isin(missing))
    out = pd.DataFrame({"period": df["TIME_PERIOD"], "value": pd.to_numeric(vals, errors="raise")})
    if out["period"].duplicated().any():
        raise ValueError("duplicate periods: key matched more than one series")
    return out.sort_values("period", ignore_index=True)


def fetch(
    s: Series, *, start: str | None = None, end: str | None = None, raw_dir: Path = RAW_DIR, save: bool = True
) -> pd.DataFrame:
    url = build_url(s, start, end)
    body = http_get(url)
    df = parse(s.source, body)
    if save:
        d = raw_dir / s.source
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{s.id}.csv").write_bytes(body)
        meta = {
            "id": s.id,
            "source": s.source,
            "flow": s.flow,
            "key": s.key,
            "desc": s.desc,
            "url": url,
            "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(body).hexdigest(),
            "rows": len(df),
            "first_period": df["period"].iloc[0],
            "last_period": df["period"].iloc[-1],
            "n_missing": int(df["value"].isna().sum()),
            "citation": CITATION[s.source],
        }
        (d / f"{s.id}.meta.json").write_text(json.dumps(meta, indent=2))
    return df


def load_raw(s: Series, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Parse a previously fetched raw file (offline)."""
    return parse(s.source, (raw_dir / s.source / f"{s.id}.csv").read_bytes())
