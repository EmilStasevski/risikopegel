"""Offline tests for the SDMX fetchers (format samples copied from real responses, 2026-11-20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from risikopegel.sources import Series, build_url, load_catalog, parse

ECB = (
    b"KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS,TITLE\n"
    b'EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-03,1.0919,A,"US dollar/Euro, ECB"\n'
    b'EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,1.0956,A,"US dollar/Euro, ECB"\n'
    b'EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-04,,M,"US dollar/Euro, ECB"\n'
)
BBK = (
    "\ufeffDATAFLOW;BBK_STD_FREQ;TIME_PERIOD;OBS_VALUE;BBK_TITLE\n"
    "BBK:BBEX3(1.0);D;2024-01-01;.;Euro-Referenzkurs der EZB / 1 EUR = ... USD\n"
    "BBK:BBEX3(1.0);D;2024-01-02;1.0956;Euro-Referenzkurs der EZB / 1 EUR = ... USD\n"
).encode()
ESTAT = (
    b"DATAFLOW,LAST UPDATE,freq,s_adj,age,unit,sex,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG,CONF_STATUS\n"
    b"ESTAT:UNE_RT_M(1.0),19/11/26 23:00:00,M,SA,TOTAL,PC_ACT,T,DE,2024-01,3.2,,\n"
    b"ESTAT:UNE_RT_M(1.0),19/11/26 23:00:00,M,SA,TOTAL,PC_ACT,T,DE,2024-02,3.3,p,\n"
)


def test_parse_ecb_sorts_and_keeps_missing_as_nan() -> None:
    df = parse("ecb", ECB)
    assert df["period"].tolist() == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert df["value"].iloc[0] == 1.0956
    assert df["value"].isna().tolist() == [False, False, True]


def test_parse_bundesbank_bom_semicolon_and_dot_missing() -> None:
    df = parse("bundesbank", BBK)
    assert df["value"].isna().tolist() == [True, False]
    assert df["value"].iloc[1] == 1.0956


def test_parse_eurostat() -> None:
    df = parse("eurostat", ESTAT)
    assert df["value"].tolist() == [3.2, 3.3]


def test_parse_rejects_multi_series_and_empty() -> None:
    dup = ECB + b"EXR.D.GBP.EUR.SP00.A,D,GBP,EUR,SP00,A,2024-01-02,0.86,A,x\n"
    with pytest.raises(ValueError, match="duplicate periods"):
        parse("ecb", dup)
    with pytest.raises(ValueError, match="empty"):
        parse("ecb", b"")


def test_build_url() -> None:
    s = Series("x", "eurostat", "une_rt_m", "M.SA.TOTAL.PC_ACT.T.DE", "", "macro")
    url = build_url(s, "2024-01", None)
    assert url == (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/une_rt_m/"
        "M.SA.TOTAL.PC_ACT.T.DE?format=SDMX-CSV&startPeriod=2024-01"
    )


def test_catalog_is_valid() -> None:
    cat = load_catalog(Path(__file__).resolve().parents[1] / "config/series.toml")
    assert len(cat) >= 10
    assert {s.module for s in cat} == {"market", "macro"}
