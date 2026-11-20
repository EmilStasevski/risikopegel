"""Fetch every series in config/series.toml.

uv run python scripts/fetch_series.py --check   # tiny window per series: verifies keys, saves nothing
uv run python scripts/fetch_series.py           # full history -> data/raw/<source>/ + docs table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from risikopegel.sources import RAW_DIR, fetch, load_catalog

CHECK_WINDOW = {
    "D": ("2024-01-02", "2024-01-10"),
    "B": ("2024-01-02", "2024-01-10"),
    "M": ("2024-01", "2024-03"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    failures = []
    for s in load_catalog():
        if args.only and s.id not in args.only:
            continue
        try:
            if args.check:
                start, end = CHECK_WINDOW[s.key.split(".")[0]]
                df = fetch(s, start=start, end=end, save=False)
            else:
                df = fetch(s)
            span = f"{df['period'].iloc[0]}..{df['period'].iloc[-1]}"
            print(f"ok   {s.id:16s} {s.source:10s} rows={len(df):6d} {span}")
        except Exception as e:  # report all failures, then exit non-zero
            failures.append(s.id)
            print(f"FAIL {s.id:16s} {s.source:10s} {type(e).__name__}: {str(e)[:120]}")
        time.sleep(0.5)  # be polite to the APIs
    if not args.check and not failures:
        rows = [json.loads(p.read_text()) for p in sorted(RAW_DIR.glob("*/*.meta.json"))]
        Path("results/data").mkdir(parents=True, exist_ok=True)
        Path("results/data/series_manifest.json").write_text(json.dumps(rows, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
