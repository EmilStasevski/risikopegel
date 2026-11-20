# Data

Every dataset gets one entry here: source, date downloaded, license, number of rows, and known biases. The raw data lives in `data/raw/` and is never committed, so the repository does not redistribute it.

**Status 2026-11-20:** nothing has been downloaded yet. The table below lists the candidates with license checks from a web search on 2026-11-20. Each license is checked again on the source page itself at download time.

## Candidates

| Dataset | Use | License / terms (as found) | Confidence | Size | Notes |
|---|---|---|---|---|---|
| UCI Default of Credit Card Clients (Taiwan, id 350) | PD, main credit dataset | CC BY 4.0 on the UCI page. Some mirrors say CC0; UCI is authoritative. Cite Yeh (2009), DOI 10.24432/C55S3H | high | 30,000 rows, ~5 MB | Taiwan, 2005, not a German population. Covers only 6 months, so the out-of-time split is weak |
| UCI Statlog German Credit (id 144) | smoke-test scorecard | CC BY 4.0 (seen on the old UCI page). DOI 10.24432/C5NC77 | medium-high | 1,000 rows | UCI warns of coding errors. Prefer **South German Credit (id 522/573, CC BY 4.0)**, the corrected version. Too small for conclusions |
| Give Me Some Credit (Kaggle) | PD, second dataset | Kaggle competition rules. Needs login and accepting the terms. Not checked | low | 150,000 rows, ~7 MB | Has no time column, so no out-of-time split is possible |
| Home Credit Default Risk (Kaggle) | PD, richest features | Kaggle competition rules; I couldn't find the exact text. Kaggle ToS: personal, non-commercial use. No rehosting | low | ~2.7 GB uncompressed, **above the 1 GB ask threshold** | Relational tables, good for dbt marts. No real application dates |
| ECB Data Portal (rates, FX, HICP, …) | market risk (FX/rates), macro | ESCB reuse policy: free reuse, cite "Source: ECB statistics", no modification. Third-party data excluded | high | small (API) | Cite every series key |
| Bundesbank time-series database | macro, rates | Same ESCB reuse policy. Website terms: do not alter or distort | high | small (API) | Cite every series key |
| Eurostat / Destatis GENESIS | macro (insolvencies, IP, unemployment) | Eurostat: free reuse with attribution. Destatis: dl-de/by-2-0. **To verify** | medium | small | Destatis API may need a free account |
| FRED | macro (cross-checks) | Each series has its own terms. Some are third-party copyrighted | medium | small | Check every series |
| Stooq (DAX constituents and index) | market-risk equity portfolio | Stooq's own terms not found. Third-party summaries say "free, non-commercial". STOXX/Deutsche Börse own the DAX index terms | **low** | small | Only after the user OKs it and the terms page has been read. The fallback is an FX/rates portfolio from ECB data |
| ECB/Bundesbank speeches (text) | Module D text signal | ECB publishes a speeches dataset (CSV). Check its terms and robots.txt before use | medium | small | Use the published dataset instead of scraping |

Sources for these checks: [UCI 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [UCI 144](https://archive.ics.uci.edu/dataset/144), [UCI 522 South German Credit](https://archive.ics.uci.edu/dataset/522/south+german+credit), [ECB usage policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html), [Bundesbank ESCB reuse terms](https://www.bundesbank.de/en/homepage/user-information/terms-of-use-regarding-the-reuse-of-escb-statistics-621188), [Kaggle terms](https://kaggle.com/terms), [STOXX DAX conditions of use](https://stoxx.com/legal/dax-conditions-of-use/).

## User approval (2026-11-20)

- **Approved:** ECB/Bundesbank/Eurostat APIs; Kaggle Give Me Some Credit and Home Credit (Home Credit's 2.7 GB was approved explicitly); Stooq DAX prices (download only after Stooq's terms page has been read and recorded here).
- **Not approved:** the UCI sets (Taiwan, South German Credit). So the smoke-test scorecard (checklist step 4) moves to Give Me Some Credit.
- **Blocked on:** Kaggle API token (`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`) and accepting both competitions' rules on kaggle.com. Only the user can do either.

## Downloaded datasets

_(none yet)_
