# Data

Every dataset gets one entry here: source, date downloaded, license, number of rows, and known biases. The raw data lives in `data/raw/` and is never committed, so the repository does not redistribute it.

**Status 2026-11-21:** the datasets and models under "Downloaded datasets" are in use. The candidates table below is the license check from 2026-11-20 (web search), kept for the record. Each license was checked again on the source page itself at download time.

## Candidates

| Dataset | Use | License / terms (as found) | Confidence | Size | Notes |
|---|---|---|---|---|---|
| UCI Default of Credit Card Clients (Taiwan, id 350) | PD, main credit dataset | CC BY 4.0 on the UCI page. Some mirrors say CC0; UCI is authoritative. Cite Yeh (2009), DOI 10.24432/C55S3H | high | 30,000 rows, ~5 MB | Taiwan, 2005, not a German population. Covers only 6 months, so the out-of-time split is weak |
| UCI Statlog German Credit (id 144) | smoke-test scorecard | CC BY 4.0 (seen on the old UCI page). DOI 10.24432/C5NC77 | medium-high | 1,000 rows | UCI warns of coding errors. Prefer **South German Credit (id 522/573, CC BY 4.0)**, the corrected version. Too small for conclusions |
| Give Me Some Credit (Kaggle) | PD, second dataset | Kaggle competition rules. Needs login and accepting the terms. Not checked | low | 150,000 rows, ~7 MB | Has no time column, so no out-of-time split is possible |
| Home Credit Default Risk (Kaggle) | PD, richest features | Kaggle competition rules; I couldn't find the exact text. Kaggle ToS: personal, non-commercial use. No rehosting | low | ~2.7 GB uncompressed, **above the 1 GB ask threshold** | Relational tables, good for dbt marts. No real application dates |
| ECB Data Portal (rates, FX, HICP, …) | market risk (FX/rates), macro | ESCB reuse policy: free reuse, cite "Source: ECB statistics", no modification. Third-party data excluded | high | small (API) | Cite every series key |
| Bundesbank time-series database | macro, rates | Same ESCB reuse policy. Website terms: do not alter or distort | high | small (API) | Cite every series key |
| Eurostat | macro (HICP, IP, unemployment, retail) | **Verified 2026-11-20:** reuse is authorised, commercial or not, free of charge, if the source is acknowledged (Commission Decision 2011/833/EU, Eurostat copyright notice). Non-Eurostat source data and documents with a restrictive notice are excluded | high | small | |
| Destatis GENESIS | insolvencies (optional) | dl-de/by-2-0, **to verify** | medium | small | API may need a free account; not used yet |
| FRED | macro (cross-checks) | Each series has its own terms. Some are third-party copyrighted | medium | small | Check every series |
| Stooq (DAX constituents and index) | market-risk equity portfolio | Stooq's own terms not found. Third-party summaries say "free, non-commercial". STOXX/Deutsche Börse own the DAX index terms | **low** | small | Only after the user OKs it and the terms page has been read. The fallback is an FX/rates portfolio from ECB data |
| ECB/Bundesbank speeches (text) | Module D text signal | ECB publishes a speeches dataset (CSV). Check its terms and robots.txt before use | medium | small | Use the published dataset instead of scraping |

Sources for these checks: [Eurostat copyright notice](https://ec.europa.eu/eurostat/help/copyright-notice), [Decision 2011/833/EU](https://eur-lex.europa.eu/eli/dec/2011/833/oj), [UCI 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [UCI 144](https://archive.ics.uci.edu/dataset/144), [UCI 522 South German Credit](https://archive.ics.uci.edu/dataset/522/south+german+credit), [ECB usage policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html), [Bundesbank ESCB reuse terms](https://www.bundesbank.de/en/homepage/user-information/terms-of-use-regarding-the-reuse-of-escb-statistics-621188), [Kaggle terms](https://kaggle.com/terms), [STOXX DAX conditions of use](https://stoxx.com/legal/dax-conditions-of-use/).

## User approval (2026-11-20)

- **Approved:** ECB/Bundesbank/Eurostat APIs; Kaggle Give Me Some Credit and Home Credit (Home Credit's 2.7 GB was approved explicitly); Stooq DAX prices (download only after Stooq's terms page has been read and recorded here).
- **Not approved:** the UCI sets (Taiwan, South German Credit). So the smoke-test scorecard (checklist step 4) moves to Give Me Some Credit.
- Kaggle access: the user set up an API token (`~/.kaggle/access_token`) and accepted the competition rules on 2026-11-20. Downloads use Kaggle CLI 2.2.5 through `uvx` (`task data:gmsc`).

## Downloaded datasets

### Give Me Some Credit (Kaggle `GiveMeSomeCredit`)

| Field | Value |
|---|---|
| Downloaded | 2026-11-20, `task data:gmsc` (Kaggle CLI 2.2.5), `GiveMeSomeCredit.zip` 5.4 MB |
| Terms | Kaggle competition rules, accepted by the user; Kaggle ToS: personal, non-commercial use. Not redistributed, `data/` is git-ignored |
| Files | `cs-training.csv` (150,000 labelled rows, sha256 `1bd46da4…24ac`), `cs-test.csv` (101,503 rows, **no labels**, sha256 `bab363a2…5b78`), `Data Dictionary.xls` |
| Target | `SeriousDlqin2yrs` (90+ days past due within 2 years), default rate 6.684% in cs-training |
| Missing values | `MonthlyIncome` 29,731 rows; `NumberOfDependents` 3,924 rows |
| Quirks (verified) | All three past-due counts hold codes **96/98** in the same 269 rows, so they are treated as special codes, not counts. One row has `age == 0`. 609 exact duplicate rows (kept, because they may be real look-alike applicants) |
| Known biases / limits | US consumer credit, collection period and population not documented, no application date, so **no out-of-time split** is possible and stability/drift can't be tested. Not a German population |
| Public reference | Top Kaggle leaderboard AUC 0.86955 (Kaggle CLI leaderboard, fetched 2026-11-20). It's measured on the unlabelled test set, so it can only be compared after a late submission |


### Home Credit Default Risk (Kaggle `home-credit-default-risk`)

| Field | Value |
|---|---|
| Downloaded | 2026-11-20, Kaggle CLI 2.2.5. Zip 722 MB, 3.2 GB on disk unpacked. The first attempt failed on a DNS error; the retry succeeded |
| Terms | Kaggle competition rules, accepted by the user; personal, non-commercial use. Not redistributed |
| Files (lines incl. header, `wc -l`) | application_train 307,512 · application_test 48,745 (no labels) · bureau 1,716,429 · bureau_balance 27,299,926 · previous_application 1,670,215 · POS_CASH_balance 10,001,359 · credit_card_balance 3,840,313 · installments_payments 13,605,402 |
| Checksums | `data/raw/homecredit/SHA256SUMS` |
| Leakage-relevant quirks (verified) | `DAYS_EMPLOYED = 365243` placeholder in 55,374 train rows, set to null + flag. In `bureau`, 17 rows have `DAYS_CREDIT_UPDATE > 0` (debt/overdue as of a date after the application), dropped. In `bureau_balance`, 610,965 rows have `MONTHS_BALANCE = 0` (the application month, may extend past the application), dropped. `DAYS_FIRST_DUE`/`DAYS_LAST_DUE = 365243` placeholders in previous applications (not used). No installment payment is dated after the application (0 rows), and a dbt test enforces this |
| Time proxy (R1 analysis, `reports/credit_hc_time_proxy.md`) | **None found.** `SK_ID_CURR` is unrelated to default (chi-square by decile p = 0.68), and features can't separate low from high IDs (adversarial AUC 0.500). Train and test ID ranges overlap completely |
| bureau_balance coverage | Among applicants with bureau records, monthly bureau_balance history exists for 34.9% in train but 99.8% in the Kaggle test set. That's a **data-coverage artefact**, so the two `bureau_worst_dpd_bucket*` features shift heavily (CSI ≈ 1.55) without any change in borrowers. **Resolution (user decision, 2026-11-20):** the missingness is modelled explicitly with `bureau_history_status` (no bureau / bureau without history / with history) and `bureau_history_coverage`. These columns now flag the artefact directly in monitoring (CSI 3.94 / 1.39), but the R2 champion barely uses them: importance ranks 45–58 of 61 (`results/credit/hc_explain/global_importance.csv`). So the score PSI stays small |
| Naming issue (fixed) | The mart column formerly called `credit_term_years` is the credit/annuity ratio, now `credit_to_annuity` |
| Train vs Kaggle test | **Shifted population:** adversarial AUC 0.934 (0.900 without the bureau_balance features, 0.865 also without credit/annuity). Test has 0.9% revolving loans vs 9.5% in train, plus different loan terms and amounts. Could be a later period or a policy change; the data can't distinguish these |
| Known limits | No application dates, so in-sample evaluation is a stratified split (documented exception) and no out-of-time claims are possible. Population: Home Credit customers, mainly unbanked/underbanked consumers. Not German |

### ECB / Bundesbank / Eurostat series

| Field | Value |
|---|---|
| Downloaded | 2026-11-20, `task data:series` (`scripts/fetch_series.py`, catalog `config/series.toml`), 14 series. The table with periods and counts is generated in `reports/data_series.md` |
| Terms | ESCB reuse policy (cite the source, no modification) for ECB and Bundesbank; Eurostat copyright notice (acknowledge the source). Raw responses are stored unmodified with sidecar metadata (URL, time, SHA-256) |
| Verified while fetching | (1) Eurostat **rebased HICP to 2025 = 100 with COICOP 2018 codes** (`prc_hicp_minr`, all-items code `TOTAL`). The 2015 = 100 series (ECB `ICP/…INX`, Eurostat `prc_hicp_midx` I15) end in 2025-12, so the catalog uses the new dataset. (2) The euro-area aggregate for `une_rt_m` is `EA21` (Bulgaria joined on 2026-01-01). `EA20`/`EA19` return HTTP 400. (3) The industrial-production indicator code is `PRD`, not `PROD`. (4) The ECB and Bundesbank USD reference rates are identical on all 7,141 common days. The Bundesbank series also contains holiday/weekend rows marked `.` (missing) |
| Known limits | Monthly macro series are final *revised* values, not real-time vintages, so forecasting backtests in Module C carry mild look-ahead from data revisions. This will be stated in Module C, or handled with the Bundesbank real-time database if needed |

### Added series (2026-11-21)

Thirteen more series from the same approved sources and terms, fetched with `task data:series` (`--only`):
- ECB AAA yield-curve tenors 3m, 1y, 5y, 20y, 30y, for Module E discounting and IRRBB;
- Eurostat unemployment for FR, IT and ES; HICP for FR and IT; industrial production for FR and IT (Module C);
- Eurostat `sts_rb_m` German business bankruptcy declarations (`M.BKRT.B-S_X_O_S94.SCA.I21.DE`), the default-rate proxy for the macro-conditional PD. It only starts in **2015-01**, so it does not cover 2008/09.

Periods and counts are in `reports/data_series.md`. Known limit: bankruptcies are firm insolvencies, not retail defaults. Using them for the Home Credit book is an assumption, and the R4 report says so.

### ECB speeches dataset (Module D, H5)

| Field | Value |
|---|---|
| Source | ECB, "Download all speeches" (`https://www.ecb.europa.eu/press/key/html/downloads.en.html`), file `all_ECB_speeches.csv` (pipe-separated: date, speakers, title, subtitle, contents). Updated monthly by the ECB |
| Downloaded | 2026-11-21 (`data/raw/ecb_speeches/FETCHED_UTC`, `SHA256SUMS`), 58.7 MB. One request; robots.txt does not disallow the path |
| Approval | User approved "ECB speeches + FinBERT" on 2026-11-21 |
| Terms (checked 2026-11-21) | ECB disclaimer & copyright page: free use of information obtained from the site, with the ECB cited as the source. Reproducing documents that carry authors' names needs prior written authorisation. **Therefore no speech text is reproduced or committed.** Only numeric tone scores per speech date go to `results/` |
| Known limits | Executive Board speeches only, of differing type and audience. Non-English speeches are filtered out by a stopword heuristic. Tone is not a policy signal |

### Pretrained models (downloaded fresh, approved 2026-11-21; none from ZeilenJoule)

| Model | Use | License | Size | Pinned revision |
|---|---|---|---|---|
| amazon/chronos-2 | Module C zero-shot | Apache-2.0 | 478 MB | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| amazon/chronos-bolt-small | Module C zero-shot | Apache-2.0 | 191 MB | `772f3d25d38aec6d914c8949dab4462e2d46f5d8` |
| google/timesfm-2.5-200m-pytorch | Module C zero-shot | Apache-2.0 | 925 MB | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |
| ProsusAI/finbert | Module D tone (H5) | CC BY-NC-SA 4.0: non-commercial use only. This project is non-commercial and does not redistribute the model | 438 MB | `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` |
| qwen3:4b-instruct-2507-q4_K_M (Ollama) | Module D explanations and zero-shot tone | Apache-2.0 (Qwen3) | 2.5 GB, approved explicitly above the 1 GB threshold | Ollama digest recorded in each run's JSON. The tag `qwen3:4b` was pulled first, but it now points to the *thinking* variant (Qwen3-4B-Thinking-2507): ~40 s and ~4,000 tokens per explanation, with no way to turn thinking off. The user approved swapping it for the instruct variant (same size and license, second download); the thinking variant was deleted |

The local Ollama install already held other models (qwen3:8b/14b and others). Following the no-ZeilenJoule rule, none of them is used. qwen3:4b was pulled fresh for this project.
