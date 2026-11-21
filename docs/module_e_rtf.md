# Module E: simplified Risikotragfähigkeit (RTF)

This module was added on 2026-11-21 (user decision; see `docs/versions.md`, Scope decisions). It answers Q5: **does a stylised bank, whose credit book is scored by the Module A model and whose trading book is measured by Module B, have enough risk cover in the economic perspective, and does its capital ratio stay above the minimum over 3 years in the normative perspective under the Module C scenarios?**

It is a teaching-scale model of an ICAAP. It is not a bank's RTF concept. Every balance-sheet figure, LGD, correlation and confidence level is a **labelled assumption** in `config/rtf.toml`. Every regulatory formula is checked before it is coded (table below).

## Design

1. **Stylised bank** (`config/rtf.toml`):
   - a retail credit book of N loans whose PDs are drawn from the Module A champion's test-set PD distribution (aggregate only, no rows leave `data/`);
   - a trading book equal to the Module B portfolio;
   - banking-book assets and liabilities by repricing bucket, for interest-rate risk;
   - own funds and a risk-cover potential.
2. **Economic perspective** (1-year horizon, confidence level an assumption, e.g. 99.9 %):
   - **Credit risk:** (a) the IRB retail risk-weight function as a closed-form benchmark; (b) a one-factor Gaussian-copula Monte Carlo on the loan book giving the loss distribution, expected loss (EL), unexpected loss and ES; (c) concentration and sensitivity to the asset correlation.
   - **Market risk:** ES/VaR from Module B, scaled to the 1-year horizon. The scaling is an assumption, with sensitivity shown.
   - **Interest-rate risk in the banking book (IRRBB):** ΔEVE and ΔNII under the supervisory shock scenarios on the ECB euro-area yield curve (ECB data, approved source).
   - **Operational risk:** a simple indicator-based charge, labelled as an assumption.
   - **Aggregation:** simple sum vs a variance-covariance approach with assumed correlations. Diversification benefit, limit use per risk type and a traffic light vs the risk cover.
3. **Normative perspective** (3 years): baseline and adverse paths for the capital ratio. Macro scenarios from Module C feed the macro-conditional PD (the Module A satellite model), which drives impairments (ECL) and risk-weighted assets.
4. **Stress tests:** a historical scenario (e.g. 2008/09 or 2020 macro paths), a hypothetical adverse scenario, and a **reverse stress test**: the PD multiplier or rate shock at which the risk cover is used up.
5. **Outputs:**
   - `results/rtf/*.json`;
   - `reports/rtf.md` (RTF table, limit use, sensitivities);
   - an API endpoint `/rtf` that recomputes the economic perspective for user-supplied assumptions (deployed on Azure, see `docs/azure.md`).

## Regulatory facts: status

| Fact | Status | Source |
|---|---|---|
| RTF guideline: normative perspective (capital planning ≥ 3 years) and economic perspective (1 year, internal methods); MaRisk AT 4.1 Tz. 2 | verified 2026-11-21 (whether the 2018 guideline is still unchanged is not checked) | BaFin/Bundesbank RTF-Leitfaden, 24 May 2018 |
| CRR Art. 154(3): retail exposures secured by immovable property use correlation R = 0.15; Art. 154(4): qualifying revolving retail exposures use R = 0.04 | **verified 2026-11-21**, verbatim from the 2013 adopted text (later revisions not read) | [CRR Art. 154](https://legislation.gov.uk/eur/2013/575/article/154/adopted) |
| CRR Art. 154(1): "other retail" correlation R = 0.03·(1−e^(−35·PD))/(1−e^(−35)) + 0.16·[1 − (1−e^(−35·PD))/(1−e^(−35))]; RW = (LGD·N[(G(PD) + √R·G(0.999))/√(1−R)] − LGD·PD)·12.5·1.06; no maturity adjustment for retail | **open**: the formulas are images on the official page, and only a secondary summary was read. Must be checked against the CRR text (EUR-Lex) before implementation | secondary summary only |
| IRRBB: six supervisory shock scenarios for EVE (parallel up/down, steepener, flattener, short up/down), two for NII (parallel up/down); EVE outlier test at 15 % of Tier 1; decay parameter x = 4 | verified 2026-11-21 from the BCBS Basel Framework SRP31 (in force 2026-01-01) via a page summary. The EU version is in Delegated Regulation (EU) 2024/856 (not read; EUR-Lex fetch failed) | [BCBS SRP31](https://www.bis.org/committees/bcbs/basel-framework/standard/srp/31/inforce/2026-01-01/published/2024-07-16) |
| EUR shock sizes (bp) for parallel / short / long | **open**: SRP31 (2026) apparently gives 225 / 350 / 200 (from a summary, not verbatim). This differs from the 200 / 250 / 100 recalled from the 2016 standard, which was not verified. The EU 2024/856 calibration is unknown. Resolve before implementation | |
| Post-shock interest-rate floor; NII outlier threshold | **open** | |
