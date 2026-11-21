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
   - an API endpoint `/rtf` that recomputes the economic perspective for user-supplied assumptions.

## Regulatory facts: status

All rows below were checked on 2026-11-21 against the primary text. The PDFs were downloaded from EUR-Lex. Formula images were rendered and read page by page, not taken from summaries.

| Fact | Status | Source |
|---|---|---|
| RTF guideline: normative perspective (capital planning ≥ 3 years) and economic perspective (1 year, internal methods); MaRisk AT 4.1 Tz. 2 | verified (whether the 2018 guideline is still unchanged is not checked) | BaFin/Bundesbank RTF-Leitfaden, 24 May 2018 |
| CRR Art. 154(1)(ii), retail, PD < 1: RW = (LGD·N[G(PD)/√(1−R) + √(R/(1−R))·G(0.999)] − LGD·PD)·12.5. **No 1.06 scaling factor** in the current text (amended by CRR3, marker M17). No maturity adjustment for retail | **verified**, consolidated CRR 26.06.2026, p. 234 | [CRR consolidated 26.06.2026](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02013R0575-20260626) |
| CRR Art. 154(1): other retail R = 0.03·(1−e^(−35·PD))/(1−e^(−35)) + 0.16·[1 − (1−e^(−35·PD))/(1−e^(−35))]; Art. 154(3): residential property R = 0.15; Art. 154(4): QRRE R = 0.04 | **verified**, pp. 234–235 | [CRR consolidated 26.06.2026](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02013R0575-20260626) |
| CRR Art. 163(1): PD input floor 0.05 % for retail other than QRRE revolvers (0.1 % for QRRE revolvers). Art. 164(4) Table 1: LGD input floor 30 % for unsecured other retail | **verified**, pp. 246–248 | [CRR consolidated 26.06.2026](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02013R0575-20260626) |
| CRR Art. 92(1): CET1 4.5 %, Tier 1 6 %, total capital 8 %, leverage 3 %. CRD Art. 129: capital conservation buffer 2.5 % CET1 | **verified** | [CRR consolidated 26.06.2026](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02013R0575-20260626); [CRD consolidated 17.01.2025](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02013L0036-20250117) |
| CRR Art. 312–313 (operational risk, CRR3): own funds requirement = BIC = 0.12·BI for BI ≤ €1 bn (0.12 + 0.15·(BI−1) up to €30 bn) | **verified**, p. 450 | [CRR consolidated 26.06.2026](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02013R0575-20260626) |
| IRRBB, Delegated Reg. (EU) 2024/856. Art. 1: six EVE scenarios (parallel up/down, steepener, flattener, short up/down), NII uses the two parallel ones. Art. 2: ΔR_short(t) = ±R_short·e^(−t/4), ΔR_long(t) = ±R_long·(1−e^(−t/4)), steepener = −0.65·abs(ΔR_short) + 0.9·abs(ΔR_long), flattener = +0.8·abs(ΔR_short) − 0.6·abs(ΔR_long) | **verified** (OJ L 24.4.2024) | [Reg. (EU) 2024/856](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R0856) |
| EUR shock sizes, Annex Part A: parallel 200 bp, short 250 bp, long 100 bp. These apply in the EU. BCBS SRP31 (in force 2026) recalibrated EUR, apparently to 225/350/200 per a page summary; that is not used here | **verified** (EU); BCBS values not verified | same |
| Art. 3(6) run-off balance sheet; Art. 3(7) post-shock floor −150 bp at immediate maturity, +3 bp per year, 0 % from 50 years; Art. 3(8) positive EVE changes weighted at 50 % | **verified** | same |
| Outlier thresholds: ΔEVE decline > 15 % of Tier 1 (CRD Art. 98(5)(a)); NII decline > 5 % of Tier 1 (Reg. 2024/856 Art. 5(1)) | **verified** | CRD consolidated 17.01.2025; Reg. 2024/856 |
