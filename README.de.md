# Risikopegel

[English version](README.md)

Eine Modellrisiko-Plattform im Stil bankfachlicher Modelldokumentation: Kreditrisiko, Marktpreisrisiko, Makroprognosen, eine Sicht auf die Risikotragfähigkeit (RTF/ICAAP) einer Musterbank und ein LLM-Modul, dessen Ausgaben deterministisch geprüft werden. Alles läuft lokal. Jede Zahl in diesem README und in `reports/` wird mit `task report` aus `results/` erzeugt.

> **Keine Anlageberatung.** Das Projekt misst Risiken, es ist kein Handelssystem. Es macht keine Renditeprognosen und erhebt keinen Anspruch auf aufsichtsrechtliche Konformität (Basel, CRR/CRD, IFRS 9, KI-Verordnung, DSGVO). Wo eine aufsichtliche Formel verwendet wird, ist sie am Primärtext geprüft (`docs/versions.md`, `docs/module_e_rtf.md`).

## Kernergebnisse

<!-- BEGIN GENERATED: scripts/report_readme.py -->

| Modul | Kernergebnis (alle Zahlen aus `results/` erzeugt) |
|---|---|
| A, Kredit-PD (Home Credit) | lightgbm vs. WoE-Scorecard: Gini 0,562 vs. 0,517, ΔAUC 0,023 [0,019; 0,026]; Kalibrierungssteigung 0,99 vs. 0,99, ECE 0,0041 vs. 0,0020; Score-PSI 0,0043 vs. 0,0006 |
| B, VaR/ES-Backtesting | 10 Modelle, EZB-FX/Zins-Portfolio, 2008-08-01 bis 2026-11-19: 99-%-VaR besteht Kupiec + Christoffersen in 17/19 Jahren mit FHS GARCH-t vs. 10 (historische Simulation), 11 (Normalverteilung), höchstens 12 (ML) |
| C, Makroprognosen | 15 Eurostat-Reihen, 8 Modelle inkl. Chronos-2, Chronos-Bolt, TimesFM 2.5: MASE h=1 TimesFM 0,315 vs. ETS 0,324 (DM p = 0,27); Energie je Prognose ETS 0,29 J, TimesFM 1,78 J, Chronos-Bolt 0,10 J |
| A+C, Szenario-ECL | makrobedingte PD: ECL-Deckung 4,01 % (Basis) vs. 5,52 % (adverses Szenario) |
| E, Risikotragfähigkeit | Musterbank: Auslastung des Risikodeckungspotenzials 73 % (grün); minimale harte Kernkapitalquote adverses Szenario 17,6 % vs. 7,84 % Anforderung; inverser Stresstest: PD × 3,41; IRRBB-Ausreißer (EVE): ja |
| D, LLM-Erklärungen | 200 deutsche Erklärungen: 91,0 % modelltreu (zero-shot), 98,5 % mit Verifier + Retry, 100,0 % der ausgelieferten Texte verifiziert |
| D, EZB-Tonalität | bestes Out-of-Sample-MSE-Verhältnis 0,999 (llm, dy2; DM p = 0,54) |

<!-- END GENERATED -->

Vorab festgelegte Hypothesen und Ergebnisse: [`docs/hypotheses.md`](docs/hypotheses.md) (englisch).

## Module und Berichte

| Modul | Fragestellung | Bericht (englisch) |
|---|---|---|
| A, Kreditrisiko | Schlägt Gradient Boosting eine WoE-Scorecard, und was kostet das an Kalibrierung, Stabilität und Erklärbarkeit? | [Challenger](reports/credit_hc_challengers.md), [Tuning](reports/credit_hc_tuning.md), [Model Card](reports/model_card_hc_pd.md), [unabhängige Validierung](reports/validation_hc_pd.md) |
| A+C, ECL | Wie verschieben Makroszenarien PD und erwarteten Kreditverlust? | [makrobedingte PD und Szenario-ECL](reports/credit_ecl.md) |
| B, Marktpreisrisiko | Welche VaR/ES-Methode besteht Backtests in ruhigen und gestressten Phasen, und bringt ML etwas? | [VaR/ES-Backtesting](reports/market_var_backtest.md) |
| C, Makroprognosen | Schlagen Foundation-Modelle klassische Verfahren, und zu welchem Rechenaufwand? | [Prognosevergleich](reports/macro_forecast.md) |
| D, LLM | Lässt sich eine LLM-generierte deutsche Kreditentscheidungs-Erklärung als modelltreu nachweisen? Hilft die Tonalität von EZB-Reden bei Volatilitätsprognosen? | [verifizierte Erklärungen](reports/llm_explanations.md), [Tonalitätssignal](reports/llm_text_signal.md) |
| E, Risikotragfähigkeit | Reicht das Risikodeckungspotenzial einer Musterbank auf Basis der Module A–C (ökonomische Perspektive), und hält die Kapitalquote über 3 Jahre (normative Perspektive)? | [RTF](reports/rtf.md) |

## Aufbau

- **Daten:** Kaggle Home Credit und Give Me Some Credit (Kredit), APIs von EZB, Bundesbank und Eurostat (Wechselkurse, Zinskurven, Makro), EZB-Redendatensatz. Lizenzen und Freigaben: [`docs/data.md`](docs/data.md). Rohdaten werden nie eingecheckt.
- **Disziplin:** ausschließlich zeitbasierte Splits; jede Modellpipeline hat einen Kausalitäts- und einen Shuffled-Label-Test (`pytest -m leakage`); aufsichtliche Fakten werden geprüft, nicht aus dem Gedächtnis übernommen.
- **Technik:** Python 3.12 mit uv, ruff, mypy `--strict`, pytest, dbt + DuckDB, LightGBM/XGBoost/CatBoost/EBM/optbinning, arch/statsmodels, Chronos-2, Chronos-Bolt und TimesFM 2.5 (CPU), FinBERT, Qwen3 4B über lokales Ollama, FastAPI + Docker, Streamlit, GitHub Actions.
- **Energie:** CPU über Intel RAPL, GPU über `nvidia-smi`, im Repository selbst gemessen.

## Schnellstart

```bash
uv sync
task test        # Unit- und Leakage-Tests (offline)
task lint        # ruff + mypy --strict
task report      # alle Berichte und die README-Tabellen aus results/ neu erzeugen
task dashboard   # Streamlit-Ansicht
```
