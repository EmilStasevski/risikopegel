"""R5 (Module D part 1): German credit explanations from SHAP facts, checked by a deterministic verifier.

Protocol (fixed before the run):
- 200 test applicants, 20 per champion-PD decile (seeded)
- facts per applicant: the 4 features with the largest |TreeSHAP| contribution, excluding protected-like
  attributes (labels.SENSITIVE; how often one was in the raw top 4 is reported), with direction, and the
  PD rounded to a whole percent
- variants: template (deterministic), zero-shot LLM, few-shot LLM, few-shot + verifier with up to 2 retries
  that feed the verifier's errors back (if all fail, the template is shipped; counted separately)
- LLM: qwen3:4b via local Ollama (no data leaves the machine), thinking off, temperature 0.7, seeded
- metrics: pass rate, hallucination rate (unseen feature or number), wrong-direction rate, readability
  (Amstad), words, tokens, seconds and GPU joules per explanation
- verifier check: deliberate mutations of template texts must all be caught

Writes results/llm/explanations/{summary.json, texts.csv}.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from r2_challengers import ART, OUT, SEED

from risikopegel.credit.explain import explain
from risikopegel.credit.hc import load_mart, prepare
from risikopegel.llm.labels import LABELS, SENSITIVE
from risikopegel.llm.ollama import GpuPower, chat, model_digest
from risikopegel.llm.verifier import Facts, amstad, verify

RES = Path("results/llm/explanations")
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
N_PER_DECILE = 20
TOP_K = 4
MAX_RETRIES = 2

SYSTEM = (
    "Du bist ein Assistent einer Bank. Du schreibst kurze, sachliche Erklärungen für Kundinnen und Kunden, "
    "welche Faktoren die Risikoeinschätzung ihres Kreditantrags beeinflusst haben.\n"
    "Regeln:\n"
    "1. Verwende nur die genannten Faktoren und übernimm ihre Bezeichnung wörtlich.\n"
    "2. Schreibe für jeden Faktor genau einen eigenen Satz und sage in diesem Satz ausdrücklich, ob der Faktor "
    "das Risiko erhöht oder senkt.\n"
    "3. Nenne keine Zahlen außer der angegebenen Ausfallwahrscheinlichkeit.\n"
    "4. Keine Aussage über Genehmigung oder Ablehnung, keine Garantien, keine rechtlichen Hinweise.\n"
    "5. Sprich die Person mit Sie an. Insgesamt 4 bis 6 Sätze, nur der Text, keine Überschrift, keine Liste."
)


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def facts_prompt(f: Facts) -> str:
    lines = [f"Geschätzte Ausfallwahrscheinlichkeit: {f.pd_percent} %.", "Faktoren (wichtigster zuerst):"]
    for feat, d in f.directions.items():
        lines.append(f"- {LABELS[feat]}: {'erhöht' if d > 0 else 'senkt'} das Risiko")
    return "\n".join(lines)


def template(f: Facts) -> str:
    out = [f"Die geschätzte Ausfallwahrscheinlichkeit für Ihren Antrag liegt bei {f.pd_percent} %."]
    out += [f"{cap(LABELS[k])} {'erhöht' if d > 0 else 'senkt'} das Risiko." for k, d in f.directions.items()]
    return " ".join(out)


FEW_SHOT = [
    Facts({"ext_source_3": -1, "credit_to_annuity": 1, "days_employed": -1, "inst_share_late": 1}, 7),
    Facts({"ext_source_2": 1, "goods_price": -1, "occupation": 1, "prev_n_refused": 1}, 21),
]
FEW_SHOT_ANSWERS = [
    "Für Ihren Antrag schätzen wir eine Ausfallwahrscheinlichkeit von 7 %. Der Score der Auskunftei C senkt das "
    "Risiko deutlich. Die Laufzeit des Kredits erhöht das Risiko. Die Dauer Ihrer aktuellen Beschäftigung senkt "
    "das Risiko. Der Anteil verspäteter früherer Raten erhöht das Risiko.",
    "Wir schätzen die Ausfallwahrscheinlichkeit Ihres Antrags auf 21 %. Der Score der Auskunftei B erhöht das "
    "Risiko am stärksten. Der Preis der finanzierten Ware senkt das Risiko. Ihr Beruf erhöht das Risiko. Auch "
    "die Zahl Ihrer früher abgelehnten Anträge erhöht das Risiko.",
]


def messages(f: Facts, few_shot: bool) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for ex, ans in zip(FEW_SHOT, FEW_SHOT_ANSWERS, strict=True):
            msgs += [{"role": "user", "content": facts_prompt(ex)}, {"role": "assistant", "content": ans}]
    msgs.append({"role": "user", "content": facts_prompt(f)})
    return msgs


def build_facts() -> tuple[list[Facts], dict[str, Any]]:
    champ = json.loads((OUT / "results.json").read_text())["champion"]["name"]
    d = prepare(load_mart(), seed=SEED)
    model = pickle.loads((ART / f"{champ}.pkl").read_bytes())
    pdv = pd.read_parquet(ART / "test_predictions.parquet")[champ].to_numpy()
    dec = pd.qcut(pdv, 10, labels=False)
    rng = np.random.default_rng(SEED)
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(dec == k), N_PER_DECILE, replace=False) for k in range(10)]
    )
    ex = explain(champ, model, d.X_test.iloc[idx], d.categorical)
    facts, raw_sensitive = [], 0
    for i, row in enumerate(ex.contributions.itertuples(index=False)):
        c = pd.Series(row, index=ex.contributions.columns)
        order = c.abs().sort_values(ascending=False).index
        raw_sensitive += any(f in SENSITIVE for f in order[:TOP_K])
        top = [f for f in order if f not in SENSITIVE][:TOP_K]
        facts.append(Facts({f: int(np.sign(c[f])) or 1 for f in top}, round(100 * pdv[idx[i]])))
    return facts, {"champion": champ, "n": len(facts), "raw_top4_with_sensitive": raw_sensitive}


def mutation_check(facts: list[Facts]) -> dict[str, Any]:
    """Each mutation must make the verifier fail; reports the catch rate per mutation type."""
    other = [f for f in LABELS if f not in SENSITIVE]
    caught: dict[str, list[bool]] = {
        "flip_direction": [],
        "unseen_feature": [],
        "extra_number": [],
        "forbidden_claim": [],
        "no_direction": [],
        "garbled_label": [],
    }
    for f in facts:
        t = template(f)
        k0, d0 = next(iter(f.directions.items()))
        s0 = f"{cap(LABELS[k0])} {'erhöht' if d0 > 0 else 'senkt'} das Risiko."
        flipped = s0.replace("erhöht", "#").replace("senkt", "erhöht").replace("#", "senkt")
        caught["flip_direction"].append(not verify(t.replace(s0, flipped), f).passed)
        unseen = next(x for x in other if x not in f.directions)
        caught["unseen_feature"].append(
            not verify(t + f" {cap(LABELS[unseen])} erhöht das Risiko.", f).passed
        )
        caught["extra_number"].append(not verify(t + " Die Rate beträgt 250 Euro im Monat.", f).passed)
        caught["forbidden_claim"].append(not verify(t + " Ihr Kredit ist damit garantiert.", f).passed)
        garbled = t.replace(s0, s0[:12] + "\u4e4b" + s0[13:]) if len(s0) > 13 else t + " \u4e4b"
        caught["garbled_label"].append(not verify(garbled, f).passed)
        caught["no_direction"].append(
            not verify(t.replace(s0, f"{cap(LABELS[k0])} spielt eine Rolle."), f).passed
        )
    return {k: {"n": len(v), "caught": int(sum(v))} for k, v in caught.items()} | {
        "template_pass_rate": float(np.mean([verify(template(f), f).passed for f in facts]))
    }


def run_variant(name: str, facts: list[Facts]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    with GpuPower() as gpu:
        for i, f in enumerate(facts):
            if name == "template":
                text, tok_in, tok_out, attempts, secs = template(f), 0, 0, 0, 0.0
                v = verify(text, f)
            else:
                msgs = messages(f, few_shot=name != "zero_shot")
                tok_in = tok_out = attempts = 0
                secs = 0.0
                for a in range(1 + (MAX_RETRIES if name == "verified_retry" else 0)):
                    r = chat(MODEL, msgs, seed=SEED + 1000 * a + i)
                    attempts += 1
                    tok_in += r.prompt_tokens
                    tok_out += r.output_tokens
                    secs += r.seconds
                    text = r.text
                    v = verify(text, f)
                    if v.passed or name != "verified_retry":
                        break
                    feedback = (
                        "Der Text verletzt die Regeln: "
                        + "; ".join(v.errors)
                        + ". Schreibe ihn neu und halte dich genau an die Regeln."
                    )
                    msgs = [
                        *msgs,
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": feedback},
                    ]
            fallback = name == "verified_retry" and not v.passed
            rows.append(
                {
                    "variant": name,
                    "i": i,
                    "text": text,
                    "passed": v.passed,
                    "hallucinated": v.hallucinated,
                    "unseen": len(v.unseen) > 0,
                    "wrong_direction": len(v.wrong_direction) > 0,
                    "no_direction": len(v.no_direction) > 0,
                    "numbers": len(v.numbers) > 0,
                    "forbidden": len(v.forbidden) > 0,
                    "sensitive": len(v.sensitive) > 0,
                    "errors": " | ".join(v.errors),
                    "attempts": attempts,
                    "fallback_template": fallback,
                    "shipped_passed": v.passed or fallback,
                    "amstad": amstad(text),
                    "words": len(text.split()),
                    "tokens_in": tok_in,
                    "tokens_out": tok_out,
                    "seconds": secs,
                }
            )
    df = pd.DataFrame(rows)
    summ = {
        "n": len(df),
        "pass_rate": float(df["passed"].mean()),
        "hallucination_rate": float(df["hallucinated"].mean()),
        "unseen_feature_rate": float(df["unseen"].mean()),
        "wrong_direction_rate": float(df["wrong_direction"].mean()),
        "no_direction_rate": float(df["no_direction"].mean()),
        "number_rate": float(df["numbers"].mean()),
        "forbidden_rate": float(df["forbidden"].mean()),
        "sensitive_rate": float(df["sensitive"].mean()),
        "shipped_faithful_rate": float(df["shipped_passed"].mean()),
        "fallback_rate": float(df["fallback_template"].mean()),
        "mean_attempts": float(df["attempts"].mean()),
        "amstad_mean": float(df["amstad"].mean()),
        "words_mean": float(df["words"].mean()),
        "tokens_out_mean": float(df["tokens_out"].mean()),
        "tokens_in_mean": float(df["tokens_in"].mean()),
        "seconds_per_explanation": float(gpu.seconds / len(df)),
        "gpu_mean_watts": gpu.mean_watts,
        "gpu_joules_per_explanation": gpu.joules / len(df),
    }
    return rows, summ


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    facts, meta = build_facts()
    out: dict[str, Any] = {
        "run": {
            "script": "scripts/r5_explain_llm.py",
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": MODEL,
            "model_digest": model_digest(MODEL),
            "temperature": 0.7,
            "top_k_facts": TOP_K,
            "max_retries": MAX_RETRIES,
            "seed": SEED,
        },
        "sample": meta,
        "mutation_check": mutation_check(facts),
        "variants": {},
    }
    all_rows = []
    for name in ("template", "zero_shot", "few_shot", "verified_retry"):
        rows, summ = run_variant(name, facts)
        all_rows += rows
        out["variants"][name] = summ
        print(
            name,
            json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summ.items()}),
            flush=True,
        )
    pd.DataFrame(all_rows).to_csv(RES / "texts.csv", index=False)
    (RES / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
