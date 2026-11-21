"""Module D: the deterministic verifier must accept faithful texts and reject every error class."""

from __future__ import annotations

import pytest

from risikopegel.llm.labels import LABELS, patterns
from risikopegel.llm.verifier import Facts, amstad, find_features, verify

F = Facts({"ext_source_2": -1, "credit_to_annuity": 1, "annuity_to_income": 1, "days_employed": -1}, 9)
GOOD = (
    "Für Ihren Antrag schätzen wir eine Ausfallwahrscheinlichkeit von 9 %. Der Score der Auskunftei B senkt das "
    "Risiko. Die Laufzeit des Kredits erhöht das Risiko. Das Verhältnis von Rate zu Einkommen erhöht das Risiko. "
    "Die Dauer Ihrer aktuellen Beschäftigung wirkt sich positiv aus."
)


def test_labels_cover_all_features_and_are_unique() -> None:
    assert len(set(LABELS.values())) == len(LABELS)
    for f in LABELS:
        assert f in find_features(LABELS[f]), f


def test_longest_match_masks_shorter_labels() -> None:
    # "Einkommen" alone is income, but inside the ratio label it must not count as income
    assert find_features("Das Verhältnis von Rate zu Einkommen") == ["annuity_to_income"]
    assert sorted(find_features("Ihr Einkommen und die Laufzeit")) == ["credit_to_annuity", "income"]


def test_good_text_passes() -> None:
    v = verify(GOOD, F)
    assert v.passed, v.errors


@pytest.mark.parametrize(
    ("text", "attr"),
    [
        (GOOD.replace("Auskunftei B senkt", "Auskunftei B erhöht"), "wrong_direction"),
        (GOOD + " Ihr Einkommen erhöht das Risiko.", "unseen"),
        (GOOD + " Ihre Rate beträgt 312 Euro.", "numbers"),
        (GOOD + " Der Kredit ist garantiert.", "forbidden"),
        (
            GOOD.replace(
                "Die Laufzeit des Kredits erhöht das Risiko.", "Die Laufzeit des Kredits ist relevant."
            ),
            "no_direction",
        ),
        (GOOD + " Ihr Alter senkt das Risiko.", "sensitive"),
    ],
)
def test_each_error_class_is_caught(text: str, attr: str) -> None:
    v = verify(text, F)
    assert not v.passed
    assert getattr(v, attr), (attr, v.errors)


def test_mixed_directions_in_one_sentence_are_unverifiable() -> None:
    t = GOOD + " Der Score der Auskunftei B und die Laufzeit des Kredits erhöhen das Risiko."
    assert not verify(t, F).passed


def test_ambiguous_generic_term_fails() -> None:
    assert not verify(GOOD + " Die Scores der Auskunfteien senken das Risiko.", F).passed


def test_too_few_factors_fails() -> None:
    assert not verify("Der Score der Auskunftei B senkt das Risiko.", F).passed


def test_patterns_strip_articles() -> None:
    assert "score der auskunftei a" in patterns("ext_source_1")


def test_amstad_orders_texts_by_difficulty() -> None:
    easy = "Das ist gut. Wir sind da. Er hat Zeit."
    hard = "Die außerordentliche Kreditwürdigkeitsprüfung berücksichtigt unterschiedlichste Finanzierungsmerkmale."
    assert amstad(easy) > amstad(hard)


def test_garbled_labels_and_foreign_characters_fail() -> None:
    assert not verify(GOOD.replace("Auskunftei B", "Auskunfte\u4e4b B"), F).passed
    assert not verify(GOOD.replace("Auskunftei B", "Auskunfte und B"), F).passed
    assert verify(GOOD, F).passed
