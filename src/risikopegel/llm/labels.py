"""German customer-facing labels for every model feature, plus aliases the verifier recognises.

Each label is distinct enough to be matched by substring (lower case). Generic words that could refer to
several features ("Auskunftei", "Kredit") are listed as AMBIGUOUS: a text that uses them without the
specific label cannot be verified and fails.
"""

from __future__ import annotations

LABELS: dict[str, str] = {
    "age_years": "Ihr Alter",
    "annuity": "die Höhe der monatlichen Rate",
    "annuity_to_income": "das Verhältnis von Rate zu Einkommen",
    "bureau_days_since_first_open": "die Dauer Ihrer Kredithistorie bei anderen Banken",
    "bureau_days_since_last_open": "die Zeit seit Ihrem letzten Kredit bei anderen Banken",
    "bureau_debt_to_credit": "der Anteil noch offener Schulden bei anderen Banken",
    "bureau_history_coverage": "der Umfang der monatlichen Kontohistorie bei anderen Banken",
    "bureau_history_status": "das Vorliegen einer Kredithistorie bei anderen Banken",
    "bureau_max_amt_overdue": "der höchste überfällige Betrag bei anderen Banken",
    "bureau_max_days_overdue": "die längste Zahlungsverspätung bei anderen Banken",
    "bureau_n_active": "die Zahl laufender Kredite bei anderen Banken",
    "bureau_n_loans": "die Zahl Ihrer Kredite bei anderen Banken",
    "bureau_n_prolong": "die Zahl verlängerter Kredite bei anderen Banken",
    "bureau_total_credit": "die Summe Ihrer Kredite bei anderen Banken",
    "bureau_total_debt": "die Summe Ihrer offenen Schulden bei anderen Banken",
    "bureau_total_overdue": "die Summe überfälliger Beträge bei anderen Banken",
    "bureau_worst_dpd_bucket": "der schwerste Zahlungsrückstand bei anderen Banken",
    "bureau_worst_dpd_bucket_12m": "der schwerste Zahlungsrückstand der letzten zwölf Monate bei anderen Banken",
    "cc_max_dpd": "der längste Zahlungsrückstand auf Ihrer Kreditkarte",
    "cc_mean_utilisation": "die durchschnittliche Auslastung Ihrer Kreditkarte",
    "cc_mean_utilisation_6m": "die Auslastung Ihrer Kreditkarte im letzten halben Jahr",
    "contract_type": "die Art des Kreditvertrags",
    "credit_amount": "der beantragte Kreditbetrag",
    "credit_to_annuity": "die Laufzeit des Kredits",
    "credit_to_income": "das Verhältnis von Kreditbetrag zu Einkommen",
    "days_birth": "Ihr Geburtsdatum",
    "days_employed": "die Dauer Ihrer aktuellen Beschäftigung",
    "days_id_publish": "das Ausstellungsdatum Ihres Ausweises",
    "days_last_phone_change": "der letzte Wechsel Ihrer Telefonnummer",
    "days_registration": "die Dauer Ihrer aktuellen Meldeadresse",
    "education": "Ihr Bildungsabschluss",
    "employed_placeholder": "eine fehlende Angabe zur Beschäftigungsdauer",
    "ext_source_1": "der Score der Auskunftei A",
    "ext_source_2": "der Score der Auskunftei B",
    "ext_source_3": "der Score der Auskunftei C",
    "family_status": "Ihr Familienstand",
    "goods_price": "der Preis der finanzierten Ware",
    "housing_type": "Ihre Wohnsituation",
    "income": "Ihr Einkommen",
    "income_type": "die Art Ihres Einkommens",
    "inst_mean_days_late": "die durchschnittliche Verspätung Ihrer früheren Ratenzahlungen",
    "inst_n": "die Zahl Ihrer früheren Ratenzahlungen bei uns",
    "inst_share_late": "der Anteil verspäteter früherer Raten",
    "inst_share_underpaid": "der Anteil zu niedrig gezahlter früherer Raten",
    "inst_share_unpaid": "der Anteil nicht gezahlter früherer Raten",
    "n_bureau_requests_year": "die Zahl der Kreditanfragen im letzten Jahr",
    "n_children": "die Zahl Ihrer Kinder",
    "n_family_members": "die Größe Ihres Haushalts",
    "occupation": "Ihr Beruf",
    "own_car_age": "das Alter Ihres Autos",
    "owns_car": "der Besitz eines Autos",
    "owns_realty": "der Besitz von Wohneigentum",
    "pos_max_dpd": "der längste Rückstand bei früheren Ratenkäufen",
    "pos_mean_dpd_def": "die durchschnittliche Verspätung bei früheren Ratenkäufen",
    "prev_days_since_last_decision": "die Zeit seit Ihrem letzten Kreditantrag bei uns",
    "prev_mean_credit_to_application": "das Verhältnis von bewilligtem zu beantragtem Betrag bei früheren Anträgen",
    "prev_n_applications": "die Zahl Ihrer früheren Kreditanträge bei uns",
    "prev_n_approved": "die Zahl Ihrer früher bewilligten Anträge",
    "prev_n_refused": "die Zahl Ihrer früher abgelehnten Anträge",
    "region_population_rel": "die Bevölkerungsdichte Ihrer Wohnregion",
    "region_rating": "die Einstufung Ihrer Wohnregion",
}

# Extra spellings that unambiguously name one feature (all lower case).
ALIASES: dict[str, list[str]] = {
    "ext_source_1": ["auskunftei a"],
    "ext_source_2": ["auskunftei b"],
    "ext_source_3": ["auskunftei c"],
    "credit_to_annuity": ["laufzeit"],
    "credit_amount": ["kreditbetrag", "kreditsumme"],
    "annuity": ["monatliche rate", "monatlichen rate", "ratenhöhe"],
    "income": ["einkommen"],
    "days_employed": [
        "beschäftigungsdauer",
        "dauer ihrer beschäftigung",
        "dauer ihrer aktuellen beschäftigung",
    ],
    "goods_price": ["preis der finanzierten ware", "warenpreis", "kaufpreis"],
    "occupation": ["ihr beruf", "ihres berufs", "ihrem beruf"],
    "education": ["bildungsabschluss", "schulabschluss", "ausbildung"],
    "age_years": ["ihr alter", "ihres alters", "lebensalter"],
    "family_status": ["familienstand"],
    "region_rating": ["einstufung ihrer wohnregion", "wohnregion"],
    "contract_type": ["kreditvertrag", "vertragsart", "art des kredit"],
    "inst_n": ["früheren ratenzahlungen bei uns", "zahl ihrer früheren ratenzahlungen"],
    "days_last_phone_change": ["telefonnummer"],
    "days_id_publish": ["ausweis"],
    "days_registration": ["meldeadresse"],
    "owns_car": ["besitz eines autos"],
    "own_car_age": ["alter ihres autos"],
    "owns_realty": ["wohneigentum"],
    "housing_type": ["wohnsituation"],
    "n_children": ["kinder"],
    "n_family_members": ["haushalt"],
}

AMBIGUOUS = ["auskunfteien", "bonität", "schufa", "scores der", "externe bewertung", "externen bewertungen"]

# Attributes a customer letter must not cite even if the model uses them (protected-like; see the model
# card's fairness section). The verifier flags them; they are reported, not silently dropped.
SENSITIVE = {"age_years", "days_birth", "family_status", "n_children", "n_family_members"}


def patterns(feature: str) -> list[str]:
    lab = LABELS[feature].lower()
    core = (
        lab.split(" ", 1)[1] if lab.split(" ", 1)[0] in {"der", "die", "das", "ihr", "ihre", "eine"} else lab
    )
    return sorted({core, *ALIASES.get(feature, [])}, key=len, reverse=True)
