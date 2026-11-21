"""Deterministic verifier for German credit-decision explanations.

Given the facts the text may use (top SHAP features with their direction, the rounded PD), a text passes
only if:
1. every feature it names is among the facts (no unseen feature), and no ambiguous generic term is used;
2. every sentence that names a feature states a direction, and the direction matches the SHAP sign
   (risk-increasing vs risk-decreasing); a sentence naming features of opposite directions is rejected as
   unverifiable;
3. it contains no number other than the allowed PD percentage;
4. it contains none of the forbidden claims (decisions, guarantees, legal statements, protected attributes);
5. it names at least `min_features` of the facts;
6. it contains only Latin/German characters, and no factor-like word ("Auskunft", "Score") is left over once
   all recognised labels are removed (catches garbled labels such as "Auskunfte\u4e4b C"; added after the
   smoke test on 2026-11-21 showed such texts passing).
Matching is plain lower-case substring search, longest pattern first, with matched spans masked so that a
long label ("Verhältnis von Rate zu Einkommen") is not also counted as a shorter one ("Einkommen").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from risikopegel.llm.labels import AMBIGUOUS, LABELS, SENSITIVE, patterns

UP = [
    "erhöht",
    "erhöhen",
    "erhöhend",
    "steigert",
    "steigern",
    "verschlechtert",
    "verschlechtern",
    "höheres risiko",
    "höheren risiko",
    "wirkt sich negativ",
    "wirken sich negativ",
    "spricht gegen",
    "sprechen gegen",
    "belastet",
    "belasten",
    "risikoerhöhend",
    "ungünstig",
]
DOWN = [
    "senkt",
    "senken",
    "verringert",
    "verringern",
    "reduziert",
    "reduzieren",
    "verbessert",
    "verbessern",
    "niedrigeres risiko",
    "niedrigeren risiko",
    "geringeres risiko",
    "wirkt sich positiv",
    "wirken sich positiv",
    "spricht für",
    "sprechen für",
    "entlastet",
    "entlasten",
    "risikosenkend",
    "günstig",
]
FORBIDDEN = [
    r"\bgarantiert",
    r"\bgenehmigt\b",
    r"\babgelehnt\b",
    r"\bzugesagt",
    r"\bbewilligt\b",
    r"rechtsanspruch",
    r"\bdsgvo\b",
    r"gesetzlich",
    r"\bgeschlecht",
    r"\bherkunft",
    r"\breligion",
    r"nationalität",
    r"\bschufa\b",
    r"\bsicher(?:lich)?\b",
    r"\bimmer\b",
    r"\bnie(?:mals)?\b",
]
RESIDUAL_STEMS = ["auskunft", "score"]
_ALLOWED_CHARS = re.compile("[A-Za-zÄÖÜäöüß0-9\\s.,;:!?%()/'\"\\-\u2013\u201e\u201c]")
_PATTERNS = sorted(((p, f) for f in LABELS for p in patterns(f)), key=lambda x: len(x[0]), reverse=True)


@dataclass
class Facts:
    directions: dict[str, int]  # feature -> +1 (raises risk) / -1 (lowers risk)
    pd_percent: int


@dataclass
class Verdict:
    passed: bool
    errors: list[str] = field(default_factory=list)
    named: list[str] = field(default_factory=list)
    unseen: list[str] = field(default_factory=list)
    wrong_direction: list[str] = field(default_factory=list)
    no_direction: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    sensitive: list[str] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return bool(self.unseen or self.numbers)


def _scan(text: str) -> tuple[list[str], str]:
    """Recognised features and the lower-case text with every recognised label masked out."""
    t = text.lower()
    found: list[str] = []
    for p, f in _PATTERNS:
        i = t.find(p)
        while i >= 0:
            found.append(f)
            t = t[:i] + "\0" * len(p) + t[i + len(p) :]
            i = t.find(p)
    return found, t


def find_features(text: str) -> list[str]:
    return _scan(text)[0]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;:])\s+|\n+", text) if s.strip()]


def _direction(sentence: str) -> int:
    s = sentence.lower()
    # "ungünstig" contains "günstig": count the longer cue first by masking
    up = sum(s.count(c) for c in UP)
    for c in UP:
        s = s.replace(c, " ")
    down = sum(s.count(c) for c in DOWN)
    if up and not down:
        return 1
    if down and not up:
        return -1
    return 0 if not up and not down else 2  # 2 = both


def verify(text: str, facts: Facts, min_features: int = 2) -> Verdict:  # noqa: PLR0912 (one branch per check)
    v = Verdict(passed=False)
    named, residual = _scan(text)
    bad_chars = sorted({c for c in text if not _ALLOWED_CHARS.match(c)})
    leftovers = [w for w in RESIDUAL_STEMS if w in residual]
    v.named = sorted(set(named))
    v.unseen = sorted({f for f in named if f not in facts.directions})
    v.sensitive = sorted({f for f in named if f in SENSITIVE})
    low = text.lower()
    amb = [a for a in AMBIGUOUS if a in low]
    for sent in _sentences(text):
        feats = sorted({f for f in find_features(sent) if f in facts.directions})
        if not feats:
            continue
        signs = {facts.directions[f] for f in feats}
        d = _direction(sent)
        if len(signs) > 1 or d == 2:
            v.wrong_direction += [f for f in feats if d != facts.directions[f]]
            if d == 2:
                v.no_direction += feats
            continue
        if d == 0:
            v.no_direction += feats
        elif d != signs.pop():
            v.wrong_direction += feats
    allowed = {str(facts.pd_percent)}
    v.numbers = [n for n in re.findall(r"\d+(?:[.,]\d+)?", text) if n not in allowed]
    v.forbidden = [p for p in FORBIDDEN if re.search(p, low)]
    if amb:
        v.errors.append(f"ambiguous terms: {amb}")
    if bad_chars:
        v.errors.append(f"non-German characters: {bad_chars}")
    if leftovers:
        v.errors.append(f"unrecognised factor wording (garbled label?): {leftovers}")
    if v.unseen:
        v.errors.append(f"features not in the explanation facts: {v.unseen}")
    if v.wrong_direction:
        v.errors.append(f"wrong or contradictory direction for: {sorted(set(v.wrong_direction))}")
    if v.no_direction:
        v.errors.append(f"no clear direction stated for: {sorted(set(v.no_direction))}")
    if v.numbers:
        v.errors.append(f"numbers not in the facts: {v.numbers}")
    if v.forbidden:
        v.errors.append(f"forbidden wording: {v.forbidden}")
    if v.sensitive:
        v.errors.append(f"protected-like attribute cited: {v.sensitive}")
    if len(set(named) & set(facts.directions)) < min_features:
        v.errors.append(f"names fewer than {min_features} of the given factors")
    v.passed = not v.errors
    return v


def syllables(word: str) -> int:
    """German syllable estimate: groups of vowels (incl. umlauts and common diphthongs count once)."""
    return max(1, len(re.findall(r"[aeiouyäöü]+", word.lower())))


def amstad(text: str) -> float:
    """Flesch reading ease, German adaptation (Amstad 1978): 180 - ASL - 58.5 * ASW."""
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", text)
    if not sents or not words:
        return float("nan")
    return 180 - len(words) / len(sents) - 58.5 * sum(syllables(w) for w in words) / len(words)
