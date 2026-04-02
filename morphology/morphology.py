"""Task 2: Morphological analysis module."""

from __future__ import annotations

from typing import Dict, List

import spacy
from nltk.stem import PorterStemmer, SnowballStemmer


def manual_morpheme_dictionary() -> Dict[str, Dict[str, str]]:
    """Create manual morpheme analysis for 15 words."""
    return {
        "unhappiness": {"prefix": "un", "root": "happy", "suffix": "ness"},
        "redoing": {"prefix": "re", "root": "do", "suffix": "ing"},
        "careless": {"prefix": "", "root": "care", "suffix": "less"},
        "disliked": {"prefix": "dis", "root": "like", "suffix": "ed"},
        "friendship": {"prefix": "", "root": "friend", "suffix": "ship"},
        "unknown": {"prefix": "un", "root": "know", "suffix": "n"},
        "darkness": {"prefix": "", "root": "dark", "suffix": "ness"},
        "fearful": {"prefix": "", "root": "fear", "suffix": "ful"},
        "kindness": {"prefix": "", "root": "kind", "suffix": "ness"},
        "rewritten": {"prefix": "re", "root": "write", "suffix": "en"},
        "hopeless": {"prefix": "", "root": "hope", "suffix": "less"},
        "misdeeds": {"prefix": "mis", "root": "deed", "suffix": "s"},
        "impatient": {"prefix": "im", "root": "patient", "suffix": ""},
        "singularly": {"prefix": "", "root": "singular", "suffix": "ly"},
        "undemonstrative": {
            "prefix": "un",
            "root": "demonstrate",
            "suffix": "ive",
        },
    }


def rule_based_affix_strip(word: str) -> str:
    """Simple rule-based suffix stripping: ing, ed, es, s."""
    word_l = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if word_l.endswith(suffix) and len(word_l) > len(suffix) + 2:
            return word_l[: -len(suffix)]
    return word_l


def spacy_morphological_features(
    nlp: spacy.language.Language, words: List[str]
) -> Dict[str, Dict[str, str]]:
    """Extract POS, tense, number, and lemma features with spaCy."""
    doc = nlp(" ".join(words))
    features: Dict[str, Dict[str, str]] = {}

    for token in doc:
        morph_tense = token.morph.get("Tense")
        morph_number = token.morph.get("Number")
        features[token.text.lower()] = {
            "pos": token.pos_ if token.pos_ else "N/A",
            "tense": morph_tense[0] if morph_tense else "N/A",
            "number": morph_number[0] if morph_number else "N/A",
            "lemma": token.lemma_ if token.lemma_ else token.text.lower(),
        }
    return features


def run_morphology_analysis(nlp: spacy.language.Language) -> None:
    """Run morphological analysis and print comparison tables."""
    manual_map = manual_morpheme_dictionary()
    words = list(manual_map.keys())

    print("Manual morpheme decomposition (15 words)")
    print("-" * 70)
    print(f"{'Word':<16}{'Prefix':<12}{'Root':<20}{'Suffix':<12}")
    print("-" * 70)
    for word, parts in manual_map.items():
        print(f"{word:<16}{parts['prefix']:<12}{parts['root']:<20}{parts['suffix']:<12}")

    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    spacy_feats = spacy_morphological_features(nlp, words)

    print("\nComparison table")
    print("-" * 98)
    print(f"{'Word':<16}{'Rule-based':<16}{'Porter':<14}{'Snowball':<14}{'Lemma (spaCy)':<18}")
    print("-" * 98)

    for word in words:
        rb = rule_based_affix_strip(word)
        p = porter.stem(word)
        s = snowball.stem(word)
        lemma = spacy_feats.get(word, {}).get("lemma", word)
        print(f"{word:<16}{rb:<16}{p:<14}{s:<14}{lemma:<18}")

    print("\nspaCy morphological features")
    print("-" * 70)
    print(f"{'Word':<16}{'POS':<10}{'Tense':<12}{'Number':<12}")
    print("-" * 70)
    for word in words:
        feats = spacy_feats.get(word, {})
        print(f"{word:<16}{feats.get('pos', 'N/A'):<10}{feats.get('tense', 'N/A'):<12}{feats.get('number', 'N/A'):<12}")

    print("\nDiscussion")
    print("-" * 70)
    print(
        "Rule-based stripping is transparent and fast but over-simplifies morphology "
        "and can produce non-words."
    )
    print(
        "Porter and Snowball are more robust for stemming, but stems may still be "
        "linguistically coarse."
    )
    print(
        "spaCy provides context-aware lemmatization and grammatical features (POS, tense, number), "
        "which are generally better for linguistic interpretation."
    )
