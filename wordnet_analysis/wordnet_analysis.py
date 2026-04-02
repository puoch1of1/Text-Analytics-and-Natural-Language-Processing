"""Task 3: WordNet lexical analysis module."""

from __future__ import annotations

from typing import List

from nltk.corpus.reader.wordnet import Synset
from nltk.corpus import wordnet


def get_hypernym_names(synset: Synset) -> List[str]:
    """Return readable hypernym names from a synset."""
    return [h.name().split(".")[0] for h in synset.hypernyms()]


def run_wordnet_analysis(top_words: List[str]) -> None:
    """Run synset, hypernym, synonym, and query expansion analysis."""
    selected = top_words[:5]
    print(f"Selected 5 frequent words: {', '.join(selected)}")

    for word in selected:
        print("\n" + "-" * 90)
        print(f"Word: {word}")
        synsets = wordnet.synsets(word)
        if not synsets:
            print("No WordNet synsets found.")
            continue

        for idx, synset in enumerate(synsets[:3], start=1):
            synonyms = sorted({lemma.name().replace("_", " ") for lemma in synset.lemmas()})
            hypernyms = get_hypernym_names(synset)
            print(f"Synset {idx}: {synset.name()}")
            print(f"  Definition: {synset.definition()}")
            print(f"  Example synonyms: {', '.join(synonyms[:6]) if synonyms else 'N/A'}")
            print(f"  Hypernyms: {', '.join(hypernyms[:6]) if hypernyms else 'N/A'}")

    query_word = selected[0] if selected else "good"
    syn_expansion = sorted(
        {
            lemma.name().replace("_", " ")
            for synset in wordnet.synsets(query_word)
            for lemma in synset.lemmas()
        }
    )
    syn_expansion = [w for w in syn_expansion if w != query_word][:8]

    docs = [
        "The doctor showed remarkable kindness to his friend.",
        "His behavior revealed deep goodness and charity.",
        "The legal case transformed public opinion.",
    ]

    base_matches = [doc for doc in docs if query_word.lower() in doc.lower()]
    expanded_terms = [query_word] + syn_expansion
    expanded_matches = [
        doc for doc in docs if any(term.lower() in doc.lower() for term in expanded_terms)
    ]

    print("\nSynonym expansion demonstration")
    print("-" * 90)
    print(f"Query word: {query_word}")
    print(f"Expanded terms: {', '.join(expanded_terms[:10])}")
    print(f"Matches without expansion: {len(base_matches)}")
    for doc in base_matches:
        print(f"  - {doc}")
    print(f"Matches with expansion:    {len(expanded_matches)}")
    for doc in expanded_matches:
        print(f"  - {doc}")
