"""Task 5: Text generation module using a bigram model."""

from __future__ import annotations

from typing import List

import numpy as np

from ngram_model.ngram import NGramModel, conditional_probability


def sample_next_word(model: NGramModel, current_word: str, laplace: bool = False) -> str:
    """Sample next word from bigram distribution."""
    if not laplace:
        candidates = model.next_word_map.get(current_word)
        if not candidates:
            return "</s>"
        words = list(candidates.keys())
        probs = np.array(list(candidates.values()), dtype=np.float64)
        probs = probs / probs.sum()
        return str(np.random.choice(words, p=probs))

    words = model.vocab
    probs = np.array(
        [conditional_probability(model, current_word, w, laplace=True) for w in words],
        dtype=np.float64,
    )
    probs = probs / probs.sum()
    return str(np.random.choice(words, p=probs))


def generate_sentence(model: NGramModel, max_len: int = 20, laplace: bool = False) -> str:
    """Generate one sentence from the bigram model."""
    words: List[str] = []
    current = "<s>"

    for _ in range(max_len):
        nxt = sample_next_word(model, current, laplace=laplace)
        if nxt == "</s>":
            break
        if nxt == "<s>":
            continue
        words.append(nxt)
        current = nxt

    if not words:
        return "[empty generation]"
    return " ".join(words)


def generate_sentences(
    model: NGramModel,
    num_sentences: int,
    max_len: int,
    laplace: bool,
) -> List[str]:
    """Generate a list of random sentences from the model."""
    return [generate_sentence(model, max_len=max_len, laplace=laplace) for _ in range(num_sentences)]


def format_generation_report(raw_sentences: List[str], smooth_sentences: List[str]) -> str:
    """Format generated outputs and discussion into a text report."""
    lines: List[str] = []
    lines.append("Generated sentences WITHOUT smoothing")
    lines.append("-" * 90)
    for i, sentence in enumerate(raw_sentences, start=1):
        lines.append(f"{i}. {sentence}")

    lines.append("\nGenerated sentences WITH Laplace smoothing")
    lines.append("-" * 90)
    for i, sentence in enumerate(smooth_sentences, start=1):
        lines.append(f"{i}. {sentence}")

    lines.append("\nAnalysis")
    lines.append("-" * 90)
    lines.append(
        "Without smoothing, generated text is usually more locally coherent but limited to observed transitions."
    )
    lines.append(
        "With Laplace smoothing, text is more diverse but often less coherent because it allows many low-probability transitions."
    )
    lines.append(
        "Bigram models capture only short-range context, so they struggle with global syntax, semantics, and long-distance dependencies."
    )

    return "\n".join(lines)
