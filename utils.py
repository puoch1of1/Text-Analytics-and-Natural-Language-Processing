"""Shared utility helpers for the modular NLP pipeline."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Sequence

import nltk
import numpy as np
import spacy

from config import DEFAULT_CORPUS_PATH, NLTK_RESOURCES, SPACY_MODEL_NAME


def print_header(title: str) -> None:
    """Print a consistent section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def setup_reproducibility(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dir(path: Path) -> None:
    """Create output directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def download_nltk_resources() -> None:
    """Download required NLTK resources if they are missing."""
    for resource in NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


def load_spacy_model() -> spacy.language.Language:
    """Load spaCy English model, downloading it if needed.

    Falls back to a lightweight blank English pipeline if model download fails.
    """
    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError:
        try:
            from spacy.cli import download

            download(SPACY_MODEL_NAME)
            return spacy.load(SPACY_MODEL_NAME)
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp


def get_corpus_text(provided_corpus_text: str | None = None) -> str:
    """Get corpus text from argument or default file path."""
    if isinstance(provided_corpus_text, str):
        return provided_corpus_text

    if not DEFAULT_CORPUS_PATH.exists():
        raise FileNotFoundError(
            "corpus_text was not provided, and default corpus file was not found."
        )
    return DEFAULT_CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")


def compute_ttr(tokens: Sequence[str]) -> float:
    """Compute type-token ratio for a token sequence."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def save_csv_rows(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Save rows to CSV file with a header."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_text(path: Path, text: str) -> None:
    """Save plain text content to file."""
    path.write_text(text, encoding="utf-8")


def save_top20_plot(words: Sequence[str], counts: Sequence[int], output_path: Path) -> str:
    """Plot and save top-word frequency chart.

    Returns a status message so callers can print outcomes cleanly.
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.bar(words, counts)
        plt.title("Top 20 Word Frequencies (Cleaned Corpus)")
        plt.xlabel("Word")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return f"Bonus plot saved: {output_path}"
    except Exception as exc:
        return f"Matplotlib plot skipped: {exc}"
