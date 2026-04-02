"""Task 1: Text preprocessing module."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from config import METRICS_TXT, STOPWORDS_LANGUAGE, TOP_WORDS_CSV
from utils import compute_ttr, save_csv_rows, save_text


@dataclass
class PreprocessingResult:
    """Container for preprocessing outputs and summary metrics."""

    raw_sentences: List[str]
    tokenized_sentences: List[List[str]]
    cleaned_sentences: List[List[str]]
    raw_tokens: List[str]
    cleaned_tokens: List[str]
    vocab_before: int
    vocab_after: int
    ttr_before: float
    ttr_after: float
    top_20: List[Tuple[str, int]]


def preprocess_text(corpus_text: str) -> PreprocessingResult:
    """Run sentence/word tokenization and normalization pipeline."""
    sentences = sent_tokenize(corpus_text)
    tokenized = [word_tokenize(sentence) for sentence in sentences]
    raw_tokens = [token for sentence in tokenized for token in sentence]

    english_stopwords = set(stopwords.words(STOPWORDS_LANGUAGE))
    cleaned_sentences: List[List[str]] = []

    for sentence_tokens in tokenized:
        cleaned_sentence = []
        for token in sentence_tokens:
            token_lower = token.lower()
            token_alpha_num = re.sub(r"[^a-z0-9]", "", token_lower)
            if token_alpha_num and token_alpha_num not in english_stopwords:
                cleaned_sentence.append(token_alpha_num)
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)

    cleaned_tokens = [token for sent in cleaned_sentences for token in sent]
    top_20 = Counter(cleaned_tokens).most_common(20)

    return PreprocessingResult(
        raw_sentences=sentences,
        tokenized_sentences=tokenized,
        cleaned_sentences=cleaned_sentences,
        raw_tokens=raw_tokens,
        cleaned_tokens=cleaned_tokens,
        vocab_before=len(set(raw_tokens)),
        vocab_after=len(set(cleaned_tokens)),
        ttr_before=compute_ttr(raw_tokens),
        ttr_after=compute_ttr(cleaned_tokens),
        top_20=top_20,
    )


def print_preprocessing_results(result: PreprocessingResult) -> None:
    """Print preprocessing metrics in a clean labeled format."""
    print(f"Total sentences: {len(result.raw_sentences)}")
    print(f"Total tokens before cleaning: {len(result.raw_tokens)}")
    print(f"Total tokens after cleaning:  {len(result.cleaned_tokens)}")
    print(f"Vocabulary size before cleaning: {result.vocab_before}")
    print(f"Vocabulary size after cleaning:  {result.vocab_after}")
    print(f"TTR before cleaning: {result.ttr_before:.4f}")
    print(f"TTR after cleaning:  {result.ttr_after:.4f}")

    print("\nTop 20 most frequent words (cleaned corpus)")
    print("-" * 50)
    print(f"{'Rank':<6}{'Word':<20}{'Count':>8}")
    print("-" * 50)
    for idx, (word, count) in enumerate(result.top_20, start=1):
        print(f"{idx:<6}{word:<20}{count:>8}")

    vocab_change = result.vocab_before - result.vocab_after
    token_change = len(result.raw_tokens) - len(result.cleaned_tokens)
    print("\nAnalysis")
    print("-" * 50)
    print(
        "Preprocessing reduced noise by removing punctuation and stopwords, "
        "which lowered both token and vocabulary counts."
    )
    print(
        f"Vocabulary reduced by {vocab_change} terms, and token count reduced by "
        f"{token_change} tokens."
    )
    print(
        "The cleaned frequency list is more semantically meaningful for downstream "
        "tasks such as lexical analysis, language modeling, and generation."
    )


def save_preprocessing_outputs(result: PreprocessingResult, output_dir: Path) -> None:
    """Save preprocessing metrics and frequency table to output files."""
    metrics_text = (
        "TASK 1 PREPROCESSING METRICS\n"
        + "=" * 50
        + "\n"
        + f"Vocabulary before: {result.vocab_before}\n"
        + f"Vocabulary after:  {result.vocab_after}\n"
        + f"TTR before: {result.ttr_before:.6f}\n"
        + f"TTR after:  {result.ttr_after:.6f}\n"
    )
    save_text(output_dir / METRICS_TXT.name, metrics_text)

    rows = [[idx, word, count] for idx, (word, count) in enumerate(result.top_20, start=1)]
    save_csv_rows(output_dir / TOP_WORDS_CSV.name, ["rank", "word", "count"], rows)


def run_preprocessing(corpus_text: str, output_dir: Path) -> PreprocessingResult:
    """Execute preprocessing task and persist key outputs."""
    result = preprocess_text(corpus_text)
    print_preprocessing_results(result)
    save_preprocessing_outputs(result, output_dir)
    return result
