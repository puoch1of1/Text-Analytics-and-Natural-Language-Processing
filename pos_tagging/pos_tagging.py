"""Task 6: POS tagging and evaluation module."""

from __future__ import annotations

from typing import List, Tuple

import nltk
from nltk.corpus import treebank
from nltk.tag import UnigramTagger
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import save_text


def evaluate_pos_tagging(corpus_text: str) -> tuple[float, float, List[Tuple[str, str, str]], str]:
    """Evaluate default NLTK POS tagging and a unigram tagger.

    Returns default accuracy, unigram accuracy, sample errors, and printable preview text.
    """
    sentences = sent_tokenize(corpus_text)
    corpus_word_sents = [word_tokenize(s) for s in sentences[:20]]
    corpus_tagged_preview = [nltk.pos_tag(ws) for ws in corpus_word_sents[:3] if ws]

    tagged_sents = treebank.tagged_sents()
    split_idx = int(0.8 * len(tagged_sents))
    train_data = tagged_sents[:split_idx]
    test_data = tagged_sents[split_idx:]

    unigram_tagger = UnigramTagger(train_data)

    default_correct = 0
    default_total = 0
    unigram_correct = 0
    unigram_total = 0
    errors: List[Tuple[str, str, str]] = []

    for sent in test_data:
        words = [w for w, _ in sent]
        gold_tags = [t for _, t in sent]

        default_pred = [tag for _, tag in nltk.pos_tag(words)]
        unigram_pred_pairs = unigram_tagger.tag(words)
        unigram_pred = [tag if tag is not None else "NN" for _, tag in unigram_pred_pairs]

        for w, gold, d_pred, u_pred in zip(words, gold_tags, default_pred, unigram_pred):
            default_total += 1
            unigram_total += 1
            if d_pred == gold:
                default_correct += 1
            if u_pred == gold:
                unigram_correct += 1

            if u_pred != gold and len(errors) < 10:
                errors.append((w, u_pred, gold))

    default_acc = default_correct / default_total if default_total else 0.0
    unigram_acc = unigram_correct / unigram_total if unigram_total else 0.0

    preview_lines = ["NLTK default POS tagging preview on corpus (first 3 sentences)", "-" * 90]
    for i, tagged in enumerate(corpus_tagged_preview, start=1):
        preview_lines.append(f"Sentence {i}: {tagged}")

    return default_acc, unigram_acc, errors, "\n".join(preview_lines)


def format_pos_report(
    preview_text: str,
    default_acc: float,
    unigram_acc: float,
    errors: List[Tuple[str, str, str]],
) -> str:
    """Format POS evaluation output as a clean report string."""
    lines: List[str] = [preview_text]

    lines.append("\nAccuracy comparison")
    lines.append("-" * 90)
    lines.append(f"Default NLTK tagger accuracy: {default_acc:.4f}")
    lines.append(f"Unigram tagger accuracy:      {unigram_acc:.4f}")

    lines.append("\nAt least 10 unigram tagging errors (predicted vs gold)")
    lines.append("-" * 90)
    lines.append(f"{'Word':<20}{'Predicted':<14}{'Gold':<14}")
    lines.append("-" * 90)
    for word, pred, gold in errors:
        lines.append(f"{word:<20}{pred:<14}{gold:<14}")

    lines.append("\nLinguistic explanation")
    lines.append("-" * 90)
    lines.append(
        "Tagging errors often come from lexical ambiguity (e.g., words functioning as noun vs verb in different contexts)."
    )
    lines.append(
        "Unknown or rare words are harder for unigram models because they rely mostly on single-word statistics from training data."
    )
    lines.append(
        "A unigram tagger ignores sentence context, so it cannot effectively resolve context-dependent POS decisions."
    )

    return "\n".join(lines)


def save_pos_report(report_text: str, output_path) -> None:
    """Persist POS evaluation report to disk."""
    save_text(output_path, report_text)
