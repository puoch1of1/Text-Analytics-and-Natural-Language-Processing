"""Task 4: N-gram language model module."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NGramModel:
    """Simple bigram language model with optional Laplace smoothing."""

    unigram_counts: Counter
    bigram_counts: Counter
    vocab: List[str]
    vocab_size: int
    next_word_map: Dict[str, Counter]


def build_ngram_model(cleaned_sentences: List[List[str]]) -> NGramModel:
    """Build unigram and bigram counts from cleaned tokenized sentences."""
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    next_word_map: Dict[str, Counter] = defaultdict(Counter)

    for sentence in cleaned_sentences:
        seq = ["<s>"] + sentence + ["</s>"]
        unigram_counts.update(seq)
        bigrams = list(zip(seq[:-1], seq[1:]))
        bigram_counts.update(bigrams)
        for w1, w2 in bigrams:
            next_word_map[w1][w2] += 1

    vocab = sorted(unigram_counts.keys())
    return NGramModel(
        unigram_counts=unigram_counts,
        bigram_counts=bigram_counts,
        vocab=vocab,
        vocab_size=len(vocab),
        next_word_map=next_word_map,
    )


def conditional_probability(model: NGramModel, w1: str, w2: str, laplace: bool = False) -> float:
    """Compute conditional probability P(w2|w1), with optional Laplace smoothing."""
    bigram_count = model.bigram_counts[(w1, w2)]
    unigram_count = model.unigram_counts[w1]

    if laplace:
        return (bigram_count + 1) / (unigram_count + model.vocab_size)

    if unigram_count == 0:
        return 0.0
    return bigram_count / unigram_count


def sentence_probability(model: NGramModel, sentence_tokens: List[str], laplace: bool = True) -> float:
    """Compute sentence probability using bigram chain rule."""
    seq = ["<s>"] + sentence_tokens + ["</s>"]
    prob = 1.0
    for w1, w2 in zip(seq[:-1], seq[1:]):
        prob *= conditional_probability(model, w1, w2, laplace=laplace)
    return prob


def sentence_log_probability(model: NGramModel, sentence_tokens: List[str], laplace: bool = True) -> float:
    """Compute log probability of a sentence to avoid underflow."""
    seq = ["<s>"] + sentence_tokens + ["</s>"]
    total_log_prob = 0.0
    for w1, w2 in zip(seq[:-1], seq[1:]):
        p = conditional_probability(model, w1, w2, laplace=laplace)
        if p <= 0.0:
            return float("-inf")
        total_log_prob += math.log(p)
    return total_log_prob


def approximate_perplexity(model: NGramModel, test_sentences: List[List[str]], laplace: bool = True) -> float:
    """Compute approximate perplexity from sentence log probabilities."""
    total_log_prob = 0.0
    total_tokens = 0

    for sentence in test_sentences:
        seq_len = len(sentence) + 1
        total_tokens += seq_len
        total_log_prob += sentence_log_probability(model, sentence, laplace=laplace)

    if total_tokens == 0:
        return float("inf")

    avg_neg_log_prob = -total_log_prob / total_tokens
    return math.exp(avg_neg_log_prob)


def run_ngram_analysis(
    cleaned_sentences: List[List[str]],
    train_split: float,
    laplace: bool,
) -> tuple[NGramModel, List[List[str]], List[str], float, float, float]:
    """Build model and return demo statistics for reporting in main."""
    split_idx = int(train_split * len(cleaned_sentences))
    train_sents = cleaned_sentences[:split_idx]
    test_sents = cleaned_sentences[split_idx:] if split_idx < len(cleaned_sentences) else cleaned_sentences

    model = build_ngram_model(train_sents)
    demo_sentence = test_sents[0] if test_sents else ["strange", "case"]
    demo_prob = sentence_probability(model, demo_sentence, laplace=laplace)
    demo_log_prob = sentence_log_probability(model, demo_sentence, laplace=laplace)
    ppl = approximate_perplexity(model, test_sents, laplace=laplace)

    return model, test_sents, demo_sentence, demo_prob, demo_log_prob, ppl
