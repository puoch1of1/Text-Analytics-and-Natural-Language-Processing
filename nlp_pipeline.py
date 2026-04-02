"""Mini NLP pipeline project for academic submission.

This script implements:
1) Text preprocessing
2) Morphological analysis
3) WordNet lexical analysis
4) N-gram language modeling
5) Text generation
6) POS tagging and evaluation

It is designed to run end-to-end with clean, labeled outputs.
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nltk
import numpy as np
import spacy
from nltk.corpus import stopwords, treebank, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tag import UnigramTagger
from nltk.tokenize import sent_tokenize, word_tokenize


SEED = 42


@dataclass
class PreprocessingResult:
    """Container for preprocessing outputs and metrics."""

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


@dataclass
class NGramModel:
    """Simple bigram language model with optional Laplace smoothing."""

    unigram_counts: Counter
    bigram_counts: Counter
    vocab: List[str]
    vocab_size: int
    next_word_map: Dict[str, Counter]


def print_header(title: str) -> None:
    """Print a consistent section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def setup_reproducibility(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def download_nltk_resources() -> None:
    """Download required NLTK resources if they are missing."""
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "treebank",
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)


def load_spacy_model() -> spacy.language.Language:
    """Load spaCy English model, downloading it if necessary.

    Returns:
        A spaCy language pipeline.
    """
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        try:
            from spacy.cli import download

            download(model_name)
            return spacy.load(model_name)
        except Exception:
            # Fallback keeps the script executable even if model download is blocked.
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp


def get_corpus_text() -> str:
    """Get corpus text.

    Priority:
    1) Existing global variable named corpus_text (if injected in notebook/script context)
    2) Default local file path in this workspace
    """
    global_vars = globals()
    if "corpus_text" in global_vars and isinstance(global_vars["corpus_text"], str):
        return global_vars["corpus_text"]

    default_path = Path("NLP ASSIGNMENT") / "pg43 (2).txt"
    if not default_path.exists():
        raise FileNotFoundError(
            "corpus_text was not provided, and default corpus file was not found."
        )
    return default_path.read_text(encoding="utf-8", errors="ignore")


def compute_ttr(tokens: Sequence[str]) -> float:
    """Compute type-token ratio for a token sequence."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def preprocess_text(corpus_text: str) -> PreprocessingResult:
    """Run sentence tokenization, word tokenization, and normalization.

    Normalization includes lowercase conversion, punctuation removal via regex,
    and NLTK stopword removal.
    """
    sentences = sent_tokenize(corpus_text)
    tokenized = [word_tokenize(sentence) for sentence in sentences]
    raw_tokens = [token for sentence in tokenized for token in sentence]

    english_stopwords = set(stopwords.words("english"))
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
    """Print Task 1 metrics and interpretation."""
    print_header("TASK 1: TEXT PREPROCESSING")

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
    """Simple rule-based suffix stripping.

    Removes common suffixes: ing, ed, es, s.
    """
    word_l = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if word_l.endswith(suffix) and len(word_l) > len(suffix) + 2:
            return word_l[: -len(suffix)]
    return word_l


def spacy_morphological_features(nlp: spacy.language.Language, words: List[str]) -> Dict[str, Dict[str, str]]:
    """Extract POS, tense, number and lemma for a word list using spaCy."""
    text = " ".join(words)
    doc = nlp(text)
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


def run_morphological_analysis(nlp: spacy.language.Language) -> None:
    """Run Task 2 morphological analysis and print a comparison table."""
    print_header("TASK 2: MORPHOLOGICAL ANALYSIS")

    manual_map = manual_morpheme_dictionary()
    words = list(manual_map.keys())

    print("Manual morpheme decomposition (15 words)")
    print("-" * 70)
    print(f"{'Word':<16}{'Prefix':<12}{'Root':<20}{'Suffix':<12}")
    print("-" * 70)
    for word, parts in manual_map.items():
        print(
            f"{word:<16}{parts['prefix']:<12}{parts['root']:<20}{parts['suffix']:<12}"
        )

    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    spacy_feats = spacy_morphological_features(nlp, words)

    print("\nComparison table")
    print("-" * 98)
    print(
        f"{'Word':<16}{'Rule-based':<16}{'Porter':<14}{'Snowball':<14}{'Lemma (spaCy)':<18}"
    )
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
        print(
            f"{word:<16}{feats.get('pos', 'N/A'):<10}{feats.get('tense', 'N/A'):<12}{feats.get('number', 'N/A'):<12}"
        )

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


def get_hypernym_names(synset: wordnet.synset) -> List[str]:
    """Return readable hypernym names from a synset."""
    return [h.name().split(".")[0] for h in synset.hypernyms()]


def run_wordnet_analysis(top_words: List[str]) -> None:
    """Run Task 3 WordNet lexical analysis."""
    print_header("TASK 3: WORDNET LEXICAL ANALYSIS")

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

    # Synonym expansion demo
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
        doc
        for doc in docs
        if any(term.lower() in doc.lower() for term in expanded_terms)
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


def build_ngram_model(cleaned_sentences: List[List[str]]) -> NGramModel:
    """Build unigram and bigram counts from cleaned sentences."""
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


def conditional_probability(
    model: NGramModel,
    w1: str,
    w2: str,
    laplace: bool = False,
) -> float:
    """Compute conditional probability P(w2|w1), with optional Laplace smoothing."""
    bigram_count = model.bigram_counts[(w1, w2)]
    unigram_count = model.unigram_counts[w1]

    if laplace:
        return (bigram_count + 1) / (unigram_count + model.vocab_size)

    if unigram_count == 0:
        return 0.0
    return bigram_count / unigram_count


def sentence_probability(
    model: NGramModel,
    sentence_tokens: List[str],
    laplace: bool = True,
) -> float:
    """Compute sentence probability via bigram chain rule."""
    seq = ["<s>"] + sentence_tokens + ["</s>"]
    prob = 1.0
    for w1, w2 in zip(seq[:-1], seq[1:]):
        p = conditional_probability(model, w1, w2, laplace=laplace)
        prob *= p
    return prob


def sentence_log_probability(
    model: NGramModel,
    sentence_tokens: List[str],
    laplace: bool = True,
) -> float:
    """Compute log probability of a sentence to avoid underflow."""
    seq = ["<s>"] + sentence_tokens + ["</s>"]
    total_log_prob = 0.0
    for w1, w2 in zip(seq[:-1], seq[1:]):
        p = conditional_probability(model, w1, w2, laplace=laplace)
        if p <= 0.0:
            return float("-inf")
        total_log_prob += math.log(p)
    return total_log_prob


def approximate_perplexity(
    model: NGramModel,
    test_sentences: List[List[str]],
    laplace: bool = True,
) -> float:
    """Compute approximate perplexity using sentence log probabilities."""
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


def sample_next_word(
    model: NGramModel,
    current_word: str,
    laplace: bool = False,
) -> str:
    """Sample the next word from bigram probabilities."""
    if not laplace:
        candidates = model.next_word_map.get(current_word)
        if not candidates:
            return "</s>"
        words = list(candidates.keys())
        probs = np.array(list(candidates.values()), dtype=np.float64)
        probs = probs / probs.sum()
        return np.random.choice(words, p=probs)

    # Laplace-smoothed sampling over the full vocabulary.
    words = model.vocab
    probs = np.array(
        [conditional_probability(model, current_word, w, laplace=True) for w in words],
        dtype=np.float64,
    )
    probs = probs / probs.sum()
    return str(np.random.choice(words, p=probs))


def generate_sentence(
    model: NGramModel,
    max_len: int = 20,
    laplace: bool = False,
) -> str:
    """Generate a single sentence from the bigram model."""
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


def run_language_model_and_generation(cleaned_sentences: List[List[str]]) -> None:
    """Run Tasks 4 and 5: language model, perplexity, and text generation."""
    print_header("TASK 4: N-GRAM LANGUAGE MODEL")

    split_idx = int(0.8 * len(cleaned_sentences))
    train_sents = cleaned_sentences[:split_idx]
    test_sents = cleaned_sentences[split_idx:] if split_idx < len(cleaned_sentences) else cleaned_sentences

    model = build_ngram_model(train_sents)

    print(f"Training sentences: {len(train_sents)}")
    print(f"Test sentences:     {len(test_sents)}")
    print(f"Vocabulary size:    {model.vocab_size}")
    print(f"Unique unigrams:    {len(model.unigram_counts)}")
    print(f"Unique bigrams:     {len(model.bigram_counts)}")

    # Use first test sentence for probability demonstration.
    demo_sentence = test_sents[0] if test_sents else ["strange", "case"]
    demo_prob = sentence_probability(model, demo_sentence, laplace=True)
    demo_log_prob = sentence_log_probability(model, demo_sentence, laplace=True)
    ppl = approximate_perplexity(model, test_sents, laplace=True)

    print("\nSentence probability demonstration (Laplace smoothed)")
    print("-" * 90)
    print(f"Test sentence: {' '.join(demo_sentence)}")
    print(f"Probability:   {demo_prob:.12e}")
    print(f"Log-prob:      {demo_log_prob:.6f}")
    print(f"Perplexity:    {ppl:.4f}")

    print_header("TASK 5: TEXT GENERATION")
    print("Generated sentences WITHOUT smoothing")
    print("-" * 90)
    for i in range(1, 6):
        print(f"{i}. {generate_sentence(model, max_len=20, laplace=False)}")

    print("\nGenerated sentences WITH Laplace smoothing")
    print("-" * 90)
    for i in range(1, 6):
        print(f"{i}. {generate_sentence(model, max_len=20, laplace=True)}")

    print("\nAnalysis")
    print("-" * 90)
    print(
        "Without smoothing, generated text is usually more locally coherent but limited "
        "to observed transitions."
    )
    print(
        "With Laplace smoothing, text is more diverse but often less coherent because "
        "it allows many low-probability transitions."
    )
    print(
        "Bigram models capture only short-range context, so they struggle with global "
        "syntax, semantics, and long-distance dependencies."
    )


def evaluate_pos_taggers(corpus_text: str) -> None:
    """Run Task 6 POS tagging and evaluation.

    - Applies NLTK default POS tagger to the project corpus (demonstration).
    - Trains/evaluates an NLTK UnigramTagger on Treebank for measurable accuracy.
    """
    print_header("TASK 6: POS TAGGING AND EVALUATION")

    # POS tagging on project corpus
    sentences = sent_tokenize(corpus_text)
    corpus_word_sents = [word_tokenize(s) for s in sentences[:20]]
    corpus_tagged_preview = [nltk.pos_tag(ws) for ws in corpus_word_sents[:3] if ws]

    print("NLTK default POS tagging preview on corpus (first 3 sentences)")
    print("-" * 90)
    for i, tagged in enumerate(corpus_tagged_preview, start=1):
        print(f"Sentence {i}: {tagged}")

    # Supervised evaluation using Treebank gold tags
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

    print("\nAccuracy comparison")
    print("-" * 90)
    print(f"Default NLTK tagger accuracy: {default_acc:.4f}")
    print(f"Unigram tagger accuracy:      {unigram_acc:.4f}")

    print("\nAt least 10 unigram tagging errors (predicted vs gold)")
    print("-" * 90)
    print(f"{'Word':<20}{'Predicted':<14}{'Gold':<14}")
    print("-" * 90)
    for word, pred, gold in errors:
        print(f"{word:<20}{pred:<14}{gold:<14}")

    print("\nLinguistic explanation")
    print("-" * 90)
    print(
        "Tagging errors often come from lexical ambiguity (e.g., words functioning as "
        "noun vs verb in different contexts)."
    )
    print(
        "Unknown or rare words are harder for unigram models because they rely mostly on "
        "single-word statistics from training data."
    )
    print(
        "A unigram tagger ignores sentence context, so it cannot effectively resolve "
        "context-dependent POS decisions."
    )


def save_bonus_outputs(
    preprocessing_result: PreprocessingResult,
    output_dir: Path,
) -> None:
    """Save bonus outputs to files and plot frequency distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "preprocessing_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("TASK 1 PREPROCESSING METRICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Vocabulary before: {preprocessing_result.vocab_before}\n")
        f.write(f"Vocabulary after:  {preprocessing_result.vocab_after}\n")
        f.write(f"TTR before: {preprocessing_result.ttr_before:.6f}\n")
        f.write(f"TTR after:  {preprocessing_result.ttr_after:.6f}\n")

    top_words_csv = output_dir / "top20_words.csv"
    with top_words_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "count"])
        for idx, (word, count) in enumerate(preprocessing_result.top_20, start=1):
            writer.writerow([idx, word, count])

    try:
        import matplotlib.pyplot as plt

        words = [w for w, _ in preprocessing_result.top_20]
        counts = [c for _, c in preprocessing_result.top_20]

        plt.figure(figsize=(12, 5))
        plt.bar(words, counts)
        plt.title("Top 20 Word Frequencies (Cleaned Corpus)")
        plt.xlabel("Word")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = output_dir / "top20_frequency_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nBonus plot saved: {plot_path}")
    except Exception as exc:
        print(f"\nMatplotlib plot skipped: {exc}")

    print(f"Bonus files saved in: {output_dir}")


def main() -> None:
    """Execute the complete NLP pipeline."""
    setup_reproducibility(SEED)
    download_nltk_resources()
    nlp = load_spacy_model()

    corpus = get_corpus_text()

    preprocessing_result = preprocess_text(corpus)
    print_preprocessing_results(preprocessing_result)

    run_morphological_analysis(nlp)

    top_words = [w for w, _ in preprocessing_result.top_20]
    run_wordnet_analysis(top_words)

    run_language_model_and_generation(preprocessing_result.cleaned_sentences)

    evaluate_pos_taggers(corpus)

    print_header("BONUS OUTPUTS")
    save_bonus_outputs(preprocessing_result, Path("outputs"))


if __name__ == "__main__":
    main()
