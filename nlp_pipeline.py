"""Organized NLP assignment pipeline for a Gutenberg corpus.

The script is structured in VS Code/Jupyter-style cells so each task can be
run independently while still supporting a single end-to-end execution.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tag import DefaultTagger, UnigramTagger
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    import spacy
except ImportError:  # pragma: no cover - depends on local environment
    spacy = None


# %% Setup And Shared Configuration

SEED = 42
RNG = random.Random(SEED)
CORPUS_PATH = Path("NLP ASSIGNMENT") / "pg43 (2).txt"

COMMON_PREFIXES = ("dis", "mis", "non", "pre", "un", "im", "in")
COMMON_SUFFIXES = (
    "ation",
    "ition",
    "ingly",
    "edly",
    "ness",
    "ment",
    "less",
    "able",
    "ible",
    "ious",
    "ance",
    "ence",
    "sion",
    "tion",
    "ship",
    "ful",
    "ive",
    "ing",
    "est",
    "ous",
    "ism",
    "ist",
    "ied",
    "ies",
    "ed",
    "ly",
    "es",
    "s",
)

# A small gold set drawn from the corpus so Task 6 remains reproducible offline.
MANUAL_POS_GOLD: List[List[Tuple[str, str]]] = [
    [("mr", "NOUN"), ("utterson", "NOUN"), ("was", "VERB"), ("a", "DET"), ("lawyer", "NOUN")],
    [("he", "PRON"), ("was", "VERB"), ("lean", "ADJ"), ("long", "ADJ"), ("dusty", "ADJ"), ("dreary", "ADJ")],
    [("yet", "ADV"), ("somehow", "ADV"), ("lovable", "ADJ")],
    [("his", "DET"), ("friends", "NOUN"), ("liked", "VERB"), ("him", "PRON")],
    [("the", "DET"), ("street", "NOUN"), ("shone", "VERB"), ("out", "PRT"), ("in", "ADP"), ("contrast", "NOUN")],
    [("the", "DET"), ("door", "NOUN"), ("was", "VERB"), ("blistered", "ADJ"), ("and", "CONJ"), ("distained", "ADJ")],
    [("the", "DET"), ("man", "NOUN"), ("trampled", "VERB"), ("calmly", "ADV"), ("over", "ADP"), ("the", "DET"), ("child", "NOUN")],
    [("it", "PRON"), ("was", "VERB"), ("hellish", "ADJ"), ("to", "PRT"), ("see", "VERB")],
    [("the", "DET"), ("doctor", "NOUN"), ("turned", "VERB"), ("sick", "ADJ"), ("and", "CONJ"), ("white", "ADJ")],
    [("i", "PRON"), ("asked", "VERB"), ("him", "PRON"), ("for", "ADP"), ("the", "DET"), ("name", "NOUN")],
    [("poole", "NOUN"), ("was", "VERB"), ("still", "ADV"), ("afraid", "ADJ")],
    [("the", "DET"), ("fog", "NOUN"), ("still", "ADV"), ("slept", "VERB"), ("on", "ADP"), ("the", "DET"), ("wing", "NOUN"), ("above", "ADP"), ("the", "DET"), ("drowned", "ADJ"), ("city", "NOUN")],
    [("the", "DET"), ("servants", "NOUN"), ("were", "VERB"), ("whispering", "VERB"), ("together", "ADV")],
    [("he", "PRON"), ("looked", "VERB"), ("at", "ADP"), ("the", "DET"), ("fire", "NOUN")],
    [("the", "DET"), ("house", "NOUN"), ("was", "VERB"), ("quiet", "ADJ"), ("and", "CONJ"), ("dark", "ADJ")],
]


@dataclass
class PreprocessingResult:
    raw_sentences: List[str]
    tokenized_sentences: List[List[str]]
    raw_word_tokens: List[str]
    cleaned_sentences: List[List[str]]
    cleaned_tokens: List[str]
    vocab_before: int
    vocab_after: int
    ttr_before: float
    ttr_after: float
    top_20: List[Tuple[str, int]]
    stemmed_vocab_size: int
    lemmatized_vocab_size: int
    stem_lemma_rows: List[Tuple[str, str, str, str]]


@dataclass
class NGramModel:
    unigram_counts: Counter
    bigram_counts: Counter
    vocab: List[str]
    vocab_size: int
    next_word_map: Dict[str, Counter]


@dataclass
class TaggingError:
    word: str
    predicted: str
    gold: str
    explanation: str


def print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def resource_available(resource_path: str) -> bool:
    try:
        nltk.data.find(resource_path)
        return True
    except Exception:
        return False


def load_spacy_model() -> Any | None:
    """Load spaCy if available; fall back to a blank English pipeline."""
    if spacy is None:
        return None

    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None


def get_corpus_text() -> str:
    """Load the corpus and remove Gutenberg front matter and back matter."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

    text = CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(
        r"(?is)^.*?\*\*\*\s*start of the project gutenberg ebook.*?\*\*\*",
        "",
        text,
    )
    text = re.sub(
        r"(?is)\*\*\*\s*end of the project gutenberg ebook.*$",
        "",
        text,
    )

    narrative_anchor = re.search(r"(?i)mr\.\s*utterson\s+the\s+lawyer", text)
    if narrative_anchor:
        text = text[narrative_anchor.start() :]

    return text.strip()


def safe_sentence_tokenize(text: str) -> List[str]:
    if resource_available("tokenizers/punkt"):
        return sent_tokenize(text)
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]


def safe_word_tokenize(text: str) -> List[str]:
    if resource_available("tokenizers/punkt"):
        return word_tokenize(text)
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text)


def get_stopwords() -> set[str]:
    if resource_available("corpora/stopwords"):
        try:
            return set(stopwords.words("english"))
        except Exception:
            pass

    return {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }


def compute_ttr(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def safe_wordnet_synsets(word: str) -> List[Any]:
    try:
        return wordnet.synsets(word)
    except Exception:
        return []


def lemmatize_tokens(tokens: Sequence[str], nlp: Any | None) -> List[str]:
    """Use spaCy lemmatization if available; otherwise fall back to lowercase."""
    if nlp is not None:
        try:
            doc = nlp(" ".join(tokens))
            lemmas = []
            for token in doc:
                lemma = token.lemma_.lower().strip() if token.lemma_ else ""
                if not lemma or lemma == "-pron-":
                    lemma = token.text.lower()
                lemmas.append(lemma)
            if len(lemmas) == len(tokens):
                return lemmas
        except Exception:
            pass

    return [token.lower() for token in tokens]


def build_stem_lemma_rows(words: Sequence[str], nlp: Any | None) -> List[Tuple[str, str, str, str]]:
    porter = PorterStemmer()
    lemmas = lemmatize_tokens(words, nlp)
    rows: List[Tuple[str, str, str, str]] = []
    for word, lemma in zip(words, lemmas):
        rows.append((word, porter.stem(word), lemma, "same" if porter.stem(word) == lemma else "different"))
    return rows


# %% Task 1 - Text Preprocessing

def preprocess_text(corpus_text: str, nlp: Any | None) -> PreprocessingResult:
    """Run sentence tokenization, word tokenization, and normalization."""
    sentences = safe_sentence_tokenize(corpus_text)
    tokenized_sentences = [safe_word_tokenize(sentence) for sentence in sentences]

    raw_word_tokens = [
        token
        for sentence in tokenized_sentences
        for token in sentence
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token)
    ]

    english_stopwords = get_stopwords()
    cleaned_sentences: List[List[str]] = []

    for sentence_tokens in tokenized_sentences:
        cleaned_sentence: List[str] = []
        for token in sentence_tokens:
            token_lower = token.lower()
            token_clean = re.sub(r"[^a-z0-9]", "", token_lower)
            if token_clean and token_clean not in english_stopwords:
                cleaned_sentence.append(token_clean)
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)

    cleaned_tokens = [token for sentence in cleaned_sentences for token in sentence]
    top_20 = Counter(cleaned_tokens).most_common(20)

    stem_focus_words = [
        "unhappy",
        "darkness",
        "swiftly",
        "plainly",
        "looking",
        "lighted",
        "opened",
        "changed",
        "walked",
        "followed",
        "returned",
        "startling",
        "consciousness",
        "disappearance",
        "suddenly",
    ]

    stem_lemma_rows = build_stem_lemma_rows(stem_focus_words, nlp)
    stemmed_vocab_size = len({PorterStemmer().stem(token) for token in cleaned_tokens})
    lemmatized_vocab_size = len(set(lemmatize_tokens(cleaned_tokens, nlp)))

    return PreprocessingResult(
        raw_sentences=sentences,
        tokenized_sentences=tokenized_sentences,
        raw_word_tokens=raw_word_tokens,
        cleaned_sentences=cleaned_sentences,
        cleaned_tokens=cleaned_tokens,
        vocab_before=len(set(raw_word_tokens)),
        vocab_after=len(set(cleaned_tokens)),
        ttr_before=compute_ttr(raw_word_tokens),
        ttr_after=compute_ttr(cleaned_tokens),
        top_20=top_20,
        stemmed_vocab_size=stemmed_vocab_size,
        lemmatized_vocab_size=lemmatized_vocab_size,
        stem_lemma_rows=stem_lemma_rows,
    )


def print_preprocessing_results(result: PreprocessingResult) -> None:
    print_header("TASK 1: TEXT PREPROCESSING")

    print(f"Sentence count: {len(result.raw_sentences)}")
    print(f"Word tokens before cleaning: {len(result.raw_word_tokens)}")
    print(f"Word tokens after cleaning:  {len(result.cleaned_tokens)}")
    print(f"Vocabulary size before cleaning: {result.vocab_before}")
    print(f"Vocabulary size after cleaning:  {result.vocab_after}")
    print(f"Type-token ratio before cleaning: {result.ttr_before:.4f}")
    print(f"Type-token ratio after cleaning:  {result.ttr_after:.4f}")
    print(f"Stemmed vocabulary size:          {result.stemmed_vocab_size}")
    print(f"Lemmatized vocabulary size:       {result.lemmatized_vocab_size}")

    print("\nTop 20 most frequent cleaned words")
    print("-" * 60)
    print(f"{'Rank':<6}{'Word':<20}{'Count':>10}")
    print("-" * 60)
    for index, (word, count) in enumerate(result.top_20, start=1):
        print(f"{index:<6}{word:<20}{count:>10}")

    print("\nStemming vs lemmatization comparison")
    print("-" * 84)
    print(f"{'Word':<18}{'Porter stem':<18}{'Lemma':<18}{'Comment':<18}")
    print("-" * 84)
    for word, stem, lemma, comment in result.stem_lemma_rows:
        print(f"{word:<18}{stem:<18}{lemma:<18}{comment:<18}")

    vocab_reduction = result.vocab_before - result.vocab_after
    token_reduction = len(result.raw_word_tokens) - len(result.cleaned_tokens)
    print("\nAnalysis of preprocessing impact")
    print("-" * 60)
    print(
        "Case folding merged uppercase and lowercase forms, punctuation removal "
        "removed noisy token variants, and stopword removal reduced high-frequency "
        "function words."
    )
    print(
        f"These steps reduced the vocabulary by {vocab_reduction} types and the corpus "
        f"by {token_reduction} word tokens."
    )
    print(
        "The post-cleaning type-token ratio is higher because token count falls sharply "
        "once repeated function words are removed, leaving a denser set of content words."
    )
    print(
        "For language modeling this usually helps lexical focus, but overly aggressive "
        "cleaning can remove grammatical cues that help predict fluent sentences."
    )


# %% Task 2 - Morphological Analysis

def manual_morpheme_dictionary() -> Dict[str, Dict[str, str]]:
    """Fifteen words selected from the corpus and manually segmented."""
    return {
        "unhappy": {"prefix": "un", "root": "happy", "suffix": ""},
        "darkness": {"prefix": "", "root": "dark", "suffix": "ness"},
        "swiftly": {"prefix": "", "root": "swift", "suffix": "ly"},
        "plainly": {"prefix": "", "root": "plain", "suffix": "ly"},
        "looking": {"prefix": "", "root": "look", "suffix": "ing"},
        "lighted": {"prefix": "", "root": "light", "suffix": "ed"},
        "opened": {"prefix": "", "root": "open", "suffix": "ed"},
        "changed": {"prefix": "", "root": "change", "suffix": "ed"},
        "walked": {"prefix": "", "root": "walk", "suffix": "ed"},
        "followed": {"prefix": "", "root": "follow", "suffix": "ed"},
        "returned": {"prefix": "", "root": "return", "suffix": "ed"},
        "startling": {"prefix": "", "root": "startle", "suffix": "ing"},
        "consciousness": {"prefix": "", "root": "conscious", "suffix": "ness"},
        "disappearance": {"prefix": "dis", "root": "appear", "suffix": "ance"},
        "suddenly": {"prefix": "", "root": "sudden", "suffix": "ly"},
    }


def normalize_rule_based_stem(stem: str, suffix: str) -> str:
    """Undo a few common spelling side effects after suffix stripping."""
    if suffix == "ied" and stem:
        return stem + "y"
    if suffix == "ies" and stem:
        return stem + "y"
    if suffix in {"ed", "ing"} and len(stem) >= 3:
        if stem.endswith(("chang", "clos", "hop", "lik", "nam", "op", "rid", "us", "wav")):
            return stem + "e"
    return stem


def rule_based_affix_strip(word: str) -> str:
    """Simple, transparent affix stripping using hand-written rules."""
    stem = word.lower()

    for prefix in COMMON_PREFIXES:
        if stem.startswith(prefix) and len(stem) - len(prefix) >= 3:
            stem = stem[len(prefix) :]
            break

    removed_suffix = ""
    for suffix in COMMON_SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
            stem = stem[: -len(suffix)]
            removed_suffix = suffix
            break

    return normalize_rule_based_stem(stem, removed_suffix)


def get_spacy_morphological_features(nlp: Any | None, words: Sequence[str]) -> Dict[str, Dict[str, str]]:
    if nlp is None:
        return {}

    try:
        doc = nlp(" ".join(words))
    except Exception:
        return {}

    features: Dict[str, Dict[str, str]] = {}
    for token in doc:
        tense = token.morph.get("Tense")
        number = token.morph.get("Number")
        features[token.text.lower()] = {
            "lemma": token.lemma_ if token.lemma_ else token.text.lower(),
            "pos": token.pos_ if token.pos_ else "N/A",
            "tense": tense[0] if tense else "N/A",
            "number": number[0] if number else "N/A",
        }
    return features


def run_morphological_analysis(nlp: Any | None) -> None:
    print_header("TASK 2: MORPHOLOGICAL ANALYSIS")

    manual_map = manual_morpheme_dictionary()
    words = list(manual_map.keys())
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    spacy_features = get_spacy_morphological_features(nlp, words)

    print("Manual morpheme analysis for 15 corpus words")
    print("-" * 76)
    print(f"{'Word':<18}{'Prefix':<12}{'Root':<22}{'Suffix':<12}")
    print("-" * 76)
    for word, parts in manual_map.items():
        print(f"{word:<18}{parts['prefix']:<12}{parts['root']:<22}{parts['suffix']:<12}")

    print("\nAffix stripping comparison")
    print("-" * 100)
    print(f"{'Word':<18}{'Rule-based':<18}{'Porter':<16}{'Snowball':<16}{'spaCy lemma':<18}")
    print("-" * 100)
    for word in words:
        rule_based = rule_based_affix_strip(word)
        porter_stem = porter.stem(word)
        snowball_stem = snowball.stem(word)
        lemma = spacy_features.get(word, {}).get("lemma", "N/A")
        print(f"{word:<18}{rule_based:<18}{porter_stem:<16}{snowball_stem:<16}{lemma:<18}")

    print("\nspaCy morphological features")
    print("-" * 72)
    print(f"{'Word':<18}{'POS':<12}{'Tense':<12}{'Number':<12}")
    print("-" * 72)
    if not spacy_features:
        print("spaCy morphological features unavailable in this environment.")
    else:
        for word in words:
            features = spacy_features.get(word, {})
            print(
                f"{word:<18}{features.get('pos', 'N/A'):<12}"
                f"{features.get('tense', 'N/A'):<12}{features.get('number', 'N/A'):<12}"
            )

    print("\nDiscussion of limitations")
    print("-" * 72)
    print(
        "The rule-based affix stripper is easy to interpret, but it ignores irregular "
        "forms, spelling alternations, and the fact that some strings only look like "
        "prefixes or suffixes."
    )
    print(
        "Porter and Snowball usually produce shorter, more normalized stems, but the "
        "result is often not a valid dictionary word."
    )
    print(
        "spaCy is more linguistically informative because it can expose lemma, tense, "
        "and number, but isolated words still lack sentence context, so some POS and "
        "morphological analyses remain approximate."
    )


# %% Task 3 - WordNet Lexical Analysis

def get_hypernym_names(synset: Any) -> List[str]:
    try:
        return [hypernym.name().split(".")[0] for hypernym in synset.hypernyms()]
    except Exception:
        return []


def select_wordnet_words(frequencies: Sequence[Tuple[str, int]], limit: int = 5) -> List[str]:
    selected: List[str] = []
    skip_words = {
        "utterson",
        "hyde",
        "jekyll",
        "mr",
        "said",
        "one",
        "sir",
        "upon",
        "would",
        "could",
        "see",
        "well",
        "even",
        "like",
    }
    for word, _count in frequencies:
        if len(word) < 3:
            continue
        if word in skip_words:
            continue
        if safe_wordnet_synsets(word):
            selected.append(word)
        if len(selected) == limit:
            break
    return selected


def run_wordnet_analysis(top_20: Sequence[Tuple[str, int]]) -> None:
    print_header("TASK 3: WORDNET LEXICAL ANALYSIS")

    selected_words = select_wordnet_words(top_20, limit=5)
    if not selected_words:
        print("WordNet data is unavailable, so this section could not be completed locally.")
        return

    print(f"Selected frequent words with WordNet coverage: {', '.join(selected_words)}")

    for word in selected_words:
        synsets = safe_wordnet_synsets(word)
        print("\n" + "-" * 96)
        print(f"Word: {word}")
        for index, synset in enumerate(synsets[:3], start=1):
            synonyms = sorted({lemma.name().replace("_", " ") for lemma in synset.lemmas()})
            hypernyms = get_hypernym_names(synset)
            print(f"Synset {index}: {synset.name()}")
            print(f"  Definition: {synset.definition()}")
            print(f"  Example synonyms: {', '.join(synonyms[:6]) if synonyms else 'N/A'}")
            print(f"  Hypernyms: {', '.join(hypernyms[:6]) if hypernyms else 'N/A'}")

    query_word = "lawyer" if "lawyer" in selected_words else selected_words[0]
    expansion_terms = sorted(
        {
            lemma.name().replace("_", " ")
            for synset in safe_wordnet_synsets(query_word)
            for lemma in synset.lemmas()
        }
    )
    expansion_terms = [term for term in expansion_terms if term.lower() != query_word.lower()]

    demo_documents = [
        "The lawyer read Dr. Jekyll's will late at night.",
        "The attorney reviewed the document before sunrise.",
        "The doctor returned to the laboratory.",
        "A careful witness described the strange visitor.",
    ]

    without_expansion = [doc for doc in demo_documents if query_word.lower() in doc.lower()]
    with_expansion = [
        doc
        for doc in demo_documents
        if any(term.lower() in doc.lower() for term in [query_word] + expansion_terms[:8])
    ]

    print("\nSynonym expansion for retrieval")
    print("-" * 96)
    print(f"Query term: {query_word}")
    print(f"Expanded terms: {', '.join([query_word] + expansion_terms[:8])}")
    print(f"Matches without expansion: {len(without_expansion)}")
    for doc in without_expansion:
        print(f"  - {doc}")
    print(f"Matches with expansion:    {len(with_expansion)}")
    for doc in with_expansion:
        print(f"  - {doc}")
    print(
        "Synonym expansion improves recall because documents using related lexical choices "
        "such as 'attorney' can still be retrieved even when the original query used 'lawyer'."
    )


# %% Task 4 - Building An N-Gram Language Model

def build_ngram_model(cleaned_sentences: Sequence[Sequence[str]]) -> NGramModel:
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    next_word_map: Dict[str, Counter] = defaultdict(Counter)

    for sentence in cleaned_sentences:
        sequence = ["<s>"] + list(sentence) + ["</s>"]
        unigram_counts.update(sequence)
        bigrams = list(zip(sequence[:-1], sequence[1:]))
        bigram_counts.update(bigrams)
        for left, right in bigrams:
            next_word_map[left][right] += 1

    vocab = sorted(unigram_counts.keys())
    return NGramModel(
        unigram_counts=unigram_counts,
        bigram_counts=bigram_counts,
        vocab=vocab,
        vocab_size=len(vocab),
        next_word_map=next_word_map,
    )


def conditional_probability(model: NGramModel, left: str, right: str, laplace: bool = False) -> float:
    bigram_count = model.bigram_counts[(left, right)]
    unigram_count = model.unigram_counts[left]

    if laplace:
        return (bigram_count + 1) / (unigram_count + model.vocab_size)

    if unigram_count == 0:
        return 0.0
    return bigram_count / unigram_count


def sentence_probability(model: NGramModel, sentence_tokens: Sequence[str], laplace: bool = True) -> float:
    probability = 1.0
    sequence = ["<s>"] + list(sentence_tokens) + ["</s>"]
    for left, right in zip(sequence[:-1], sequence[1:]):
        probability *= conditional_probability(model, left, right, laplace=laplace)
    return probability


def sentence_log_probability(model: NGramModel, sentence_tokens: Sequence[str], laplace: bool = True) -> float:
    total_log_probability = 0.0
    sequence = ["<s>"] + list(sentence_tokens) + ["</s>"]

    for left, right in zip(sequence[:-1], sequence[1:]):
        probability = conditional_probability(model, left, right, laplace=laplace)
        if probability <= 0.0:
            return float("-inf")
        total_log_probability += math.log(probability)

    return total_log_probability


def approximate_perplexity(model: NGramModel, test_sentences: Sequence[Sequence[str]], laplace: bool = True) -> float:
    total_log_probability = 0.0
    total_transitions = 0

    for sentence in test_sentences:
        total_log_probability += sentence_log_probability(model, sentence, laplace=laplace)
        total_transitions += len(sentence) + 1

    if total_transitions == 0:
        return float("inf")

    return math.exp(-total_log_probability / total_transitions)


def top_lexical_unigrams(model: NGramModel, limit: int = 10) -> List[Tuple[str, int]]:
    lexical_counts = Counter(
        {token: count for token, count in model.unigram_counts.items() if token not in {"<s>", "</s>"}}
    )
    return lexical_counts.most_common(limit)


def top_lexical_bigrams(model: NGramModel, limit: int = 10) -> List[Tuple[Tuple[str, str], int]]:
    lexical_counts = Counter(
        {
            bigram: count
            for bigram, count in model.bigram_counts.items()
            if "<s>" not in bigram and "</s>" not in bigram
        }
    )
    return lexical_counts.most_common(limit)


def print_language_model_results(cleaned_sentences: Sequence[Sequence[str]]) -> NGramModel:
    print_header("TASK 4: BUILDING AN N-GRAM LANGUAGE MODEL")

    split_index = max(1, int(0.8 * len(cleaned_sentences)))
    train_sentences = list(cleaned_sentences[:split_index])
    test_sentences = list(cleaned_sentences[split_index:]) or list(cleaned_sentences[-1:])
    model = build_ngram_model(train_sentences)

    print(f"Training sentences: {len(train_sentences)}")
    print(f"Test sentences:     {len(test_sentences)}")
    print(f"Vocabulary size:    {model.vocab_size}")
    print(f"Unique unigrams:    {len(model.unigram_counts)}")
    print(f"Unique bigrams:     {len(model.bigram_counts)}")

    print("\nTop unigram frequencies")
    print("-" * 60)
    print(f"{'Rank':<6}{'Word':<20}{'Count':>10}")
    print("-" * 60)
    for index, (word, count) in enumerate(top_lexical_unigrams(model, limit=10), start=1):
        print(f"{index:<6}{word:<20}{count:>10}")

    print("\nTop bigram frequencies")
    print("-" * 72)
    print(f"{'Rank':<6}{'Bigram':<34}{'Count':>10}{'P(w2|w1)':>16}")
    print("-" * 72)
    for index, ((left, right), count) in enumerate(top_lexical_bigrams(model, limit=10), start=1):
        probability = conditional_probability(model, left, right, laplace=False)
        print(f"{index:<6}{left + ' ' + right:<34}{count:>10}{probability:>16.4f}")

    demo_sentence = test_sentences[0]
    unsmoothed_probability = sentence_probability(model, demo_sentence, laplace=False)
    smoothed_probability = sentence_probability(model, demo_sentence, laplace=True)
    smoothed_log_probability = sentence_log_probability(model, demo_sentence, laplace=True)
    smoothed_perplexity = approximate_perplexity(model, test_sentences, laplace=True)

    print("\nSentence probability example")
    print("-" * 72)
    print(f"Test sentence: {' '.join(demo_sentence)}")
    print(f"Unsmoothed probability: {unsmoothed_probability:.12e}")
    print(f"Laplace-smoothed probability: {smoothed_probability:.12e}")
    print(f"Laplace-smoothed log probability: {smoothed_log_probability:.6f}")
    print(f"Approximate perplexity: {smoothed_perplexity:.4f}")

    return model


# %% Task 5 - Text Generation

def sample_next_word(model: NGramModel, current_word: str, laplace: bool = False) -> str:
    if not laplace:
        candidates = model.next_word_map.get(current_word)
        if not candidates:
            return "</s>"
        words = list(candidates.keys())
        weights = list(candidates.values())
        return RNG.choices(words, weights=weights, k=1)[0]

    words = [word for word in model.vocab if word != "<s>"]
    weights = [conditional_probability(model, current_word, word, laplace=True) for word in words]
    return RNG.choices(words, weights=weights, k=1)[0]


def generate_sentence(model: NGramModel, max_length: int = 18, laplace: bool = False) -> str:
    generated: List[str] = []
    current = "<s>"

    for _ in range(max_length):
        next_word = sample_next_word(model, current, laplace=laplace)
        if next_word == "</s>":
            break
        if next_word == "<s>":
            continue
        generated.append(next_word)
        current = next_word

    if not generated:
        return "[empty generation]"
    return " ".join(generated)


def print_generation_results(model: NGramModel) -> None:
    print_header("TASK 5: TEXT GENERATION")

    print("Random sentences from the unsmoothed bigram model")
    print("-" * 72)
    for index in range(1, 6):
        print(f"{index}. {generate_sentence(model, laplace=False)}")

    print("\nRandom sentences from the Laplace-smoothed bigram model")
    print("-" * 72)
    for index in range(1, 6):
        print(f"{index}. {generate_sentence(model, laplace=True)}")

    print("\nCoherence and limitations")
    print("-" * 72)
    print(
        "The unsmoothed model tends to produce more locally plausible phrases because it "
        "only follows transitions that were actually observed in training."
    )
    print(
        "The smoothed model is safer probabilistically, but its sentences are usually less "
        "coherent because smoothing assigns non-zero probability to many weak transitions."
    )
    print(
        "Both models remain limited by the bigram assumption, so they cannot reliably model "
        "long-distance agreement, discourse structure, or consistent meaning across a sentence."
    )


# %% Task 6 - POS Tagging And Evaluation

def prepare_pos_taggers() -> Tuple[DefaultTagger, UnigramTagger, List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    split_index = 10
    train_data = MANUAL_POS_GOLD[:split_index]
    test_data = MANUAL_POS_GOLD[split_index:]
    default_tagger = DefaultTagger("NOUN")
    unigram_tagger = UnigramTagger(train_data, backoff=default_tagger)
    return default_tagger, unigram_tagger, train_data, test_data


def accuracy_on_tagged_data(tagger: Any, tagged_sentences: Sequence[Sequence[Tuple[str, str]]]) -> float:
    total = 0
    correct = 0

    for sentence in tagged_sentences:
        words = [word for word, _tag in sentence]
        gold_tags = [tag for _word, tag in sentence]
        predicted_tags = [tag for _word, tag in tagger.tag(words)]
        for predicted, gold in zip(predicted_tags, gold_tags):
            total += 1
            if predicted == gold:
                correct += 1

    return correct / total if total else 0.0


def explain_tagging_error(word: str, predicted: str, gold: str) -> str:
    if predicted == "NOUN" and gold == "VERB":
        return "An unseen or rare verb defaulted to the noun backoff tag."
    if predicted == "NOUN" and gold == "ADJ":
        return "The tagger memorized no adjective pattern for this word and backed off to noun."
    if predicted == "NOUN" and gold == "ADV":
        return "Adverbs are sparse in the small training sample, so the word fell back to noun."
    if predicted == "NOUN" and gold == "ADP":
        return "Function words that were unseen in training are especially hard for a unigram model."
    if predicted == "NOUN" and gold == "CONJ":
        return "Conjunctions rely on lexical memorization here, and unseen items fall back to noun."
    return "The unigram model ignored sentence context and relied only on single-word memory."


def collect_unigram_errors(
    unigram_tagger: UnigramTagger,
    tagged_sentences: Sequence[Sequence[Tuple[str, str]]],
    limit: int = 10,
) -> List[TaggingError]:
    errors: List[TaggingError] = []
    seen: set[Tuple[str, str, str]] = set()

    for sentence in tagged_sentences:
        words = [word for word, _tag in sentence]
        gold_tags = [tag for _word, tag in sentence]
        predicted_tags = [tag for _word, tag in unigram_tagger.tag(words)]
        for word, predicted, gold in zip(words, predicted_tags, gold_tags):
            key = (word, predicted, gold)
            if predicted != gold and key not in seen:
                seen.add(key)
                errors.append(
                    TaggingError(
                        word=word,
                        predicted=predicted,
                        gold=gold,
                        explanation=explain_tagging_error(word, predicted, gold),
                    )
                )
            if len(errors) == limit:
                return errors

    return errors


def tag_corpus_preview(
    corpus_text: str,
    unigram_tagger: UnigramTagger,
    sentence_limit: int = 5,
) -> List[List[Tuple[str, str]]]:
    preview: List[List[Tuple[str, str]]] = []
    for sentence in safe_sentence_tokenize(corpus_text)[:sentence_limit]:
        words = [token.lower() for token in safe_word_tokenize(sentence) if token.isalpha()]
        if words:
            preview.append(unigram_tagger.tag(words))
    return preview


def run_pos_tagging_evaluation(corpus_text: str) -> None:
    print_header("TASK 6: POS TAGGING AND EVALUATION")

    default_tagger, unigram_tagger, train_data, test_data = prepare_pos_taggers()

    print(
        "Resource note: the pretrained NLTK perceptron tagger and Treebank corpus are not "
        "available offline in this environment, so Task 6 uses NLTK's DefaultTagger and "
        "UnigramTagger on a manually gold-tagged subset of corpus sentences."
    )
    print(f"Training sentences in gold subset: {len(train_data)}")
    print(f"Test sentences in gold subset:     {len(test_data)}")

    tagged_preview = tag_corpus_preview(corpus_text, unigram_tagger)
    print("\nTagged corpus preview")
    print("-" * 96)
    for index, sentence in enumerate(tagged_preview[:3], start=1):
        print(f"Sentence {index}: {sentence}")

    default_accuracy = accuracy_on_tagged_data(default_tagger, test_data)
    unigram_accuracy = accuracy_on_tagged_data(unigram_tagger, test_data)

    print("\nAccuracy comparison on held-out gold sentences")
    print("-" * 96)
    print(f"Default tagger accuracy: {default_accuracy:.4f}")
    print(f"Unigram tagger accuracy: {unigram_accuracy:.4f}")

    print("\nAt least 10 unigram tagging errors")
    print("-" * 120)
    print(f"{'Word':<16}{'Predicted':<14}{'Gold':<14}{'Explanation'}")
    print("-" * 120)
    for error in collect_unigram_errors(unigram_tagger, test_data, limit=10):
        print(f"{error.word:<16}{error.predicted:<14}{error.gold:<14}{error.explanation}")

    print("\nLinguistic explanation")
    print("-" * 96)
    print(
        "Most errors come from lexical ambiguity and data sparsity. A unigram tagger only "
        "remembers the most likely tag for each seen word and falls back when a word is unseen."
    )
    print(
        "That means it cannot use surrounding context to distinguish adjectives from nouns, "
        "verbs from participial adjectives, or adverbs from other open-class categories."
    )


# %% Final Written Answers

def print_assignment_answers() -> None:
    print_header("SHORT ANSWERS TO THE ASSIGNMENT QUESTIONS")

    answers = [
        (
            "1. How does preprocessing influence language modeling results?",
            "Preprocessing removes noise, reduces vocabulary size, and lowers sparsity, so probability estimates become more stable. The trade-off is that heavy cleaning can remove function words and punctuation that help model syntax and style.",
        ),
        (
            "2. How does morphology affect vocabulary size and sparsity?",
            "Morphology creates many surface variants such as walked, walking, and walks. Those variants increase vocabulary size and push more words into the low-frequency tail, which makes sparse count-based models harder to estimate.",
        ),
        (
            "3. Why is smoothing necessary?",
            "Without smoothing, any unseen n-gram gives a sentence probability of zero. Smoothing reallocates a little probability mass to unseen events so the model can score new sentences and produce finite perplexity.",
        ),
        (
            "4. What limitations did you observe in N-gram models?",
            "The bigram model only uses very short context, so it often produces local fragments that sound plausible but lose global meaning. It also struggles with unseen word combinations, long-range dependencies, and consistent grammar across a sentence.",
        ),
        (
            "5. How might neural language models improve performance?",
            "Neural language models learn distributed representations, share information across related words, use much longer context windows, and can model subword patterns. In practice that usually means lower perplexity, better fluency, and better handling of morphology and ambiguity.",
        ),
    ]

    for question, answer in answers:
        print(question)
        print(f"   {answer}\n")


# %% Main Execution

def main() -> None:
    RNG.seed(SEED)
    nlp = load_spacy_model()
    corpus_text = get_corpus_text()

    preprocessing_result = preprocess_text(corpus_text, nlp)
    print_preprocessing_results(preprocessing_result)

    run_morphological_analysis(nlp)
    run_wordnet_analysis(preprocessing_result.top_20)

    language_model = print_language_model_results(preprocessing_result.cleaned_sentences)
    print_generation_results(language_model)

    run_pos_tagging_evaluation(corpus_text)
    print_assignment_answers()


if __name__ == "__main__":
    main()
