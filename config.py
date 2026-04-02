"""Project configuration constants for the modular NLP pipeline."""

from pathlib import Path

SEED = 42

# Corpus settings
DEFAULT_CORPUS_PATH = Path("NLP ASSIGNMENT") / "pg43 (2).txt"

# NLP resources
NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "treebank",
]
SPACY_MODEL_NAME = "en_core_web_sm"
STOPWORDS_LANGUAGE = "english"

# Modeling parameters
TRAIN_SPLIT = 0.8
NUM_GENERATED_SENTENCES = 5
MAX_GENERATION_LENGTH = 20
USE_LAPLACE_SMOOTHING = True

# Output paths
OUTPUT_DIR = Path("outputs")
TOP_WORDS_CSV = OUTPUT_DIR / "top20_words.csv"
METRICS_TXT = OUTPUT_DIR / "preprocessing_metrics.txt"
GENERATION_TXT = OUTPUT_DIR / "generated_sentences.txt"
POS_EVAL_TXT = OUTPUT_DIR / "pos_evaluation.txt"
TOP_WORDS_PLOT = OUTPUT_DIR / "top20_frequency_plot.png"
