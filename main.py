"""Entry point for the modular NLP pipeline project."""

from __future__ import annotations

from config import (
    GENERATION_TXT,
    MAX_GENERATION_LENGTH,
    NUM_GENERATED_SENTENCES,
    OUTPUT_DIR,
    POS_EVAL_TXT,
    SEED,
    TOP_WORDS_PLOT,
    TRAIN_SPLIT,
    USE_LAPLACE_SMOOTHING,
)
from morphology.morphology import run_morphology_analysis
from ngram_model.ngram import run_ngram_analysis
from pos_tagging.pos_tagging import evaluate_pos_tagging, format_pos_report, save_pos_report
from preprocessing.preprocessing import run_preprocessing
from text_generation.generator import format_generation_report, generate_sentences
from utils import (
    download_nltk_resources,
    ensure_output_dir,
    get_corpus_text,
    load_spacy_model,
    print_header,
    save_text,
    save_top20_plot,
    setup_reproducibility,
)
from wordnet_analysis.wordnet_analysis import run_wordnet_analysis


def run_pipeline(corpus_text: str | None = None) -> None:
    """Execute the complete NLP pipeline in required task order."""
    setup_reproducibility(SEED)
    ensure_output_dir(OUTPUT_DIR)
    download_nltk_resources()
    nlp = load_spacy_model()

    corpus = get_corpus_text(corpus_text)

    print_header("TASK 1: TEXT PREPROCESSING")
    preprocessing_result = run_preprocessing(corpus, OUTPUT_DIR)

    print_header("TASK 2: MORPHOLOGICAL ANALYSIS")
    run_morphology_analysis(nlp)

    print_header("TASK 3: WORDNET LEXICAL ANALYSIS")
    top_words = [w for w, _ in preprocessing_result.top_20]
    run_wordnet_analysis(top_words)

    print_header("TASK 4: N-GRAM LANGUAGE MODEL")
    model, test_sents, demo_sentence, demo_prob, demo_log_prob, ppl = run_ngram_analysis(
        preprocessing_result.cleaned_sentences,
        train_split=TRAIN_SPLIT,
        laplace=USE_LAPLACE_SMOOTHING,
    )
    train_count = int(TRAIN_SPLIT * len(preprocessing_result.cleaned_sentences))
    print(f"Training sentences: {train_count}")
    print(f"Test sentences:     {len(test_sents)}")
    print(f"Vocabulary size:    {model.vocab_size}")
    print(f"Unique unigrams:    {len(model.unigram_counts)}")
    print(f"Unique bigrams:     {len(model.bigram_counts)}")
    print("\nSentence probability demonstration (Laplace smoothed)")
    print("-" * 90)
    print(f"Test sentence: {' '.join(demo_sentence)}")
    print(f"Probability:   {demo_prob:.12e}")
    print(f"Log-prob:      {demo_log_prob:.6f}")
    print(f"Perplexity:    {ppl:.4f}")

    print_header("TASK 5: TEXT GENERATION")
    raw_sentences = generate_sentences(
        model,
        num_sentences=NUM_GENERATED_SENTENCES,
        max_len=MAX_GENERATION_LENGTH,
        laplace=False,
    )
    smooth_sentences = generate_sentences(
        model,
        num_sentences=NUM_GENERATED_SENTENCES,
        max_len=MAX_GENERATION_LENGTH,
        laplace=True,
    )
    generation_report = format_generation_report(raw_sentences, smooth_sentences)
    print(generation_report)
    save_text(OUTPUT_DIR / GENERATION_TXT.name, generation_report)

    print_header("TASK 6: POS TAGGING AND EVALUATION")
    default_acc, unigram_acc, errors, preview_text = evaluate_pos_tagging(corpus)
    pos_report = format_pos_report(preview_text, default_acc, unigram_acc, errors)
    print(pos_report)
    save_pos_report(pos_report, OUTPUT_DIR / POS_EVAL_TXT.name)

    print_header("BONUS OUTPUTS")
    words = [w for w, _ in preprocessing_result.top_20]
    counts = [c for _, c in preprocessing_result.top_20]
    print(save_top20_plot(words, counts, OUTPUT_DIR / TOP_WORDS_PLOT.name))
    print(f"Saved: {OUTPUT_DIR / GENERATION_TXT.name}")
    print(f"Saved: {OUTPUT_DIR / POS_EVAL_TXT.name}")


if __name__ == "__main__":
    provided_corpus = globals().get("corpus_text")
    run_pipeline(provided_corpus if isinstance(provided_corpus, str) else None)
