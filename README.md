# Text Analytics and NLP

This repository contains an NLP course assignment built around a Gutenberg corpus. The main pipeline demonstrates a complete text analytics workflow, from preprocessing and morphology to word sense resources, language modeling, text generation, and POS tagging.

## Project Overview

 The script is organized into six tasks:

1. Text preprocessing and normalization
2. Morphological analysis using stemming, lemmatization, and affix stripping
3. WordNet lexical analysis and synonym expansion
4. Bigram language model construction and evaluation
5. Text generation with unsmoothed and Laplace-smoothed models
6. Part-of-speech tagging and evaluation

## Repository Contents

- [nlp_pipeline.py](nlp_pipeline.py): End-to-end pipeline that prints the assignment results
- [nlp_pipeline.ipynb](nlp_pipeline.ipynb): Notebook version of the workflow
- [NLP ASSIGNMENT/pg43 (2).txt](NLP%20ASSIGNMENT/pg43%20(2).txt): Corpus used by the pipeline

## Requirements

The project uses Python and the following main libraries:

- `nltk`
- `spacy`

The script includes fallback behavior when some NLTK resources or spaCy models are unavailable, but the best results come from having the standard NLP resources installed locally.

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies used by the notebook and script.
3. Download the required NLTK data packages if they are not already present.
4. Optionally install the spaCy English model for better lemmatization and morphology output.

Example commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install nltk spacy
python -m nltk.downloader punkt stopwords wordnet omw-1.4
python -m spacy download en_core_web_sm
```

## Running The Pipeline

Run the full assignment pipeline from the project root with:

```powershell
python nlp_pipeline.py
```

The script prints the results for each task in order and uses the bundled corpus file automatically.

## Notes

- The corpus is read from [NLP ASSIGNMENT/pg43 (2).txt](NLP%20ASSIGNMENT/pg43%20(2).txt), so keep that file in place.
- If spaCy cannot load `en_core_web_sm`, the script falls back to a blank English pipeline where possible.
- Some outputs depend on local NLTK data availability, so results may vary slightly across environments.
