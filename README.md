# Empathy Classification on the AcnEmpathize Dataset

This project contains an NLP experimentation pipeline for binary empathy classification on forum posts from the AcnEmpathize dataset.

The work is organized as a sequence of Jupyter notebooks, starting with exploratory analysis and ending with a final classifier.

## Project Goals

- Explore the AcnEmpathize dataset and empathy-related language patterns.
- Build and compare classical baseline models.
- Test preprocessing and feature representation variations.
- Compare sparse (TF-IDF) and dense (Word2Vec/FastText/Doc2Vec/GloVe) features.
- Train a final empathy classifier using the best choices found across experiments.

## Repository Structure

```text
assignment1/
├── 0.eda.ipynb
├── 1.baseline.ipynb
├── 2.preprocessing_variations.ipynb
├── 3.feature_representation_variations.ipynb
├── 4.dense_feature_experiments.ipynb
├── 5.final_empathy_classifier.ipynb
├── AcnEmpathize_dataset.csv
├── results/
│   ├── baseline_results.csv
│   └── dense_feature_experiments_results.csv
├── utils/
│   └── log_odds.py
```

## Notebook Workflow

Run notebooks in this order:

1. `0.eda.ipynb`
   - Dataset exploration and linguistic analysis.
   - Uses `utils/log_odds.py` for discriminative token analysis.
2. `1.baseline.ipynb`
   - Baseline models with paper-aligned preprocessing and TF-IDF features.
3. `2.preprocessing_variations.ipynb`
   - Controlled comparison of multiple preprocessing strategies.
4. `3.feature_representation_variations.ipynb`
   - TF-IDF, Bag-of-Words, LDA topic features, and combinations.
5. `4.dense_feature_experiments.ipynb`
   - Dense embeddings and comparison against the sparse reference.
6. `5.final_empathy_classifier.ipynb`
   - Final model configuration using best-performing design choices.

## Environment Setup

Python version used in this project:

- `3.13.7` (from `.python-version`)

Create and activate a virtual environment:

```bash
cd assignment1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

## NLTK and Model Downloads

Some notebooks download NLTK resources (for tokenization/lemmatization), and dense-feature experiments may fetch pretrained vectors (for example, GloVe via `gensim.downloader`).

If needed, run in Python once:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Data and Labels

- Main dataset file: `AcnEmpathize_dataset.csv`
- Target label: `combined_empathy` (`1` = empathy, `0` = non-empathy)

Typical text column used for modeling: `text`

## Saved Results

- `results/baseline_results.csv`: metrics for baseline models.
- `results/dense_feature_experiments_results.csv`: sparse vs dense feature experiment metrics.

## Notes on Reproducibility

- Most notebooks use train/test splits and model training that can be sensitive to random seeds and package versions.
- Keep notebook execution order consistent with the workflow above.
- Re-run cells top-to-bottom in each notebook for clean reproduction.

## Citation

Reference paper included in:

- `papers/2024.lrec-main.13.pdf`
