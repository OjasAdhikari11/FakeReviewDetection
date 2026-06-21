# Fake Review Detection

An NLP project that detects **deceptive (fake) hotel reviews** and performs **sentiment (polarity) analysis** on review text. Built using classical machine learning with TF-IDF text vectorization.

## Overview

The project uses the [Deceptive Opinion Spam Corpus](https://www.kaggle.com/datasets/dshedden/deceptive-opinion-spam-corpus) — a dataset of truthful and deceptive hotel reviews collected from TripAdvisor and MTurk. Two classification tasks are tackled:

1. **Fake vs. Truthful** — Is the review deceptive or genuine?
2. **Sentiment Polarity** — Is the review positive or negative?

Multiple classifiers are trained and compared, with Logistic Regression and MLP achieving the best results.

## Features

- Fake/deceptive review detection
- Review sentiment (polarity) classification
- TF-IDF text vectorization with English stop words
- Model comparison: Logistic Regression, SVM, Naive Bayes, MLP
- scikit-learn Pipelines for clean train/predict workflows
- Interactive prediction on sample review text

## Tech Stack

| Layer | Technology |
|-------|------------|
| NLP | TF-IDF Vectorizer (scikit-learn) |
| ML models | Logistic Regression, LinearSVC, MultinomialNB, MLPClassifier |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Serialization | joblib |

## Project Structure

```
FakeReviewDetection/
├── deceptive-opinion.csv      # Dataset (~3,200 hotel reviews)
├── FinalFakeReview.ipynb      # Full EDA, training & evaluation
└── README.md
```

## Dataset

`deceptive-opinion.csv` columns:

| Column | Description |
|--------|-------------|
| deceptive | `truthful` or `deceptive` (target for fake detection) |
| hotel | Hotel name |
| polarity | `positive` or `negative` (target for sentiment) |
| source | Review source (TripAdvisor, MTurk, etc.) |
| text | Full review text |

## Model Performance

### Sentiment (Polarity) Classification

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~95% |
| LinearSVC | ~95% |
| Multinomial Naive Bayes | ~95% |
| MLP Classifier | ~95% |

### Fake vs. Truthful Review Detection

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~84% |
| LinearSVC | ~88% |
| MLP Classifier | ~88% |

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook

### Installation

```bash
git clone https://github.com/OjasAdhikari11/FakeReviewDetection.git
cd FakeReviewDetection

pip install pandas scikit-learn matplotlib seaborn jupyter joblib
```

### Run

Open and run `FinalFakeReview.ipynb` in Jupyter. The notebook covers:

1. Data loading and exploratory analysis
2. Text preprocessing with TF-IDF
3. Training and evaluating multiple classifiers
4. Confusion matrices and classification reports
5. Sample predictions on new review text

## How It Works

1. Review text is vectorized using TF-IDF with English stop words removed.
2. For polarity: a Logistic Regression model classifies reviews as positive or negative.
3. For fake detection: models distinguish between truthful and deceptive reviews.
4. Pipelines combine vectorization and classification for reproducible inference.

## Limitations

- Classical ML only (no transformers or deep NLP)
- Dataset is limited to hotel reviews in English
- Models are trained and evaluated in-notebook (no deployed API)
- Serialized models are not included in the repository

