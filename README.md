<![CDATA[<div align="center">

# DisasterSenseAI

**Disaster Tweet Classification using Transformer Ensembles**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace_Transformers-4.46-FFD21E?style=flat)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-NLP_Disaster_Tweets-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/nlp-getting-started)

</div>

---

## Overview

DisasterSenseAI classifies tweets as **disaster** or **non-disaster** using a fine-tuned ensemble of DeBERTa-v3 and Twitter-RoBERTa. It's built on the [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) dataset — 7,613 labeled tweets with a 57/43 class split.

The core challenge is distinguishing figurative language from literal reports:

| Tweet | Label |
|---|---|
| *"Forest fire near La Ronge, Sask. Canada"* | Disaster |
| *"My mixtape is straight fire"* | Not Disaster |
| *"@RedCross earthquake relief efforts underway"* | Disaster |
| *"That concert was an absolute earthquake"* | Not Disaster |

The project walks through the full ML pipeline — from TF-IDF baselines through BERT to a 5-fold cross-validated transformer ensemble.

### Features

- Binary tweet classification (disaster vs. non-disaster)
- DeBERTa-v3 + Twitter-RoBERTa ensemble with 5-fold CV
- Threshold-tuned inference for precision/recall tradeoff
- Python inference API with batch prediction support
- 7 documented Jupyter notebooks covering the full pipeline

### Planned

- Streamlit web interface with single-tweet and CSV batch prediction
- Confidence score visualization
- Interactive results dashboard
- REST API (FastAPI)

---

## Results

### Model Progression

| # | Model | Accuracy | F1 (Disaster) | Notes |
|---|---|:---:|:---:|---|
| 1 | TF-IDF + Logistic Regression | 82.2% | 0.775 | Classical ML baseline |
| 2 | BERT-base fine-tuned | 84.0% | 0.810 | First transformer |
| 3 | DeBERTa-v3-base (single split) | 85.0% | 0.822 | Best single-model result |
| 4 | DeBERTa-v3 + feature injection | 85.0% | 0.822 | Keywords + location metadata |
| 5 | **Ensemble — 5-fold CV** | **84.9%** | **0.813** | Most honest estimate |

### 5-Fold Cross-Validation (Notebook 07)

| Model | Accuracy | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| DeBERTa-v3-base | 0.8442 | 0.8448 | 0.7770 | 0.8095 |
| Twitter-RoBERTa | 0.8407 | 0.8340 | 0.7817 | 0.8070 |
| **Ensemble (tuned threshold)** | **0.8496** | **0.8648** | **0.7666** | **0.8128** |
---

## Approach

```
Raw Tweets (7,613)
    |
    +-- Data Audit ------------------- NB01: EDA, class distribution, text stats
    |
    +-- Preprocessing ---------------- NB02: Duplicate removal, TF-IDF + LR
    |                                        baseline (F1=0.775)
    |
    +-- BERT Exploration ------------- NB03-04: Tokenizer analysis, fine-tuning
    |                                           (F1=0.810)
    |
    +-- DeBERTa-v3 Fine-tuning ------- NB05-06: Raw text strategy, feature
    |                                           injection (F1=0.822)
    |
    +-- Ensemble --------------------- NB07: Twitter-RoBERTa + DeBERTa-v3,
                                             5-fold CV, threshold tuning
                                             (F1=0.813)
```

### Key decisions

- **Raw text for transformers** — aggressive preprocessing (lowercasing, stopword removal) hurts transformer performance because it strips out the linguistic cues the model was pretrained on.
- **Contradictory duplicate removal** — the dataset contains identical tweets labeled as both disaster and not-disaster. I removed both sides instead of picking one.
- **Domain-specific pretraining** — Twitter-RoBERTa (`cardiffnlp/twitter-roberta-base`) was pretrained on 58M tweets, so it handles hashtags, @mentions, and slang out of the box.
- **Probability-based ensembling** — I averaged softmax probabilities rather than raw logits so that architecturally different models contribute equally.
- **Threshold tuning on out-of-fold predictions** — tuning the classification threshold on OOF predictions avoids data leakage.

---

## Tech Stack

| Category | Technologies |
|---|---|
| ML / DL | PyTorch, Hugging Face Transformers, scikit-learn |
| Models | DeBERTa-v3-base (184M params), Twitter-RoBERTa-base (125M params) |
| Data | pandas, NumPy, NLTK |
| Visualization | matplotlib, seaborn, Plotly, WordCloud |
| Training | Google Colab (T4 GPU), Hugging Face Trainer API |
| Dev Tools | Jupyter, Git |

---

## Project Structure

```
DisasterSenseAI/
├── data/
│   ├── raw/                    # Original Kaggle dataset
│   │   ├── train.csv           # 7,613 labeled tweets
│   │   ├── test.csv            # Kaggle test set
│   │   └── sample_submission.csv
│   └── processed/
│       └── cleaned_train.csv   # Preprocessed dataset
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_bert_baseline.ipynb
│   ├── 04_bert_fine_tuning.ipynb
│   ├── 05_deberta_v3_fine_tuning.ipynb
│   ├── 06_deberta_v3_optimized.ipynb
│   └── 07_deberta_v3_advanced.ipynb
│
├── src/
│   └── models/
│       └── deberta_classifier.py   # Inference wrapper
│
├── saved_models/                   # .gitignored
├── frontend/                       # Coming soon
├── reports/figures/
├── requirements.txt
└── LICENSE
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA GPU for training (CPU works for inference)

### Installation

```bash
git clone https://github.com/abhay-astro/DisasterSenseAI.git
cd DisasterSenseAI

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Quick Inference

Requires a trained model in `saved_models/deberta_model/`.

```python
from src.models.deberta_classifier import DebertaDisasterClassifier

classifier = DebertaDisasterClassifier()

result = classifier.predict("Massive earthquake hits Turkey, thousands feared dead")
print(result)
# DisasterPrediction(label='Disaster', is_disaster=True, confidence=0.97, ...)

result = classifier.predict("That exam was an absolute disaster lol")
print(result)
# DisasterPrediction(label='Non-Disaster', is_disaster=False, confidence=0.91, ...)
```

### Training

The notebooks are set up for Google Colab with a T4 GPU:

1. Upload the notebook to Colab
2. Upload `data/raw/train.csv`
3. Run all cells
4. Download the saved model

---

## Notebooks

| Notebook | What it does | Result |
|---|---|---|
| **01** Data Audit | Dataset exploration, class balance, text stats | Distributions and summary statistics |
| **02** Feature Engineering | Text cleaning, TF-IDF, Logistic Regression | 82.2% accuracy, F1=0.775 |
| **03** BERT Baseline | Tokenizer walkthrough, zero-shot inference | Subword tokenization intuition |
| **04** BERT Fine-tuning | Full fine-tuning with Trainer API | 84.0% accuracy, F1=0.810 |
| **05** DeBERTa-v3 | DeBERTa-v3-base fine-tuning | 85.0% accuracy, F1=0.822 |
| **06** DeBERTa Optimized | Feature injection (keyword + location) | Minimal gain — raw text is enough |
| **07** Advanced Ensemble | DeBERTa + Twitter-RoBERTa, 5-fold CV | 84.9% accuracy, F1=0.813 (honest) |

---

## What I Learned

1. **85% is the ceiling** for this dataset. It matches Kaggle SOTA for honest submissions — perfect scores on the leaderboard are from test label leakage.
2. **Don't over-preprocess for transformers.** Raw text consistently outperformed aggressively cleaned text.
3. **Domain pretraining is a big deal.** Twitter-RoBERTa matched DeBERTa-v3 despite being smaller, because it was trained on 58M tweets.
4. **5-fold CV matters on small datasets.** Single-split results were ~1% inflated.
5. **Pin your dependencies.** `pip install -U transformers` silently pulled v5.x and caused NaN explosions during training. Took hours to debug.

---

## Roadmap

- [x] Data audit and EDA
- [x] Classical ML baseline (TF-IDF + Logistic Regression)
- [x] BERT fine-tuning
- [x] DeBERTa-v3 fine-tuning and optimization
- [x] Multi-model ensemble with 5-fold CV
- [ ] Ensemble inference pipeline
- [ ] Streamlit web app
- [ ] Tests
- [ ] Deployment

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [Kaggle](https://www.kaggle.com/c/nlp-getting-started) for the dataset
- [Hugging Face](https://huggingface.co) for pretrained models and Trainer
- [Cardiff NLP](https://github.com/cardiffnlp) for Twitter-RoBERTa
- [Microsoft](https://github.com/microsoft/DeBERTa) for DeBERTa-v3
]]>