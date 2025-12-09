# Final Project: Sentiment-Driven Movie Review Classification Using DistilBERT

---

## 1. Project Overview

This repository contains the code and report for final project.  
The goal is to build a sentiment classification system on the IMDB movie review dataset and compare:

1. A **TF–IDF + Logistic Regression** baseline.
2. A **fine-tuned DistilBERT** model using Hugging Face Transformers.

The entire implementation is in a single Colab-style Jupyter notebook:

- `stats507final.ipynb`

The final report is written in IEEE conference format:

- `report/final_report.pdf`

---

## 2. Files

- `stats507final.ipynb`  
  - End-to-end pipeline: data loading, EDA, baseline, DistilBERT fine-tuning, evaluation, and error analysis.
  - Designed to run on **Google Colab**.
- `figures/` (optional but recommended)
  - `imdb_length_histogram.png`
  - `confusion_baseline.png`
  - `confusion_distilbert.png`
  - `training_curve_distilbert.png`
- `report/final_report.tex`
- `report/final_report.pdf`

---

## 3. How to Run (Google Colab)

1. Open the notebook in Colab:
   - Option A: From GitHub → “Open in Colab”
   - Option B: Download `stats507final.ipynb` and upload it to https://colab.research.google.com

2. Run the notebook from top to bottom:
   - The first cell installs all required packages:
     ```python
     !pip install -q "transformers>=4.45.0" "datasets>=3.0.0" \
                   "accelerate>=1.0.0" "evaluate>=0.4.0" \
                   scikit-learn matplotlib
     ```
   - GPU is automatically used if available in Colab.

3. The notebook will:
   - Load the IMDB dataset via `datasets.load_dataset("imdb")`
   - Perform basic EDA (length distribution, label balance)
   - Train and evaluate the TF–IDF + Logistic Regression baseline
   - Fine-tune `distilbert-base-uncased` on IMDB
   - Evaluate on validation and test sets
   - Generate and save figures in the `figures/` folder (if running locally)

---

## 4. Key Results (IMDB Test Set)

- **Baseline (TF–IDF + Logistic Regression)**  
  - Accuracy: **0.8814**  
  - F1-score: **≈ 0.88**

- **DistilBERT (Fine-Tuned)**  
  - Accuracy: **0.9086**  
  - F1-score: **0.9091**  
  - Confusion matrix (negative / positive):
    - True negative: 11,294  
    - False positive: 1,206  
    - False negative: 1,079  
    - True positive: 11,421  

---

## 5. Error Analysis (Summary)

By inspecting the top 20 highest-confidence misclassified examples, we find that:

- The model is **overly influenced by strongly positive lexical cues** in reviews that are globally negative.
- Because inputs are truncated to 216 tokens, **negative content at the end of long reviews is often discarded**, leading to false positives.
- The model struggles with **sarcasm and rhetorical exaggeration**, where literally positive phrases express negative intent.
- Short, ambiguous reviews are also harder to classify.

More details are provided in `report/final_report.pdf`.

---
