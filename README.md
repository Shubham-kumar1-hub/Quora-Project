# 📝 Quora Question Pair Duplicate Checker

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?logo=scikit-learn)
![Sentence-Transformers](https://img.shields.io/badge/SBERT-All--MiniLM--L6--v2-green)
![Flask](https://img.shields.io/badge/Framework-Flask/FastAPI-lightgrey)

An end-to-end machine learning solution designed to identify semantically equivalent questions. By combining traditional NLP linguistic features with deep learning embeddings, this tool achieves high-precision duplicate detection for Quora-style datasets.

---

## 🚀 Features

* **Web UI** — Simple browser-based form to check any two questions.
* **REST API** — `/predict` endpoint for programmatic access.
* **Semantic Similarity** — Uses **SBERT (all-MiniLM-L6-v2)** for deep sentence embeddings.
* **Rich Feature Engineering** — Fuzzy matching, character/word set ratios, longest common substring, and more.
* **Fast Inference** — Pre-trained model loaded via `joblib` for low-latency predictions.

---

## 🧠 How It Works

The system processes question pairs through a multi-stage pipeline:

### 1. Preprocessing
Both questions are cleaned by converting to lowercase, stripping HTML tags, and normalizing special characters to ensure consistency.

### 2. Feature Extraction
18 features are extracted per pair to capture both surface-level and semantic differences:

| Category | Features Included |
| :--- | :--- |
| **Fuzzy Ratios** | `fuzz_ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio` |
| **Length Metrics** | `abs_len_diff`, `mean_len` |
| **Overlap Scores** | `cwc_min/max`, `csc_min/max`, `ctc_min/max` |
| **Structural** | `first_word_eq`, `last_word_eq`, `longest_substr_ratio` |
| **Deep Learning** | `sbert_cosine`, `sbert_dot`, `sbert_absdiff` |



### 3. Classification
These features are passed to a pre-trained classifier (`quora_best_model.pkl`) which returns a prediction on whether the pair is a duplicate or not.

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/quora-duplicate-checker.git](https://github.com/yourusername/quora-duplicate-checker.git)
   cd quora-duplicate-checker
