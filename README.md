# 🛡️ Lab 04: Phish or Legit? — Calibrated Text Classification & Data Augmentation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phish-or-legit-lab04-mg7z3rnjygtzbgfxhoyslj.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/loubnamahrach/phish-or-legit-lab04)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

An end-to-end NLP text classification system that detects whether an incoming message is **phishing** or **legitimate**, explains its prediction via additive per-feature log-odds contributions, supports live decision-threshold tuning based on an explicit operational policy, and enhances detection robustness through a multi-strategy generative and adversarial **Data Augmentation Engine** (`PhishAugmenter`) alongside the real **UCI SMS Spam Collection**.

---

### 🌐 Liens du Projet
* **Application Web en Ligne :** [https://phish-or-legit-lab04-mg7z3rnjygtzbgfxhoyslj.streamlit.app/](https://phish-or-legit-lab04-mg7z3rnjygtzbgfxhoyslj.streamlit.app/)
* **Dépôt GitHub :** [https://github.com/loubnamahrach/phish-or-legit-lab04](https://github.com/loubnamahrach/phish-or-legit-lab04)

> ℹ️ **Notice :** *Cet outil est un support à la décision humaine et ne bloque pas les messages de manière autonome. Les données d'entraînement utilisent des marques fictives.*

---

## 📌 Project Overview & Highlights

1. **Class Imbalance & Operational Metrics:**
   - 20% positive prevalence (`phish`) vs. 80% negative prevalence (`legit`).
   - A trivial majority-class classifier predicting always `legit` achieves **80.0% accuracy** and **0.000 recall**.
   - Model performance is thus evaluated on **Precision, Recall, F1-Score, Brier Score, and Expected Calibration Error (ECE)**.

2. **Zero Data Leakage:**
   - Stratified train/test split (1,500 train / 500 test messages, `seed=42`) performed **strictly before any fitting**.
   - Vectorizers, classifiers, and data augmentation generators are trained on the training partition only.

3. **Per-Feature Additive Explainability:**
   - For Logistic Regression, the contribution of each n-gram feature to the prediction log-odds is calculated as $\text{TF-IDF}(t) \times w_t$.
   - Positive contributions push toward phishing (highlighted in 🔴 red); negative contributions push toward legitimate (highlighted in 🟢 green).

4. **Multi-Strategy NLP Data Augmentation (`PhishAugmenter`):**
   - **Markov N-Gram Language Model:** Synthesizes realistic messages using class-conditional bigram/trigram transitions fitted on `X_train`.
   - **Out-of-Fold Hard-Example Boundary Paraphrasing:** Employs 5-fold cross-validation on `X_train` to identify hard boundary cases and generates contextual paraphrases.
   - **Adversarial Homoglyph / Leetspeak Obfuscator:** Simulates attacker evasion strategies (`paypa1`, `n0rthbank`, `v3rify`, character drops/swaps) to train robust subword (`char_wb` 2–5) models.
   - **Bayes Prior-Shift Calibration:** Corrects intercept odds shifts caused by synthetic minority oversampling to maintain probability calibration to true population prevalence.

5. **External Real-World Dataset:**
   - Automatically downloads and benchmarks on the real **UCI SMS Spam Collection** (5,574 messages), as well as a domain-adapted **Curated Real Smishing** dataset.

---

## 📊 Benchmark Results

Evaluated on the untouched **500 holdout test messages** at decision threshold $\tau = 0.50$, plus an **Adversarial Obfuscated** test set:

| Classifier Pipeline | Precision | Recall | F1-Score | Accuracy | Brier Score ($\downarrow$) | ECE ($\downarrow$) | Adversarial F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline (Always `legit`)** | 0.000 | 0.000 | 0.000 | 0.800 | 0.200 | — | 0.000 |
| **Multinomial Naive Bayes (Baseline)** | 0.611 | 0.800 | 0.693 | 0.858 | 0.104 | 0.089 | 0.612 |
| **Logistic Regression (Baseline)** | 0.747 | 0.710 | 0.728 | 0.894 | 0.071 | 0.035 | 0.659 |
| **Naive Bayes + Platt Sigmoid Calibration** | 0.694 | 0.750 | 0.721 | 0.884 | 0.086 | **0.030** | 0.640 |
| **LR + Character N-grams (`char_wb` 2–5)** | 0.777 | 0.730 | 0.753 | 0.904 | 0.064 | 0.050 | 0.685 |
| **LR + Data Augmentation (Word+Char)** | 0.765 | 0.750 | 0.758 | 0.904 | 0.066 | **0.027** | 0.770 |
| **LR + Data Augmentation + URL-Aware (Enhanced)** | **0.833** | **0.800** | **0.816** | **0.928** | **0.056** | 0.038 | **0.810** |
| *External Real Dataset: UCI SMS Spam (5,574 msgs)* | *0.989* | *0.952* | *0.970* | *0.992* | *0.008* | *0.006* | — |

### Decision Threshold Policy ($\tau = 0.65$)
- **Operating Point:** $\tau = 0.65$ (Raises Baseline LR precision from **0.747** to **0.846**; Enhanced Model achieves **0.897** precision).
- **Policy Statement:** *"Because this tool supports human decisions where false positives disrupt legitimate banking or delivery messages and damage user trust, we select $\tau = 0.65$ to prioritize high precision ($\ge 85\%$): we accept missing subtle phishing messages rather than falsely alarming legitimate messages."*

---

## 📁 Repository Structure

```
04_phish_or_legit/
├── app.py                      # Interactive Streamlit Web Application (Tabs 1-4)
├── requirements.txt            # Production dependencies for local & cloud deployment
├── README.md                   # Complete documentation and setup guide
├── data/
│   ├── make_data.py            # Synthetic dataset generator (fictional brands)
│   ├── messages.csv            # 2,000 synthetic messages (20% phish, 80% legit)
│   ├── augment_data.py         # CLI for downloading UCI SMS and fitting PhishAugmenter
│   ├── uci_sms_spam.csv        # Downloaded real UCI SMS Spam Collection (5,574 msgs)
│   ├── real_smishing_curated.csv # Extracted real smishing subset (20% prevalence)
│   ├── messages_train_augmented.csv # Augmented training corpus (2,824 msgs)
│   └── feedback_log.csv        # Human analyst misclassification feedback log
├── src/
│   ├── pipeline.py             # TF-IDF, LR, Naive Bayes pipelines, explainability
│   ├── evaluate.py             # Benchmark script (calibration, curves, metrics.json)
│   ├── augmentation.py         # Markov LM, Hard-Example Miner, Adversarial Obfuscator
│   └── take_screenshots.py    # Automated headless CDP screenshot generator
└── results/
    ├── metrics.json            # Machine-readable test metrics and confusion matrices
    ├── run_log.txt             # Full evaluation logs across all models
    ├── failures.txt            # Analysis of false negatives and false positives
    ├── phish_results.png       # Confusion matrix & reliability diagrams
    ├── augmentation_comparison.png # Precision/Recall curves & Data Augmentation impact
    ├── app_screenshot_1_classify.png # Screenshot of interactive classifier
    ├── app_screenshot_2_evaluation.png # Screenshot of threshold & calibration curves
    └── app_screenshot_3_augmentation.png # Screenshot of data augmentation engine
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed on your machine.

### 2. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/loubnamahrach/phish-or-legit-lab04.git
cd phish-or-legit-lab04
pip install -r requirements.txt
```

### 3. (Optional) Run Evaluation and Data Augmentation
To re-evaluate models and re-generate all charts and benchmarks:
```bash
python -m src.evaluate
```

### 4. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Déploiement Public

L'application est déployée en production et accessible publiquement sur Streamlit Community Cloud :
* **URL de l'application :** [https://phish-or-legit-lab04-mg7z3rnjygtzbgfxhoyslj.streamlit.app/](https://phish-or-legit-lab04-mg7z3rnjygtzbgfxhoyslj.streamlit.app/)

---

## 👥 Auteur & Cadre Académique
- **Auteur :** Loubna MAHRACH
- **Établissement :** Université Hassan II de Casablanca — ENSET Mohammedia
- **Formation :** Master Informatique
- **Module :** Traitement Automatique du Langage Naturel (NLP) — Année 2026/2027
- **Projet :** Lab 04 — Phish or Legit (Classification Calibrée, Politique de Seuil & Augmentation de Données)
