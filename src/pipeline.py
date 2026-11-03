"""Lab 04 - Phish or Legit?: TF-IDF + Naive Bayes / Logistic Regression (no UI code here).

The split happens BEFORE any fitting.  Everything that learns from data (TF-IDF vocabulary and idf weights,
classifier weights, data augmentation models) lives inside pipelines fitted on the TRAIN part only - this is what
prevents data leakage.
"""
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = 42
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b\w+\.(?:com|org|info|ly)\S*", re.I)
SHORT_OR_SUSP_URL_RE = re.compile(
    r"(?:https?://)?(?:bit\.ly|tinyurl\.com|secure-[a-z0-9-]+-verify\.com|[a-z0-9-]+-login\.info)\S*",
    re.I,
)
MONEY_RE = re.compile(r"[\$£]\s?\d+(?:[.,]\d+)?")
NUM_RE = re.compile(r"\d+")


def preprocess(text: str) -> str:
    """Lower-case and replace URLs / money / numbers by placeholder tokens (normalisation, as in Lab 01)."""
    text = text.lower()
    text = URL_RE.sub(" urltok ", text)
    text = MONEY_RE.sub(" moneytok ", text)
    return NUM_RE.sub(" numtok ", text)


def preprocess_url_aware(text: str) -> str:
    """Enhanced normalisation distinguishing suspicious/shortened URLs from standard domain URLs."""
    text = text.lower()
    text = SHORT_OR_SUSP_URL_RE.sub(" urlsusptok urltok ", text)
    text = URL_RE.sub(" urltrustedtok urltok ", text)
    text = MONEY_RE.sub(" moneytok ", text)
    return NUM_RE.sub(" numtok ", text)


def load_data(path="data/messages.csv"):
    """Load synthetic or real SMS dataset (supports both CSV with text,label and TSV SMSSpamCollection)."""
    if path.endswith("SMSSpamCollection"):
        df = pd.read_csv(path, sep="\t", names=["label", "text"], encoding="utf-8")
        df["label"] = df["label"].map({"spam": "phish", "ham": "legit"}).fillna(df["label"])
    else:
        df = pd.read_csv(path)
    df["y"] = (df["label"] == "phish").astype(int)          # 1 = phish, 0 = legit
    return df


def split(df, test_size=0.25):
    """Stratified, seeded split -> keeps the 20 % phishing rate in both parts."""
    return train_test_split(df["text"], df["y"], test_size=test_size, stratify=df["y"], random_state=SEED)


def make_pipeline(kind="lr", analyzer="word", url_aware=False):
    """kind: 'nb' | 'lr' | 'nb_cal' (calibrated NB) | 'lr_cal'.
    analyzer: 'word' | 'char_wb' | 'hybrid' (Word 1-2 + Char_wb 2-5).
    """
    prep = preprocess_url_aware if url_aware else preprocess
    if analyzer == "word":
        tfidf = TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    elif analyzer == "char_wb":
        tfidf = TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(2, 5),
                                min_df=3, sublinear_tf=True)
    elif analyzer == "hybrid":
        tfidf = FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=4, sublinear_tf=True)),
        ])
    else:
        raise ValueError(f"Unknown analyzer: {analyzer}")

    if kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    elif kind == "lr":
        clf = LogisticRegression(max_iter=1000, C=3.0, random_state=SEED)
    elif kind == "nb_cal":
        clf = CalibratedClassifierCV(MultinomialNB(alpha=0.1), method="sigmoid", cv=5)
    elif kind == "lr_cal":
        clf = CalibratedClassifierCV(LogisticRegression(max_iter=1000, C=3.0, random_state=SEED),
                                     method="sigmoid", cv=5)
    else:
        raise ValueError(kind)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def fit_augmented_pipeline(pipe, Xtr_aug, ytr_aug, base_prior: float = 0.20, prior_weight: float = 0.70):
    """Fit a pipeline on class-rebalanced augmented training data and apply Bayes prior-odds
    intercept correction so predicted probabilities remain calibrated to the true population prevalence."""
    pipe.fit(Xtr_aug, ytr_aug)
    pi_aug = float(np.mean(ytr_aug))
    if 0.0 < pi_aug < 1.0 and 0.0 < base_prior < 1.0:
        shift = np.log((pi_aug / (1.0 - pi_aug)) / (base_prior / (1.0 - base_prior)))
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "intercept_"):
            clf.intercept_ = clf.intercept_ - prior_weight * shift
    return pipe


def _extract_feature_weights(clf) -> np.ndarray:
    """Extract linear feature weights (log-odds coefficients for LR or log-likelihood ratios for NB)."""
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_[0], dtype=float)
    if hasattr(clf, "feature_log_prob_"):
        return np.asarray(clf.feature_log_prob_[1] - clf.feature_log_prob_[0], dtype=float)
    if hasattr(clf, "calibrated_classifiers_"):
        weights = [_extract_feature_weights(cc.estimator) for cc in clf.calibrated_classifiers_]
        return np.mean(weights, axis=0)
    raise TypeError(f"Cannot extract feature weights from classifier {type(clf).__name__}")


def predict(pipe, text: str, threshold: float = 0.5):
    """Returns (label, score) where score = model output for the 'phish' class (NOT a calibrated probability
    unless the model was calibrated and checked)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The message is empty - paste some text to classify.")
    score = float(pipe.predict_proba([text])[0][1])
    return ("phish" if score >= threshold else "legit"), score


def contributions(pipe, text: str):
    """Per-feature contribution to the log-odds of 'phish':
    contribution = tf-idf value x weight.  Positive -> pushes toward phish, Negative -> pushes toward legit."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    X = vec.transform([text])
    raw_names = vec.get_feature_names_out()
    names = [n.replace("word__", "").replace("char__", "char:") for n in raw_names]
    weights = _extract_feature_weights(clf)
    row = X.tocoo()
    contrib = {}
    for j, v in zip(row.col, row.data):
        feat_name = names[j]
        contrib[feat_name] = contrib.get(feat_name, 0.0) + float(v * weights[j])
    return dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1])))


def word_weights(pipe, text: str):
    """Contribution of each lower-cased word of the ORIGINAL text (unigram features) - used to highlight."""
    contrib = contributions(pipe, text)
    out = {}
    for w in re.findall(r"\w+", preprocess(text)):
        out[w] = contrib.get(w, 0.0)
    # Also map original surface words if URL tokens or obfuscated char n-grams fired strongly
    for tok in re.findall(r"\w+", text.lower()):
        if tok not in out:
            if tok in contrib:
                out[tok] = contrib[tok]
            else:
                # Check if character n-grams inside tok contributed
                char_sum = sum(
                    val for k, val in contrib.items()
                    if k.startswith("char:") and k.split("char:", 1)[1].strip() in f" {tok} "
                )
                out[tok] = char_sum if abs(char_sum) > 0.15 else 0.0
    return out


def global_top_features(pipe, k=8):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    raw_names = vec.get_feature_names_out()
    names = [n.replace("word__", "").replace("char__", "char:") for n in raw_names]
    coef = _extract_feature_weights(clf)
    order = np.argsort(coef)
    return [(names[i], round(float(coef[i]), 2)) for i in order[::-1][:k]], \
           [(names[i], round(float(coef[i]), 2)) for i in order[:k]]


if __name__ == "__main__":
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    pipe = make_pipeline("lr").fit(Xtr, ytr)
    print(predict(pipe, "Northbank: your account has been locked. Verify now http://bit.ly/abc123"))
