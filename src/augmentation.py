"""Lab 04 - Advanced Data Augmentation & External Real Dataset Integration.

This module provides two complementary approaches to enrich the training pipeline
without any data leakage into the evaluation test set:

1. External Dataset Download & Domain Adaptation (UCI SMS Spam Collection + Real Smishing subset):
   - Downloads the official UCI SMS Spam Collection (5,572 real SMS messages).
   - Extracts a domain-adapted real smishing/phishing-style subset (URLs, bank/account/verify/prize hooks)
     alongside full UCI benchmark support.

2. Multi-Strategy Generative & Adversarial Data Augmentation Model (PhishAugmenter):
   Fitted strictly on the training split (X_train, y_train) to prevent data leakage:
   - Strategy A (Markov N-gram Language Model): Class-conditional bigram/trigram generator fitted on X_train.
   - Strategy B (Hard-Example Boundary Augmentation): Uses out-of-fold cross-validation probabilities on X_train
     to identify hard minority/boundary cases and synthesize targeted variations.
   - Strategy C (Contextual Lexical Paraphrasing): Domain-preserving synonym & clause replacement.
   - Strategy D (Adversarial Obfuscation & Homoglyph Simulator): Simulates evasion techniques ('paypa1',
     'n0rthbank', 'v3rify', character drops/swaps) to harden word + char_wb models.
"""
from __future__ import annotations

import csv
import io
import os
import random
import re
import urllib.request
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

SEED = 42
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

# Domain-specific contextual synonyms that preserve class semantics
CONTEXTUAL_SYNONYMS: Dict[str, List[str]] = {
    "verify": ["confirm", "validate", "authenticate", "check"],
    "confirm": ["verify", "validate", "acknowledge"],
    "suspended": ["locked", "restricted", "frozen", "disabled", "blocked"],
    "locked": ["suspended", "restricted", "frozen", "blocked"],
    "immediately": ["right away", "urgently", "at once", "promptly"],
    "package": ["parcel", "shipment", "delivery", "order"],
    "parcel": ["package", "shipment", "delivery"],
    "notice": ["alert", "notification", "warning", "update"],
    "payment": ["transaction", "billing", "charge", "transfer"],
    "invoice": ["statement", "bill", "receipt"],
    "pending": ["awaiting confirmation", "processing", "on hold"],
    "reward": ["prize", "bonus", "voucher", "gift card"],
    "appointment": ["meeting", "session", "consultation"],
    "reminder": ["heads up", "note", "fyi"],
}

# Leetspeak / homoglyph substitutions used by phishers to evade word-level filters
ADVERSARIAL_CHAR_MAP: Dict[str, str] = {
    "a": "4",
    "e": "3",
    "i": "1",
    "l": "1",
    "o": "0",
    "s": "5",
}

OBFUSCATION_TARGET_WORDS = {
    "verify", "account", "bank", "northbank", "immediately", "suspended",
    "alert", "notice", "payment", "confirm", "security", "update", "urgent",
    "parcel", "package", "delta", "atlas", "civic", "paypal",
}


def download_uci_dataset(
    uci_csv_path: str = "data/uci_sms_spam.csv",
    curated_smish_path: str = "data/real_smishing_curated.csv",
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download the real UCI SMS Spam Collection and also extract a curated Smishing-style subset."""
    os.makedirs(os.path.dirname(uci_csv_path) or ".", exist_ok=True)

    if os.path.exists(uci_csv_path) and os.path.exists(curated_smish_path):
        uci_df = pd.read_csv(uci_csv_path)
        smish_df = pd.read_csv(curated_smish_path)
        return uci_df, smish_df

    rows = []
    try:
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            content = zf.read("SMSSpamCollection").decode("utf-8", errors="replace")
        for line in content.splitlines():
            if "\t" not in line:
                continue
            raw_label, text = line.split("\t", 1)
            raw_label = raw_label.strip().lower()
            text = text.strip()
            if raw_label in ("spam", "ham") and text:
                # In UCI SMS dataset, 'spam' is positive class; mapped to 'phish' column convention for pipeline compatibility
                mapped_label = "phish" if raw_label == "spam" else "legit"
                rows.append((text, mapped_label, raw_label))
    except Exception as exc:
        raise RuntimeError(f"Could not download UCI SMS Spam dataset: {exc}") from exc

    uci_df = pd.DataFrame(rows, columns=["text", "label", "original_uci_label"])
    uci_df["y"] = (uci_df["label"] == "phish").astype(int)
    uci_df.to_csv(uci_csv_path, index=False, encoding="utf-8")

    # Extract a realistic Smishing/Phishing subset from UCI where spam messages contain phishing cues
    # (URLs, bank/account/urgent/verify/claim/prize/security/code hooks) + normalize £ -> $
    smish_pattern = re.compile(
        r"(?:http|www\.|\.com|\.co|\.uk|\.net|urgent|account|verify|bank|claim|prize|"
        r"award|won|code|security|suspended|locked|customer|service|delivery|package|cash|£|\$)",
        re.I,
    )

    def normalize_currency(t: str) -> str:
        t = re.sub(r"£\s?(\d+)", r"$\1", t)
        return t

    spam_candidates = uci_df[uci_df["label"] == "phish"].copy()
    smish_positives = spam_candidates[spam_candidates["text"].apply(lambda s: bool(smish_pattern.search(s)))].copy()
    smish_positives["text"] = smish_positives["text"].apply(normalize_currency)

    rng = np.random.default_rng(seed)
    n_pos = len(smish_positives)
    n_neg = min(len(uci_df[uci_df["label"] == "legit"]), n_pos * 4)  # Maintain 20% positive prevalence
    legit_pool = uci_df[uci_df["label"] == "legit"].sample(n=n_neg, random_state=seed).copy()
    legit_pool["text"] = legit_pool["text"].apply(normalize_currency)

    smish_df = pd.concat([smish_positives[["text", "label"]], legit_pool[["text", "label"]]], ignore_index=True)
    smish_df = smish_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    smish_df["y"] = (smish_df["label"] == "phish").astype(int)
    smish_df.to_csv(curated_smish_path, index=False, encoding="utf-8")

    return uci_df, smish_df


class MarkovClassGenerator:
    """Class-conditional word/clause Bigram-Trigram Markov Language Model trained strictly on X_train."""

    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.transitions: Dict[int, Dict[Tuple[str, str], List[str]]] = {
            0: defaultdict(list),
            1: defaultdict(list),
        }
        self.starters: Dict[int, List[Tuple[str, str]]] = {0: [], 1: []}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.strip().split()

    def fit(self, texts: List[str], labels: List[int]) -> "MarkovClassGenerator":
        for text, y in zip(texts, labels):
            tokens = self._tokenize(text)
            if len(tokens) < 4:
                continue
            bos1, bos2 = "<BOS1>", "<BOS2>"
            seq = [bos1, bos2] + tokens + ["<EOS>"]
            self.starters[int(y)].append((seq[2], seq[3]))
            for i in range(len(seq) - 2):
                state = (seq[i], seq[i + 1])
                nxt = seq[i + 2]
                self.transitions[int(y)][state].append(nxt)
        return self

    def generate(self, y: int, rng: random.Random, max_tokens: int = 28) -> str:
        trans = self.transitions[int(y)]
        if not trans:
            return ""
        state = ("<BOS1>", "<BOS2>")
        out: List[str] = []
        for _ in range(max_tokens):
            candidates = trans.get(state)
            if not candidates:
                break
            nxt = rng.choice(candidates)
            if nxt == "<EOS>":
                break
            out.append(nxt)
            state = (state[1], nxt)
        return " ".join(out)


class PhishAugmenter:
    """Multi-Strategy Generative & Adversarial Data Augmentation Model for Phishing Detection.

    Strictly fits on (X_train, y_train) so zero information leaks from the test set.
    """

    def __init__(
        self,
        target_phish_ratio: float = 0.35,
        augment_factor: float = 0.65,
        include_adversarial: bool = True,
        include_external_real: bool = False,
        seed: int = SEED,
    ):
        self.target_phish_ratio = target_phish_ratio
        self.augment_factor = augment_factor
        self.include_adversarial = include_adversarial
        self.include_external_real = include_external_real
        self.seed = seed
        self.markov = MarkovClassGenerator(seed=seed)
        self.hard_phish_texts_: List[str] = []
        self.hard_legit_texts_: List[str] = []
        self.phish_urls_: List[str] = []
        self.legit_urls_: List[str] = []
        self.stats_: Dict[str, int] = {}

    @staticmethod
    def _extract_urls(texts: List[str]) -> List[str]:
        url_re = re.compile(r"(?:https?://|www\.)\S+", re.I)
        found = []
        for t in texts:
            found.extend(url_re.findall(t))
        return found

    def contextual_paraphrase(self, text: str, rng: random.Random, replace_prob: float = 0.45) -> str:
        """Replace domain words with contextually equivalent synonyms while preserving syntax & case."""
        tokens = re.findall(r"\w+|\W+", text)
        out = []
        replaced = 0
        for tok in tokens:
            low = tok.lower()
            if low in CONTEXTUAL_SYNONYMS and rng.random() < replace_prob:
                syn = rng.choice(CONTEXTUAL_SYNONYMS[low])
                if tok[0].isupper():
                    syn = syn.capitalize()
                out.append(syn)
                replaced += 1
            else:
                out.append(tok)
        result = "".join(out)
        return result if replaced > 0 else text

    def adversarial_obfuscate(self, text: str, rng: random.Random, word_prob: float = 0.45) -> str:
        """Apply realistic phishing obfuscation (homoglyphs/leetspeak, dropped letters) to target words."""
        tokens = re.findall(r"\w+|\W+", text)
        out = []
        for tok in tokens:
            low = tok.lower()
            if low in OBFUSCATION_TARGET_WORDS and len(tok) >= 4 and rng.random() < word_prob:
                mode = rng.choice(["leet", "drop", "swap"])
                chars = list(tok)
                if mode == "leet":
                    replaceable = [i for i, c in enumerate(chars) if c.lower() in ADVERSARIAL_CHAR_MAP]
                    if replaceable:
                        idx = rng.choice(replaceable)
                        chars[idx] = ADVERSARIAL_CHAR_MAP[chars[idx].lower()]
                elif mode == "drop" and len(chars) > 5:
                    idx = rng.randrange(1, len(chars) - 1)
                    chars.pop(idx)
                elif mode == "swap" and len(chars) > 5:
                    idx = rng.randrange(1, len(chars) - 2)
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                out.append("".join(chars))
            else:
                out.append(tok)
        return "".join(out)

    def fit(self, X_train: pd.Series | List[str], y_train: pd.Series | np.ndarray) -> "PhishAugmenter":
        """Fit the Markov generator and out-of-fold Hard-Example Miner on the training set ONLY."""
        texts = list(pd.Series(X_train).astype(str))
        labels = np.asarray(y_train, dtype=int)

        # 1. Fit class-conditional Markov Language Model
        self.markov.fit(texts, list(labels))

        # 2. Extract URL pools per class from training set only
        phish_texts = [t for t, y in zip(texts, labels) if y == 1]
        legit_texts = [t for t, y in zip(texts, labels) if y == 0]
        self.phish_urls_ = self._extract_urls(phish_texts)
        self.legit_urls_ = self._extract_urls(legit_texts)

        # 3. Mine hard training examples via 5-fold Stratified Cross-Validation on X_train
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_vec = vec.fit_transform(texts)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_probs = np.zeros(len(texts), dtype=float)
        for tr_idx, val_idx in skf.split(X_vec, labels):
            clf = LogisticRegression(max_iter=1000, C=3.0, random_state=self.seed)
            clf.fit(X_vec[tr_idx], labels[tr_idx])
            oof_probs[val_idx] = clf.predict_proba(X_vec[val_idx])[:, 1]

        # Hard phishing examples: true phish in X_train where out-of-fold score < 0.68
        self.hard_phish_texts_ = [
            texts[i] for i in range(len(texts)) if labels[i] == 1 and oof_probs[i] < 0.68
        ]
        # Hard legitimate examples: true legit in X_train where out-of-fold score > 0.32
        self.hard_legit_texts_ = [
            texts[i] for i in range(len(texts)) if labels[i] == 0 and oof_probs[i] > 0.32
        ]
        if not self.hard_phish_texts_:
            self.hard_phish_texts_ = phish_texts
        if not self.hard_legit_texts_:
            self.hard_legit_texts_ = legit_texts
        return self

    def fit_resample(
        self,
        X_train: pd.Series | List[str],
        y_train: pd.Series | np.ndarray,
        real_smish_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Fit on (X_train, y_train) and generate an augmented training dataset."""
        self.fit(X_train, y_train)
        rng = random.Random(self.seed)

        orig_texts = list(pd.Series(X_train).astype(str))
        orig_labels = list(np.asarray(y_train, dtype=int))
        phish_pool = [t for t, y in zip(orig_texts, orig_labels) if y == 1]
        legit_pool = [t for t, y in zip(orig_texts, orig_labels) if y == 0]

        records = [
            {"text": t, "y": int(y), "label": "phish" if y == 1 else "legit", "aug_strategy": "original_train"}
            for t, y in zip(orig_texts, orig_labels)
        ]

        n_orig = len(orig_texts)
        n_target_new = int(n_orig * self.augment_factor)

        # Allocate augmentation budget across strategies
        n_hard_phish = int(n_target_new * 0.32)
        n_markov_phish = int(n_target_new * 0.18)
        n_adv_phish = int(n_target_new * 0.20) if self.include_adversarial else 0
        n_hard_legit = int(n_target_new * 0.20)
        n_markov_legit = n_target_new - (n_hard_phish + n_markov_phish + n_adv_phish + n_hard_legit)

        # 1. Hard-Example & Contextual Paraphrase Augmentation (Phish)
        for _ in range(n_hard_phish):
            base = rng.choice(self.hard_phish_texts_ if rng.random() < 0.75 else phish_pool)
            aug = self.contextual_paraphrase(base, rng, replace_prob=0.55)
            records.append({"text": aug, "y": 1, "label": "phish", "aug_strategy": "hard_boundary_paraphrase"})

        # 2. Markov Language Model Synthesis (Phish)
        for _ in range(n_markov_phish):
            gen = self.markov.generate(1, rng)
            if len(gen.split()) >= 5:
                records.append({"text": gen, "y": 1, "label": "phish", "aug_strategy": "markov_lm_synthesis"})

        # 3. Adversarial Obfuscation & Evasion Augmentation (Phish)
        for _ in range(n_adv_phish):
            base = rng.choice(phish_pool)
            aug = self.adversarial_obfuscate(base, rng, word_prob=0.50)
            records.append({"text": aug, "y": 1, "label": "phish", "aug_strategy": "adversarial_obfuscation"})

        # 4. Hard-Example & Contextual Paraphrase Augmentation (Legit)
        for _ in range(n_hard_legit):
            base = rng.choice(self.hard_legit_texts_ if rng.random() < 0.75 else legit_pool)
            aug = self.contextual_paraphrase(base, rng, replace_prob=0.50)
            records.append({"text": aug, "y": 0, "label": "legit", "aug_strategy": "hard_boundary_paraphrase"})

        # 5. Markov Language Model Synthesis (Legit)
        for _ in range(n_markov_legit):
            gen = self.markov.generate(0, rng)
            if len(gen.split()) >= 4:
                records.append({"text": gen, "y": 0, "label": "legit", "aug_strategy": "markov_lm_synthesis"})

        # 6. Optional External Real Smishing Corpus Injection
        if self.include_external_real and real_smish_df is not None and len(real_smish_df) > 0:
            sampled_real = real_smish_df.sample(
                n=min(350, len(real_smish_df)), random_state=self.seed
            )
            for _, row in sampled_real.iterrows():
                records.append({
                    "text": str(row["text"]),
                    "y": int(row["y"]),
                    "label": str(row["label"]),
                    "aug_strategy": "external_uci_smishing",
                })

        aug_df = pd.DataFrame(records)
        # Shuffle deterministically
        aug_df = aug_df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        self.stats_ = aug_df["aug_strategy"].value_counts().to_dict()
        return aug_df["text"], aug_df["y"], aug_df


def build_adversarial_test_set(X_test: pd.Series, y_test: pd.Series, seed: int = SEED) -> pd.DataFrame:
    """Create an obfuscated test benchmark from X_test to evaluate resilience to 'paypa1' / typo evasion."""
    rng = random.Random(seed)
    augmenter = PhishAugmenter(seed=seed)
    texts = list(pd.Series(X_test).astype(str))
    labels = list(np.asarray(y_test, dtype=int))
    adv_texts = []
    for t, y in zip(texts, labels):
        if y == 1:
            # Obfuscate phishing messages to simulate an active adversary
            adv_texts.append(augmenter.adversarial_obfuscate(t, rng, word_prob=0.65))
        else:
            adv_texts.append(t)
    return pd.DataFrame({"text": adv_texts, "y": labels, "label": ["phish" if y == 1 else "legit" for y in labels]})
