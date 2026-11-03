"""Lab 04 - Data Augmentation & Real Dataset Downloader CLI.

Run from the lab folder (`04_phish_or_legit`):
    python -m data.augment_data

What it does:
1. Downloads the real UCI SMS Spam Collection (5,572 messages) -> data/uci_sms_spam.csv
2. Extracts a curated Real Smishing dataset (20% phish-like SMS) -> data/real_smishing_curated.csv
3. Splits data/messages.csv (1,500 train / 500 test, stratified, seed=42) BEFORE any augmentation
   (strictly preventing data leakage into the 500 test messages).
4. Fits the PhishAugmenter generative & adversarial model on the 1,500 training messages and exports
   data/messages_train_augmented.csv.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.augmentation import PhishAugmenter, download_uci_dataset
from src.pipeline import load_data, split


def main():
    print("=== Step 1: Downloading Real UCI SMS Spam Dataset & Extracting Smishing Corpus ===")
    uci_df, smish_df = download_uci_dataset(
        uci_csv_path="data/uci_sms_spam.csv",
        curated_smish_path="data/real_smishing_curated.csv",
    )
    print(f"  -> Saved data/uci_sms_spam.csv: {len(uci_df)} real SMS messages "
          f"({uci_df['y'].sum()} spam / {(1 - uci_df['y']).sum()} ham, {uci_df['y'].mean():.1%} positive)")
    print(f"  -> Saved data/real_smishing_curated.csv: {len(smish_df)} domain-adapted real messages "
          f"({smish_df['y'].sum()} smishing / {(1 - smish_df['y']).sum()} legit, {smish_df['y'].mean():.1%} positive)")

    print("\n=== Step 2: Training PhishAugmenter on Stratified Train Split Only (Zero Leakage) ===")
    df = load_data("data/messages.csv")
    Xtr, Xte, ytr, yte = split(df)
    print(f"  Original Train Split: {len(Xtr)} messages ({ytr.sum()} phish, {ytr.mean():.1%})")
    print(f"  Untouched Holdout Test Split: {len(Xte)} messages ({yte.sum()} phish, {yte.mean():.1%})")

    augmenter = PhishAugmenter(
        target_phish_ratio=0.35,
        augment_factor=0.65,
        include_adversarial=True,
        include_external_real=True,
    )
    Xtr_aug, ytr_aug, aug_df = augmenter.fit_resample(Xtr, ytr, real_smish_df=smish_df)
    aug_df.to_csv("data/messages_train_augmented.csv", index=False, encoding="utf-8")

    print(f"\n  -> Saved data/messages_train_augmented.csv: {len(aug_df)} total training messages "
          f"({aug_df['y'].sum()} phish, {aug_df['y'].mean():.1%} positive)")
    print("  Breakdown by augmentation strategy:")
    for strat, count in augmenter.stats_.items():
        print(f"    - {strat:<28}: {count:>4} messages")


if __name__ == "__main__":
    main()
