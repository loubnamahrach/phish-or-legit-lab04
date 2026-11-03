"""Lab 04 - evaluation: precision / recall / F1 of the PHISH class, confusion matrix, calibration,
and Data Augmentation / External Real Dataset benchmarks.

Run from the lab folder:  python -m src.evaluate
"""
import json
import os

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_fscore_support, precision_recall_curve

from src.augmentation import PhishAugmenter, build_adversarial_test_set, download_uci_dataset
from src.pipeline import contributions, fit_augmented_pipeline, global_top_features, load_data, make_pipeline, split


def ece(y, p, bins=10):
    """Expected calibration error: weighted mean |observed frequency - mean predicted score| per bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def eval_predictions(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    acc = float((pred == y_true).mean())
    return {
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "accuracy": float(acc),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": float(ece(y_true, probs)),
    }


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    yte_arr = np.asarray(yte)
    log(f"train = {len(Xtr)} messages ({ytr.mean():.1%} phish) | test = {len(Xte)} messages "
        f"({yte_arr.mean():.1%} phish)")
    log(f"Majority-class baseline accuracy (always 'legit'): {(1 - yte_arr.mean()):.1%}  <- why accuracy alone is misleading")

    # 1. Standard Baseline Models (fitted on original 1,500 train split)
    models = {
        "Naive Bayes": make_pipeline("nb"),
        "Logistic Regression": make_pipeline("lr"),
        "NB + calibration": make_pipeline("nb_cal"),
        "LR + char n-grams (extension)": make_pipeline("lr", analyzer="char_wb"),
    }

    results, probs = {}, {}
    log("\n=== 1. Baseline Evaluation on Standard Test Split (500 messages, threshold=0.5) ===")
    log("model                                   precision  recall    F1   accuracy  Brier   ECE")
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        results[name] = eval_predictions(yte_arr, p, threshold=0.5)
        probs[name] = p
        r = results[name]
        log(f"{name:<40}{r['precision']:>8.3f}{r['recall']:>9.3f}{r['f1']:>7.3f}{r['accuracy']:>9.3f}{r['brier']:>8.3f}{r['ece']:>7.3f}")

    # 2. Data Augmentation Pipeline (fitted strictly on Xtr, ytr -> zero leakage into Xte)
    uci_df, smish_df = download_uci_dataset()
    augmenter = PhishAugmenter(
        target_phish_ratio=0.35,
        augment_factor=0.65,
        include_adversarial=True,
        include_external_real=False,
    )
    Xtr_aug, ytr_aug, aug_df = augmenter.fit_resample(Xtr, ytr)
    aug_df.to_csv("data/messages_train_augmented.csv", index=False, encoding="utf-8")

    aug_models = {
        "LR + Data Aug (Word+Char)": make_pipeline("lr", analyzer="hybrid", url_aware=False),
        "LR + Data Aug + URL-Aware": make_pipeline("lr", analyzer="hybrid", url_aware=True),
    }
    log(f"\n=== 2. Data Augmentation Evaluation (Train expanded {len(Xtr)} -> {len(Xtr_aug)} msgs, tested on same 500 holdout) ===")
    log("model                                   precision  recall    F1   accuracy  Brier   ECE")
    for name, pipe in aug_models.items():
        fit_augmented_pipeline(pipe, Xtr_aug, ytr_aug, base_prior=float(ytr.mean()), prior_weight=0.70)
        models[name] = pipe
        p = pipe.predict_proba(Xte)[:, 1]
        results[name] = eval_predictions(yte_arr, p, threshold=0.5)
        probs[name] = p
        r = results[name]
        log(f"{name:<40}{r['precision']:>8.3f}{r['recall']:>9.3f}{r['f1']:>7.3f}{r['accuracy']:>9.3f}{r['brier']:>8.3f}{r['ece']:>7.3f}")

    # 3. Adversarial Obfuscation Benchmark ('paypa1', 'n0rthbank', 'v3rify', character drops)
    adv_test_df = build_adversarial_test_set(Xte, yte)
    Xte_adv = adv_test_df["text"]
    adv_results = {}
    log("\n=== 3. Adversarial Robustness Benchmark (Obfuscated Phishing Test Set) ===")
    log("model                                   precision  recall    F1   accuracy  Brier")
    for name in ("Logistic Regression", "LR + char n-grams (extension)", "LR + Data Aug (Word+Char)", "LR + Data Aug + URL-Aware"):
        p_adv = models[name].predict_proba(Xte_adv)[:, 1]
        r_adv = eval_predictions(yte_arr, p_adv, threshold=0.5)
        adv_results[name] = r_adv
        log(f"{name:<40}{r_adv['precision']:>8.3f}{r_adv['recall']:>9.3f}{r_adv['f1']:>7.3f}{r_adv['accuracy']:>9.3f}{r_adv['brier']:>8.3f}")

    # 4. External Real Dataset Benchmark (UCI SMS Spam Collection - 5,572 messages)
    Xtr_uci, Xte_uci, ytr_uci, yte_uci = split(uci_df)
    yte_uci_arr = np.asarray(yte_uci)
    uci_results = {}
    log(f"\n=== 4. External Real Dataset Benchmark: UCI SMS Spam ({len(uci_df)} real messages) ===")
    log("model                                   precision  recall    F1   accuracy  Brier")
    for name, kind_cfg in (
        ("UCI Naive Bayes", ("nb", "word")),
        ("UCI Logistic Regression", ("lr", "word")),
        ("UCI LR + char_wb + Data Aug", ("lr", "hybrid")),
    ):
        pipe_u = make_pipeline(kind_cfg[0], analyzer=kind_cfg[1])
        if "Data Aug" in name:
            aug_u = PhishAugmenter(augment_factor=0.45, include_adversarial=True)
            Xtr_u_fit, ytr_u_fit, _ = aug_u.fit_resample(Xtr_uci, ytr_uci)
            pipe_u.fit(Xtr_u_fit, ytr_u_fit)
        else:
            pipe_u.fit(Xtr_uci, ytr_uci)
        p_u = pipe_u.predict_proba(Xte_uci)[:, 1]
        r_u = eval_predictions(yte_uci_arr, p_u, threshold=0.5)
        uci_results[name] = r_u
        log(f"{name:<40}{r_u['precision']:>8.3f}{r_u['recall']:>9.3f}{r_u['f1']:>7.3f}{r_u['accuracy']:>9.3f}{r_u['brier']:>8.3f}")

    lr = models["Logistic Regression"]
    p_lr = probs["Logistic Regression"]
    cm = confusion_matrix(yte_arr, (p_lr >= 0.5).astype(int))
    log("\nConfusion matrix (Logistic Regression)   rows = truth, columns = prediction (legit, phish)")
    log(str(cm))

    log("\nDecision threshold trade-off (Logistic Regression):")
    threshold_table = {}
    for th in (0.2, 0.35, 0.5, 0.65, 0.7, 0.9):
        pred = (p_lr >= th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yte_arr, pred, average="binary", zero_division=0)
        threshold_table[str(th)] = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}
        log(f"  threshold {th:<4}: precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}")

    pos, neg = global_top_features(lr)
    log(f"\nStrongest 'phish' features: {pos[:6]}")
    log(f"Strongest 'legit' features: {neg[:6]}")

    # --- failure cases + per-feature explanations
    texts = np.asarray(Xte)
    fn = np.where((yte_arr == 1) & (p_lr < 0.5))[0]
    fp = np.where((yte_arr == 0) & (p_lr >= 0.5))[0]
    lines = [f"False negatives (phish predicted legit): {len(fn)}"]
    for i in fn[:4]:
        top_c = list(contributions(lr, texts[i]).items())[:5]
        c_str = ", ".join(f"{k}:{v:+.2f}" for k, v in top_c)
        lines.append(f"  score={p_lr[i]:.2f} | {texts[i]}  [features: {c_str}]")
    lines.append(f"False positives (legit predicted phish): {len(fp)}")
    for i in fp[:4]:
        top_c = list(contributions(lr, texts[i]).items())[:5]
        c_str = ", ".join(f"{k}:{v:+.2f}" for k, v in top_c)
        lines.append(f"  score={p_lr[i]:.2f} | {texts[i]}  [features: {c_str}]")

    os.makedirs("results", exist_ok=True)
    with open("results/failures.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("\n" + "\n".join(lines))

    with open("results/run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    metrics_payload = {
        "models": results,
        "adversarial_benchmark": adv_results,
        "uci_real_dataset_benchmark": uci_results,
        "threshold_tradeoff_lr": threshold_table,
        "augmentation_breakdown": augmenter.stats_,
        "confusion_matrix_lr": cm.tolist(),
        "confusion_matrix_aug": confusion_matrix(
            yte_arr, (probs["LR + Data Aug + URL-Aware"] >= 0.5).astype(int)
        ).tolist(),
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Standard Lab 04 Confusion Matrix + Reliability Diagram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["pred legit", "pred phish"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["true legit", "true phish"])
    axes[0].set_title("Confusion matrix (Logistic Regression)")

    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for (name, c) in (
        ("Naive Bayes", "#d95f02"),
        ("Logistic Regression", "#1b9e77"),
        ("NB + calibration", "#7570b3"),
        ("LR + Data Aug + URL-Aware", "#e7298a"),
    ):
        frac, mean = calibration_curve(yte_arr, probs[name], n_bins=8, strategy="quantile")
        axes[1].plot(mean, frac, marker="o", color=c, label=name)
    axes[1].set_xlabel("mean predicted score"); axes[1].set_ylabel("observed share of phish")
    axes[1].set_title("Reliability diagram: is a score a probability?"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/phish_results.png", dpi=160)
    plt.close(fig)

    # Plot 2: Precision/Recall vs Decision Threshold & Data Augmentation Comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ths = np.linspace(0.05, 0.95, 91)
    prs, rcs, f1s = [], [], []
    for t in ths:
        pred_t = (p_lr >= t).astype(int)
        pr_t, rc_t, f1_t, _ = precision_recall_fscore_support(yte_arr, pred_t, average="binary", zero_division=0)
        prs.append(pr_t); rcs.append(rc_t); f1s.append(f1_t)
    axes2[0].plot(ths, prs, label="Precision", color="#1f77b4", lw=2)
    axes2[0].plot(ths, rcs, label="Recall", color="#ff7f0e", lw=2)
    axes2[0].plot(ths, f1s, label="F1-score", color="#2ca02c", lw=2, linestyle="--")
    axes2[0].axvline(0.65, color="#d62728", linestyle=":", lw=1.8, label="Selected threshold (0.65)")
    axes2[0].set_xlabel("Decision Threshold")
    axes2[0].set_ylabel("Metric Value")
    axes2[0].set_title("Precision / Recall Trade-off vs. Decision Threshold (LR)")
    axes2[0].legend(fontsize=8)
    axes2[0].grid(alpha=0.25)

    model_labels = ["Baseline LR", "LR + char_wb", "LR + Data Aug\n(Word+Char)", "LR + Data Aug\n+ URL-Aware"]
    model_keys = ["Logistic Regression", "LR + char n-grams (extension)", "LR + Data Aug (Word+Char)", "LR + Data Aug + URL-Aware"]
    std_f1 = [results[k]["f1"] for k in model_keys]
    adv_f1 = [adv_results[k]["f1"] for k in model_keys]
    x = np.arange(len(model_labels))
    w = 0.35
    axes2[1].bar(x - w / 2, std_f1, width=w, label="Standard Test Set F1", color="#4c72b0")
    axes2[1].bar(x + w / 2, adv_f1, width=w, label="Adversarial Obfuscated F1", color="#dd8452")
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(model_labels, fontsize=8.5)
    axes2[1].set_ylim(0, 1.0)
    axes2[1].set_ylabel("F1 Score (Phish Class)")
    axes2[1].set_title("Impact of Data Augmentation on Standard & Adversarial F1")
    axes2[1].legend(fontsize=8)
    axes2[1].grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig("results/augmentation_comparison.png", dpi=160)
    plt.close(fig2)

    log("\nSaved results/metrics.json, results/failures.txt, results/phish_results.png, results/augmentation_comparison.png")


if __name__ == "__main__":
    main()
