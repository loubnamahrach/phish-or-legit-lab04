"""Lab 04 - Phish or Legit? (Full Streamlit Web Application + Data Augmentation & Extensions).

Run from the `04_phish_or_legit` folder:
    streamlit run app.py
"""
from datetime import datetime
import io
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_fscore_support
import streamlit as st

from src.augmentation import PhishAugmenter, build_adversarial_test_set, download_uci_dataset
from src.evaluate import ece
from src.pipeline import (
    contributions,
    fit_augmented_pipeline,
    global_top_features,
    load_data,
    make_pipeline,
    predict,
    preprocess,
    split,
    word_weights,
)

st.set_page_config(
    page_title="Phish or Legit? — Lab 04 NLP",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header & Mandatory Visible Notice (Deliverable 4.A.6)
# ---------------------------------------------------------------------------
st.title("🛡️ Phish or Legit? — Calibrated Text Classification & Explainability")
st.info("ℹ️ **This tool supports a human decision and does not block messages by itself. The data is synthetic.**")

# ---------------------------------------------------------------------------
# Preset Example Messages (Deliverable 4.A.5: >= 4 examples, 2 phish, 2 legit, including misclassified)
# ---------------------------------------------------------------------------
EXAMPLES = {
    "Custom (type or paste below)": None,
    "🔴 Phish #1 (Clear Account Lock + Short URL — True Positive)": {
        "text": "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd",
        "true_label": "phish",
        "note": "Classic phishing message with brand urgency ('locked', 'Verify now') and a shortened URL.",
    },
    "🔴 Phish #2 (Delivery Fee Request — True Positive)": {
        "text": "ParcelGo could not deliver your package today. Pay the fee now. Please reply as soon as you can. https://meet.example.org/xcu2o5zf",
        "true_label": "phish",
        "note": "Delivery scam requesting immediate fee payment with an external link.",
    },
    "🟢 Legit #1 (Class Reminder — True Negative)": {
        "text": "Hi, Reminder: class starts 8am. Details are in your online account.",
        "true_label": "legit",
        "note": "Standard legitimate reminder using strong legit indicators ('Hi', 'Reminder', time).",
    },
    "🟢 Legit #2 (Order Confirmation — True Negative)": {
        "text": "Reminder: Thanks for your order #36616. https://docs.example.com/d/0c2fjt",
        "true_label": "legit",
        "note": "Legitimate order notification with order number and trusted document link.",
    },
    "⚠️ Misclassified #1 — False Negative (Stealth Phish missed by Baseline LR @ 0.50)": {
        "text": "Dear customer, A payment of $7.67 is pending. You can manage this in the app settings. https://vg9ys-login.info/aqjv",
        "true_label": "phish",
        "note": "True label is PHISH, but Baseline LR scores it ~0.13 (Legit) because 'You can manage this in the app settings' pushes strongly toward legit (-2.4) while baseline preprocessing collapses the suspicious '-login.info' URL into a generic 'urltok'. Switch to 'LR + Data Aug + URL-Aware' to see it caught!",
    },
    "⚠️ Misclassified #2 — False Positive (Urgent Legit flagged by Baseline LR @ 0.50)": {
        "text": "SwiftPost could not deliver your package today. This is your final notice. https://www.example.com/orders/8777",
        "true_label": "legit",
        "note": "True label is LEGIT, but Baseline LR scores it ~0.55 (Phish) because 'final notice', 'package', and 'urltok' outweigh the legitimate domain 'www.example.com/orders/'. Raising the decision threshold slider to 0.65 (our high-precision policy) or using the Data-Augmented URL-Aware model resolves this false alarm!",
    },
    "🕵️ Adversarial Attack (Obfuscated Leetspeak 'N0rthbank' / 'V3rify' — Extension)": {
        "text": "ALERT: Your N0rthbank acc0unt is l0cked. V3rify your 1dentity immed1ately http://bit.ly/x9z2q1",
        "true_label": "phish",
        "note": "Adversarial evasion using character substitutions ('0' for 'o', '3' for 'e', '1' for 'i'). Compare 'Logistic Regression (Baseline)' vs 'LR + Data Aug (Word+Char)'!",
    },
}


# ---------------------------------------------------------------------------
# Cached Model & Data Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting TF-IDF + Classification & Data Augmentation pipelines on training split...")
def load_all_pipelines():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "messages.csv")
    uci_path = os.path.join(base_dir, "data", "uci_sms_spam.csv")
    smish_path = os.path.join(base_dir, "data", "real_smishing_curated.csv")

    df = load_data(data_path)
    Xtr, Xte, ytr, yte = split(df)
    yte_arr = np.asarray(yte)

    # Fit Data Augmentation model strictly on (Xtr, ytr) -> Zero data leakage
    uci_df, smish_df = download_uci_dataset(uci_csv_path=uci_path, curated_smish_path=smish_path)
    augmenter = PhishAugmenter(
        target_phish_ratio=0.35,
        augment_factor=0.65,
        include_adversarial=True,
        include_external_real=False,
    )
    Xtr_aug, ytr_aug, aug_df = augmenter.fit_resample(Xtr, ytr)

    pipelines = {
        "Logistic Regression (Baseline)": make_pipeline("lr", analyzer="word").fit(Xtr, ytr),
        "Naive Bayes (Baseline)": make_pipeline("nb", analyzer="word").fit(Xtr, ytr),
        "Naive Bayes + Calibration (Platt)": make_pipeline("nb_cal", analyzer="word").fit(Xtr, ytr),
        "LR + Char N-grams (char_wb 2-5)": make_pipeline("lr", analyzer="char_wb").fit(Xtr, ytr),
        "LR + Data Augmentation (Word+Char)": fit_augmented_pipeline(
            make_pipeline("lr", analyzer="hybrid"), Xtr_aug, ytr_aug, base_prior=float(ytr.mean()), prior_weight=0.70
        ),
        "LR + Data Aug + URL-Aware (Enhanced)": fit_augmented_pipeline(
            make_pipeline("lr", analyzer="hybrid", url_aware=True), Xtr_aug, ytr_aug, base_prior=float(ytr.mean()), prior_weight=0.70
        ),
    }
    cal_lr = make_pipeline("lr_cal", analyzer="word").fit(Xtr, ytr)

    # Precompute test-set probabilities for every model
    test_probs = {name: pipe.predict_proba(Xte)[:, 1] for name, pipe in pipelines.items()}
    adv_test_df = build_adversarial_test_set(Xte, yte)
    adv_probs = {name: pipe.predict_proba(adv_test_df["text"])[:, 1] for name, pipe in pipelines.items()}

    return {
        "df": df,
        "Xtr": Xtr,
        "Xte": Xte,
        "ytr": ytr,
        "yte": yte_arr,
        "pipelines": pipelines,
        "cal_lr": cal_lr,
        "test_probs": test_probs,
        "adv_probs": adv_probs,
        "augmenter": augmenter,
        "aug_df": aug_df,
        "uci_df": uci_df,
        "smish_df": smish_df,
    }


bundle = load_all_pipelines()
pipelines = bundle["pipelines"]
yte_arr = bundle["yte"]
test_probs = bundle["test_probs"]

# ---------------------------------------------------------------------------
# Sidebar Controls: Model Selector & Live Decision Threshold Slider
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Classifier & Operating Point")
    selected_model_name = st.selectbox(
        "Select Classifier Pipeline:",
        list(pipelines.keys()),
        index=0,
        help="Compare baseline Generative (Naive Bayes) and Discriminative (Logistic Regression) models against Calibrated, Character N-gram, and Data-Augmented models.",
    )
    active_model = pipelines[selected_model_name]

    st.markdown("---")
    st.subheader("🎚️ Live Decision Threshold")
    threshold = st.slider(
        "Phishing Decision Threshold (τ):",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Predict PHISHING when score >= τ. Raising τ increases Precision (fewer false alarms on legit messages) at the cost of lower Recall.",
    )

    # Show live test-set metrics at current threshold
    p_active = test_probs[selected_model_name]
    pred_active = (p_active >= threshold).astype(int)
    pr_now, rc_now, f1_now, _ = precision_recall_fscore_support(
        yte_arr, pred_active, average="binary", zero_division=0
    )
    brier_now = brier_score_loss(yte_arr, p_active)
    ece_now = ece(yte_arr, p_active)

    st.caption(f"**Test Set Performance @ τ = {threshold:.2f}** (500 holdout msgs):")
    c1, c2 = st.columns(2)
    c1.metric("Precision", f"{pr_now:.3f}")
    c2.metric("Recall", f"{rc_now:.3f}")
    c3, c4 = st.columns(2)
    c3.metric("F1-Score", f"{f1_now:.3f}")
    c4.metric("Brier / ECE", f"{brier_now:.3f} / {ece_now:.3f}")

    st.markdown("---")
    st.markdown(
        "**Recommended Policy (τ = 0.65):**\n"
        "> *\"We prioritize high precision (~0.87) over recall (~0.56): we accept missing some phishing messages rather than blocking or alarming users on legitimate messages.\"*"
    )

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_classify, tab_eval, tab_aug, tab_batch = st.tabs([
    "🔍 1. Classify & Explain Message",
    "📊 2. Threshold Policy, Calibration & Metrics",
    "🚀 3. Data Augmentation & Real UCI Dataset",
    "📁 4. Batch CSV Scanner & Feedback Log",
])

# ===========================================================================
# TAB 1: Interactive Classifier, Live Threshold & Per-Feature Explanation
# ===========================================================================
with tab_classify:
    col_input, col_result = st.columns([1.15, 1.0], gap="large")

    with col_input:
        st.subheader("1. Select an Example or Paste a Message")
        example_choice = st.selectbox(
            "Choose a preset example message (includes 2 Phishing, 2 Legitimate, 2 Misclassified, 1 Adversarial):",
            list(EXAMPLES.keys()),
            index=1,
        )

        if "last_example" not in st.session_state:
            st.session_state.last_example = example_choice
        if "message_text" not in st.session_state:
            st.session_state.message_text = EXAMPLES[example_choice]["text"]

        if example_choice != st.session_state.last_example:
            st.session_state.last_example = example_choice
            if EXAMPLES[example_choice] is not None:
                st.session_state.message_text = EXAMPLES[example_choice]["text"]

        if EXAMPLES[example_choice] is not None:
            ex_meta = EXAMPLES[example_choice]
            badge = "🔴 Ground Truth: **PHISH**" if ex_meta["true_label"] == "phish" else "🟢 Ground Truth: **LEGIT**"
            st.caption(f"{badge} — *{ex_meta['note']}*")

        text = st.text_area(
            "Message to classify:",
            key="message_text",
            height=115,
            placeholder="Paste an SMS or email message here...",
        )

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            classify_clicked = st.button("🔍 Classify Message", type="primary", width="stretch")
        with btn_col2:
            st.caption(f"Active model: **{selected_model_name}** | Threshold: **τ = {threshold:.2f}**")

    with col_result:
        st.subheader("2. Prediction & Calibrated Score (Live)")
        if not text or not text.strip():
            st.warning("The message is empty — please select an example or paste some text to classify.")
        else:
            label, score = predict(active_model, text, threshold=threshold)
            _, cal_score = predict(bundle["cal_lr"], text, threshold=threshold)

            if label == "phish":
                st.error(
                    f"⚠️ **Predicted Class: PHISHING (`phish`)**  ·  "
                    f"Classifier Score: **{score:.3f}** (≥ threshold {threshold:.2f})"
                )
            else:
                st.success(
                    f"✅ **Predicted Class: LEGITIMATE (`legit`)**  ·  "
                    f"Classifier Score: **{score:.3f}** (< threshold {threshold:.2f})"
                )

            st.progress(min(max(score, 0.0), 1.0), text=f"Phishing Score: {score:.1%}  |  Decision Threshold τ = {threshold:.0%}")
            st.caption(
                f"Platt-calibrated reference probability (5-fold sigmoid calibration on training data): **{cal_score:.3f}**. "
                "Remember: raw scores (especially from Naive Bayes) are only meaningful as probabilities if calibration is verified."
            )

            # Compare across all models at the current threshold
            with st.expander("Compare score across all 6 pipelines at current threshold"):
                comp_rows = []
                for m_name, m_pipe in pipelines.items():
                    m_lbl, m_sc = predict(m_pipe, text, threshold=threshold)
                    comp_rows.append({
                        "Pipeline": m_name,
                        "Score": round(m_sc, 3),
                        f"Prediction (@ τ={threshold:.2f})": "⚠️ PHISH" if m_lbl == "phish" else "✅ LEGIT",
                    })
                st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # Per-Feature Explanation (Deliverable 4.A.4)
    # -----------------------------------------------------------------------
    if text and text.strip():
        st.markdown("---")
        st.subheader("3. Per-Feature Explanation (Contributions to Log-Odds)")

        weights = word_weights(active_model, text)
        parts = []
        for tok in re.findall(r"\w+|\W+", text):
            w = weights.get(tok.lower(), 0.0)
            if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.25:
                parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
            else:
                parts.append(tok)

        st.markdown("**Highlighted Message Tokens:**  " + "".join(parts))
        st.caption(
            "🔴 **Red tokens** increased the phishing score (positive contribution to log-odds) · "
            "🟢 **Green tokens** decreased the phishing score / pushed toward legitimate (negative contribution) · "
            f"Normalized form seen by vectorizer: `{preprocess(text).strip()}`"
        )

        all_contrib = contributions(active_model, text)
        pos_contrib = [(f, v) for f, v in all_contrib.items() if v > 0]
        neg_contrib = [(f, v) for f, v in all_contrib.items() if v < 0]

        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1.2])
        with exp_col1:
            st.markdown("##### 🔺 Features Increasing Phish Score (+)")
            if pos_contrib:
                df_pos = pd.DataFrame(pos_contrib[:8], columns=["Feature (n-gram)", "Log-Odds Contribution (+)"])
                df_pos["Log-Odds Contribution (+)"] = df_pos["Log-Odds Contribution (+)"].map(lambda x: f"+{x:.3f}")
                st.table(df_pos)
            else:
                st.caption("No active features pushed the score toward phishing.")

        with exp_col2:
            st.markdown("##### 🔻 Features Decreasing Phish Score (−)")
            if neg_contrib:
                df_neg = pd.DataFrame(neg_contrib[:8], columns=["Feature (n-gram)", "Log-Odds Contribution (−)"])
                df_neg["Log-Odds Contribution (−)"] = df_neg["Log-Odds Contribution (−)"].map(lambda x: f"{x:.3f}")
                st.table(df_neg)
            else:
                st.caption("No active features pushed the score toward legitimate.")

        with exp_col3:
            st.markdown("##### 📊 Top Feature Contributions Chart")
            top_items = list(all_contrib.items())[:8]
            if top_items:
                fig_c, ax_c = plt.subplots(figsize=(4.8, 2.8))
                feats = [k for k, _ in top_items][::-1]
                vals = [v for _, v in top_items][::-1]
                colors = ["#d9534f" if v > 0 else "#2e8b57" for v in vals]
                ax_c.barh(feats, vals, color=colors)
                ax_c.axvline(0, color="black", lw=0.8)
                ax_c.set_xlabel("Contribution (TF-IDF × Weight)")
                ax_c.set_title("Per-Feature Log-Odds Impact", fontsize=10)
                fig_c.tight_layout()
                st.pyplot(fig_c)
                plt.close(fig_c)

        # Extension 5: Human feedback logger for misclassified messages
        with st.expander("🚩 Was this message misclassified? Log feedback for retraining (Extension)"):
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 1])
            with fb_col1:
                correct_label = st.selectbox("True / Corrected Label:", ["phish", "legit"], index=0 if label == "legit" else 1)
            with fb_col2:
                analyst_note = st.text_input("Analyst Note (optional):", placeholder="e.g., Legit delivery reminder flagged due to 'final notice'")
            with fb_col3:
                st.write("")
                if st.button("💾 Log Misclassified Message"):
                    fb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feedback_log.csv")
                    row_df = pd.DataFrame([{
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model": selected_model_name,
                        "threshold": round(threshold, 2),
                        "predicted_label": label,
                        "predicted_score": round(score, 4),
                        "corrected_label": correct_label,
                        "note": analyst_note,
                        "text": text,
                    }])
                    write_header = not os.path.exists(fb_path)
                    row_df.to_csv(fb_path, mode="a", header=write_header, index=False, encoding="utf-8")
                    st.success(f"Logged to `data/feedback_log.csv` (Predicted: {label} [{score:.2f}] → True: {correct_label}).")

# ===========================================================================
# TAB 2: Threshold Trade-off, Calibration & Results Table
# ===========================================================================
with tab_eval:
    st.subheader("📈 Results Table on Test Split (500 Holdout Messages, 20% Phish)")
    st.caption(
        "Majority-class classifier (always predicting 'legit') reaches **80.0% accuracy** and **0.000 recall** — "
        "which is why Precision, Recall, F1, Brier Score, and ECE are used instead of accuracy alone."
    )

    metric_rows = []
    for name, p_vec in test_probs.items():
        pred_v = (p_vec >= threshold).astype(int)
        pr_v, rc_v, f1_v, _ = precision_recall_fscore_support(yte_arr, pred_v, average="binary", zero_division=0)
        acc_v = float((pred_v == yte_arr).mean())
        metric_rows.append({
            "Classifier Pipeline": name,
            f"Precision (@ τ={threshold:.2f})": round(float(pr_v), 3),
            f"Recall (@ τ={threshold:.2f})": round(float(rc_v), 3),
            f"F1-Score (@ τ={threshold:.2f})": round(float(f1_v), 3),
            "Accuracy": round(acc_v, 3),
            "Brier Score (↓)": round(float(brier_score_loss(yte_arr, p_vec)), 3),
            "ECE (↓)": round(float(ece(yte_arr, p_vec)), 3),
        })
    st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.markdown(f"#### Precision / Recall / F1 vs. Threshold ({selected_model_name})")
        ths = np.linspace(0.05, 0.95, 91)
        prs, rcs, f1s = [], [], []
        for t in ths:
            pred_t = (p_active >= t).astype(int)
            pr_t, rc_t, f1_t, _ = precision_recall_fscore_support(yte_arr, pred_t, average="binary", zero_division=0)
            prs.append(pr_t); rcs.append(rc_t); f1s.append(f1_t)
        fig_pr, ax_pr = plt.subplots(figsize=(5.5, 3.6))
        ax_pr.plot(ths, prs, label="Precision", color="#1f77b4", lw=2)
        ax_pr.plot(ths, rcs, label="Recall", color="#ff7f0e", lw=2)
        ax_pr.plot(ths, f1s, label="F1-Score", color="#2ca02c", lw=2, linestyle="--")
        ax_pr.axvline(threshold, color="#d62728", linestyle=":", lw=2, label=f"Active Slider τ = {threshold:.2f}")
        ax_pr.set_xlabel("Decision Threshold (τ)")
        ax_pr.set_ylabel("Score")
        ax_pr.set_ylim(0, 1.05)
        ax_pr.grid(alpha=0.25)
        ax_pr.legend(fontsize=8.5)
        fig_pr.tight_layout()
        st.pyplot(fig_pr)
        plt.close(fig_pr)

    with plot_col2:
        st.markdown("#### Reliability Diagram (Calibration Curve) & Confusion Matrix")
        fig_cal, axes_cal = plt.subplots(1, 2, figsize=(6.2, 3.6))
        cm_live = confusion_matrix(yte_arr, pred_active)
        axes_cal[0].imshow(cm_live, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axes_cal[0].text(
                    j, i, int(cm_live[i, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm_live[i, j] > cm_live.max() / 2 else "black",
                )
        axes_cal[0].set_xticks([0, 1]); axes_cal[0].set_xticklabels(["Pred Legit", "Pred Phish"], fontsize=8)
        axes_cal[0].set_yticks([0, 1]); axes_cal[0].set_yticklabels(["True Legit", "True Phish"], fontsize=8)
        axes_cal[0].set_title(f"Confusion Matrix @ τ={threshold:.2f}", fontsize=9.5)

        axes_cal[1].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        for m_label, col in [
            ("Naive Bayes (Baseline)", "#d95f02"),
            ("Logistic Regression (Baseline)", "#1b9e77"),
            ("Naive Bayes + Calibration (Platt)", "#7570b3"),
            ("LR + Data Aug + URL-Aware (Enhanced)", "#e7298a"),
        ]:
            frac, mean_p = calibration_curve(yte_arr, test_probs[m_label], n_bins=8, strategy="quantile")
            axes_cal[1].plot(mean_p, frac, marker="o", markersize=4, color=col, label=m_label.split(" (")[0])
        axes_cal[1].set_xlabel("Mean Predicted Score", fontsize=8)
        axes_cal[1].set_ylabel("Observed Phish Fraction", fontsize=8)
        axes_cal[1].set_title("Reliability Diagram", fontsize=9.5)
        axes_cal[1].legend(fontsize=6.5)
        fig_cal.tight_layout()
        st.pyplot(fig_cal)
        plt.close(fig_cal)

# ===========================================================================
# TAB 3: Data Augmentation Engine & Real UCI Dataset Integration
# ===========================================================================
with tab_aug:
    st.subheader("🚀 Multi-Strategy NLP Data Augmentation & Real Dataset Integration")
    st.markdown(
        """
        To address the **20% minority class imbalance**, **lexical variability**, and **adversarial token obfuscation**
        without leaking any test data, our pipeline (`src/augmentation.py`) implements **two complementary approaches**:
        1. **Generative & Adversarial Data Augmentation Model (`PhishAugmenter`)** fitted *strictly on the 1,500 training split*:
           - **Markov N-Gram Language Model (`markov_lm_synthesis`)**: Learns class-conditional bigram/trigram transitions from `X_train` to synthesize novel phishing and legit messages.
           - **Out-of-Fold Hard-Example Boundary Augmentation (`hard_boundary_paraphrase`)**: Uses 5-fold cross-validation on `X_train` to locate low-confidence phishing & boundary legit messages and generates contextual synonym paraphrases.
           - **Adversarial Homoglyph / Leetspeak Obfuscation (`adversarial_obfuscation`)**: Simulates phishers evading filters (`n0rthbank`, `v3rify`, `acc0unt`, character drops/swaps), paired with hybrid `word + char_wb` TF-IDF and URL-risk normalization (`preprocess_url_aware`).
        2. **Real External Dataset Download (`UCI SMS Spam Collection`, 5,574 real SMS messages)**:
           - Automatically downloaded into `data/uci_sms_spam.csv` along with a domain-adapted **Real Smishing subset** (`data/real_smishing_curated.csv`, 2,480 messages at 20% positive prevalence).
        """
    )

    aug_c1, aug_c2 = st.columns([1.1, 1.0])
    with aug_c1:
        st.markdown("#### Standard vs. Adversarial Obfuscated Benchmark (500 Test Messages)")
        bench_rows = []
        for m_name in [
            "Logistic Regression (Baseline)",
            "LR + Char N-grams (char_wb 2-5)",
            "LR + Data Augmentation (Word+Char)",
            "LR + Data Aug + URL-Aware (Enhanced)",
        ]:
            p_std = (test_probs[m_name] >= 0.5).astype(int)
            pr_s, rc_s, f1_s, _ = precision_recall_fscore_support(yte_arr, p_std, average="binary", zero_division=0)
            p_adv = (bundle["adv_probs"][m_name] >= 0.5).astype(int)
            pr_a, rc_a, f1_a, _ = precision_recall_fscore_support(yte_arr, p_adv, average="binary", zero_division=0)
            bench_rows.append({
                "Pipeline": m_name,
                "Std Precision": round(float(pr_s), 3),
                "Std Recall": round(float(rc_s), 3),
                "Std F1": round(float(f1_s), 3),
                "Adversarial Recall": round(float(rc_a), 3),
                "Adversarial F1": round(float(f1_a), 3),
            })
        st.dataframe(pd.DataFrame(bench_rows), width="stretch", hide_index=True)

    with aug_c2:
        st.markdown("#### Augmentation Strategy Breakdown (`X_train_aug`)")
        aug_counts = bundle["aug_df"]["aug_strategy"].value_counts().reset_index()
        aug_counts.columns = ["Augmentation Strategy", "Generated Training Samples"]
        st.dataframe(aug_counts, width="stretch", hide_index=True)

    st.markdown("#### 🧪 Sample Augmented Training Messages & Downloaded UCI Real SMS Dataset")
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        strat_filter = st.selectbox(
            "Inspect Generated Training Samples by Strategy:",
            ["hard_boundary_paraphrase", "markov_lm_synthesis", "adversarial_obfuscation"],
        )
        sub_aug = bundle["aug_df"][bundle["aug_df"]["aug_strategy"] == strat_filter][["label", "text"]].head(6)
        st.dataframe(sub_aug, width="stretch", hide_index=True)
    with sample_col2:
        st.markdown("**Downloaded Real UCI SMS Dataset (`data/uci_sms_spam.csv`, 5,574 msgs):**")
        st.dataframe(bundle["uci_df"][["original_uci_label", "text"]].head(6), width="stretch", hide_index=True)

# ===========================================================================
# TAB 4: Batch CSV Scanner & Export + Analyst Feedback Queue (Extensions 3 & 5)
# ===========================================================================
with tab_batch:
    st.subheader("📁 Batch CSV Message Scanner & Flagged Export (Extension)")
    st.caption(
        f"Upload a CSV file containing a `text` column (or test on a sample batch) to score all messages with "
        f"**{selected_model_name}** at threshold **τ = {threshold:.2f}** and export flagged messages."
    )

    uploaded_csv = st.file_uploader("Upload CSV file with a `text` column:", type=["csv"])
    use_sample_batch = st.checkbox("Use a sample batch of 30 test messages", value=(uploaded_csv is None))

    batch_df = None
    if uploaded_csv is not None:
        batch_df = pd.read_csv(uploaded_csv)
    elif use_sample_batch:
        batch_df = pd.DataFrame({
            "text": list(bundle["Xte"])[:30],
            "ground_truth_label": ["phish" if y == 1 else "legit" for y in bundle["yte"][:30]],
        })

    if batch_df is not None and "text" in batch_df.columns:
        scores_b = active_model.predict_proba(batch_df["text"].astype(str))[:, 1]
        batch_df = batch_df.copy()
        batch_df["phish_score"] = np.round(scores_b, 3)
        batch_df["predicted_class"] = np.where(scores_b >= threshold, "phish", "legit")
        flagged_df = batch_df[batch_df["predicted_class"] == "phish"]

        st.markdown(f"**Scanned {len(batch_df)} messages → {len(flagged_df)} flagged as `phish` (@ τ = {threshold:.2f})**")
        st.dataframe(batch_df, width="stretch", hide_index=True)

        csv_bytes = flagged_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Export Flagged Phishing Messages ({len(flagged_df)} rows) as CSV",
            data=csv_bytes,
            file_name=f"flagged_phishing_messages_tau_{threshold:.2f}.csv",
            mime="text/csv",
        )
    elif batch_df is not None:
        st.error("The uploaded CSV must contain a column named `text`.")

    st.markdown("---")
    st.subheader("🗂️ Logged Analyst Feedback (`data/feedback_log.csv`)")
    fb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feedback_log.csv")
    if os.path.exists(fb_path):
        st.dataframe(pd.read_csv(fb_path), width="stretch", hide_index=True)
    else:
        st.caption("No misclassified messages logged yet. Use the '🚩 Was this message misclassified?' expander in Tab 1 to log feedback.")
