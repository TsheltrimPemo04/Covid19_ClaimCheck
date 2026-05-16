"""
COVID-19 Claim Check — Quantitative Evaluation on COVID-Fact
=============================================================
Evaluates verify_claim.verify() against the COVID-Fact benchmark
(Saakyan et al., ACL 2021): a labelled dataset of ~4K real COVID-19
claims with crowd-verified SUPPORTED / REFUTED labels.

Pipeline:
    1. Download / load COVID-Fact dataset (.jsonl)
    2. Optionally take a stratified balanced sample
    3. For each claim, call verify(claim) and record the verdict
    4. Compute accuracy, macro-F1, per-class precision/recall, and
       coverage (% of claims the system committed to vs abstained on)
    5. Save per-claim predictions to evaluation_results.csv and a
       summary table to evaluation_summary.txt

Usage:
    python evaluate.py                # evaluate 300 stratified claims (default)
    python evaluate.py --limit 100    # quicker
    python evaluate.py --limit 0      # all ~4K claims (takes hours on CPU)

Decision-rule mapping:
    system "TRUE"             -> predicts SUPPORTED
    system "FALSE"            -> predicts REFUTED
    system "NOT ENOUGH INFO"  -> abstain (counts as wrong in overall accuracy,
                                dropped in "selective accuracy" / F1)
"""
import os
import json
import argparse
import urllib.request
import urllib.error
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "covidfact.jsonl")
OUT_CSV = os.path.join(HERE, "evaluation_results.csv")
OUT_TXT = os.path.join(HERE, "evaluation_summary.txt")

COVIDFACT_URL = "https://raw.githubusercontent.com/asaakyan/covidfact/main/COVIDFACT_dataset.jsonl"


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
def download_covidfact():
    if os.path.exists(DATA_PATH):
        return
    print(f"Downloading COVID-Fact dataset to {DATA_PATH}...")
    try:
        urllib.request.urlretrieve(COVIDFACT_URL, DATA_PATH)
        print(f"  Done.")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not download COVID-Fact ({e}).\n"
            f"Manual fallback:\n"
            f"  1. Visit https://github.com/asaakyan/covidfact\n"
            f"  2. Download COVIDFACT_dataset.jsonl\n"
            f"  3. Save it as: {DATA_PATH}"
        )


def load_covidfact() -> pd.DataFrame:
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    # Normalise labels — different revisions of the dataset use different casings
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin({"SUPPORTED", "REFUTED"})]
    df = df.drop_duplicates(subset=["claim"]).reset_index(drop=True)
    return df


def stratified_sample(df: pd.DataFrame, n_per_class: int, seed: int = 0) -> pd.DataFrame:
    if n_per_class <= 0:
        return df.copy()
    parts = []
    for lab in ("SUPPORTED", "REFUTED"):
        sub = df[df["label"] == lab]
        k = min(n_per_class, len(sub))
        parts.append(sub.sample(n=k, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


# ---------------------------------------------------------------
# Metrics (implemented by hand so we don't add sklearn as a dep)
# ---------------------------------------------------------------
def precision_recall_f1(y_true, y_pred, label):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn


def confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300,
                    help="Total claims to evaluate (stratified). 0 = all. Default 300.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=15,
                    help="Evidence sentences retrieved per claim (default 15).")
    args = ap.parse_args()

    download_covidfact()
    df = load_covidfact()
    print(f"COVID-Fact dataset loaded: {len(df)} claims "
          f"({(df['label']=='SUPPORTED').sum()} SUPPORTED, "
          f"{(df['label']=='REFUTED').sum()} REFUTED)")

    if args.limit > 0:
        df = stratified_sample(df, n_per_class=args.limit // 2, seed=args.seed)
        print(f"Stratified sample: {len(df)} claims")

    # Load the verifier (heavy: triggers model downloads on first run)
    import verify_claim as vc

    # Run
    from time import time
    print(f"\nRunning verification (top_k={args.top_k})...")
    preds, mapped, n_sup, n_ref, durations = [], [], [], [], []
    t_start = time()
    for i, row in df.iterrows():
        t0 = time()
        try:
            r = vc.verify(row["claim"], top_k=args.top_k)
            verdict = r["verdict"]
            ns, nr = r["n_supporting"], r["n_refuting"]
        except Exception as e:
            print(f"  [warn] {i}: {e}")
            verdict, ns, nr = "ERROR", 0, 0
        preds.append(verdict)
        mapped.append({"TRUE": "SUPPORTED",
                       "FALSE": "REFUTED",
                       "NOT ENOUGH INFO": "ABSTAIN",
                       "ERROR": "ABSTAIN"}[verdict])
        n_sup.append(ns); n_ref.append(nr); durations.append(time() - t0)
        if (i + 1) % 10 == 0:
            elapsed = time() - t_start
            rate = (i + 1) / elapsed
            eta  = (len(df) - i - 1) / rate
            print(f"  [{i+1:>4}/{len(df)}]  {rate:4.1f} claim/s  "
                  f"elapsed {elapsed/60:5.1f}min  eta {eta/60:5.1f}min")

    df = df.copy()
    df["pred_verdict"] = preds
    df["pred_label"]   = mapped
    df["n_supporting"] = n_sup
    df["n_refuting"]   = n_ref
    df["latency_sec"]  = durations
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-claim predictions: {OUT_CSV}")

    # ---- metrics ---------------------------------------------------
    y_true = df["label"].tolist()
    y_pred = df["pred_label"].tolist()

    total = len(df)
    n_abstain = sum(1 for p in y_pred if p == "ABSTAIN")
    coverage = (total - n_abstain) / total if total else 0

    n_correct_overall = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc_overall = n_correct_overall / total if total else 0

    committed_mask = [p != "ABSTAIN" for p in y_pred]
    y_t_c = [t for t, m in zip(y_true, committed_mask) if m]
    y_p_c = [p for p, m in zip(y_pred, committed_mask) if m]
    acc_selective = (sum(1 for t, p in zip(y_t_c, y_p_c) if t == p) / len(y_t_c)
                     if y_t_c else 0)

    # Per-class metrics (on committed predictions)
    rows_m = []
    f1s = []
    for lab in ("SUPPORTED", "REFUTED"):
        prec, rec, f1, tp, fp, fn = precision_recall_f1(y_t_c, y_p_c, lab)
        rows_m.append((lab, prec, rec, f1, tp, fp, fn))
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0

    # Confusion matrix (3-way including ABSTAIN)
    labels_full = ["SUPPORTED", "REFUTED", "ABSTAIN"]
    cm = confusion_matrix(y_true, y_pred, labels_full)

    # ---- summary printing & save ----------------------------------
    summary = []
    pr = summary.append
    pr("=" * 64)
    pr("COVID-19 Claim Check — Evaluation Summary")
    pr("=" * 64)
    pr(f"Benchmark        : COVID-Fact (Saakyan et al., ACL 2021)")
    pr(f"Claims evaluated : {total}")
    pr(f"  SUPPORTED      : {sum(1 for t in y_true if t=='SUPPORTED')}")
    pr(f"  REFUTED        : {sum(1 for t in y_true if t=='REFUTED')}")
    pr(f"top_k            : {args.top_k}")
    pr("")
    pr("Headline metrics")
    pr("-" * 64)
    pr(f"  Coverage              : {coverage:6.1%}   "
       f"(committed on {total-n_abstain}/{total}; abstained on {n_abstain})")
    pr(f"  Overall accuracy      : {acc_overall:6.1%}   "
       f"(abstentions counted as wrong)")
    pr(f"  Selective accuracy    : {acc_selective:6.1%}   "
       f"(accuracy when system commits)")
    pr(f"  Macro F1              : {macro_f1:6.1%}   (committed predictions only)")
    pr("")
    pr("Per-class metrics (committed predictions)")
    pr("-" * 64)
    pr(f"{'Class':<12}{'Precision':>12}{'Recall':>10}{'F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
    for lab, prec, rec, f1, tp, fp, fn in rows_m:
        pr(f"{lab:<12}{prec:>12.1%}{rec:>10.1%}{f1:>10.1%}{tp:>6}{fp:>6}{fn:>6}")
    pr("")
    pr("Confusion matrix (rows = gold, cols = predicted)")
    pr("-" * 64)
    pr(f"{'':<12}" + "".join(f"{l:>12}" for l in labels_full))
    for i, lab in enumerate(labels_full):
        pr(f"{lab:<12}" + "".join(f"{cm[i,j]:>12d}" for j in range(len(labels_full))))
    pr("")
    pr(f"Mean latency per claim : {np.mean(durations):.2f} s")
    pr(f"Total elapsed          : {(time()-t_start)/60:.1f} min")

    text = "\n".join(summary)
    print("\n" + text)
    with open(OUT_TXT, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved summary: {OUT_TXT}")


if __name__ == "__main__":
    main()
