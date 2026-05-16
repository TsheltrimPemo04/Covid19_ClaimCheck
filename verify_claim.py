import os
import sys
import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SENT_CSV = os.path.join(HERE, "sentence_index.csv")
SENT_EMB = os.path.join(HERE, "sentence_embeddings.npy")
CONFIG = os.path.join(HERE, "index_config.json")

# Trained on MNLI + FEVER + ANLI — ideal for claim verification
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

TOP_K = 15                  # sentences retrieved per claim
MIN_EVIDENCE = 2            # sentences needed to support a verdict
MARGIN = 1                  # min lead of one stance over the other
SUPPORT_THRESHOLD = 0.55    # min NLI prob to count as a vote
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------
# Load everything (cached at module level)
# ---------------------------------------------------------------
print(f"[init] device: {DEVICE}")

with open(CONFIG) as f:
    _cfg = json.load(f)

print(f"[init] loading retriever: {_cfg['embedding_model']}")
_retriever = SentenceTransformer(_cfg["embedding_model"], device=DEVICE)

print(f"[init] loading NLI model: {NLI_MODEL}")
_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
_nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
_nli.eval()

# MNLI/FEVER label order used by this model:
# id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
_ID2LABEL = _nli.config.id2label
_LABEL_IDX = {v.lower(): k for k, v in _ID2LABEL.items()}
_ENT = _LABEL_IDX["entailment"]
_NEU = _LABEL_IDX["neutral"]
_CON = _LABEL_IDX["contradiction"]

print("[init] loading sentence index...")
_sent_df = pd.read_csv(SENT_CSV)
_sent_embs = np.load(SENT_EMB)
assert len(_sent_df) == _sent_embs.shape[0], "index/embedding length mismatch"
print(f"[init] ready ({len(_sent_df)} evidence sentences)\n")


# ---------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------
def retrieve(claim: str, top_k: int = TOP_K) -> pd.DataFrame:
    q = _retriever.encode([claim], normalize_embeddings=True).astype("float32")
    sims = _sent_embs @ q[0]                                   # (N,)
    idx = np.argpartition(-sims, top_k)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    out = _sent_df.iloc[idx].copy()
    out["similarity"] = sims[idx]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------
# Stance classification (NLI)
# ---------------------------------------------------------------
@torch.no_grad()
def classify_stance(claim: str, evidences: list[str], batch_size: int = 16):
    """For each (evidence, claim) pair compute entail / neutral / contradict probs.

    Note: NLI is directional — premise (evidence) entails hypothesis (claim).
    """
    probs = []
    for i in range(0, len(evidences), batch_size):
        batch = evidences[i : i + batch_size]
        enc = _tokenizer(
            batch, [claim] * len(batch),
            return_tensors="pt", truncation=True, padding=True, max_length=256,
        ).to(DEVICE)
        logits = _nli(**enc).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs) if probs else np.zeros((0, 3))


# ---------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------
def verify(claim: str, top_k: int = TOP_K) -> dict:
    ev = retrieve(claim, top_k=top_k)
    probs = classify_stance(claim, ev["sentence"].tolist())

    ev["p_support"]  = probs[:, _ENT]
    ev["p_neutral"]  = probs[:, _NEU]
    ev["p_refute"]   = probs[:, _CON]

    def _label(row):
        p = {"SUPPORTS": row["p_support"],
            "REFUTES":  row["p_refute"],
            "NEUTRAL":  row["p_neutral"]}
        top = max(p, key=p.get)
        if p[top] < SUPPORT_THRESHOLD:
            return "NEUTRAL"
        return top

    ev["stance"] = ev.apply(_label, axis=1)
    n_sup = (ev["stance"] == "SUPPORTS").sum()
    n_ref = (ev["stance"] == "REFUTES").sum()

    if n_sup >= MIN_EVIDENCE and n_sup >= n_ref + MARGIN:
        verdict = "TRUE"
    elif n_ref >= MIN_EVIDENCE and n_ref >= n_sup + MARGIN:
        verdict = "FALSE"
    else:
        verdict = "NOT ENOUGH INFO"

    return {
        "claim": claim,
        "verdict": verdict,
        "n_supporting": int(n_sup),
        "n_refuting": int(n_ref),
        "evidence": ev.to_dict(orient="records"),
    }


# ---------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------
def _fmt_evidence(e):
    cite = e.get("title", "")[:90] + ("..." if len(e.get("title", "")) > 90 else "")
    return (
        f"  [{e['stance']}] sim={e['similarity']:.2f} "
        f"sup={e['p_support']:.2f} ref={e['p_refute']:.2f}\n"
        f"    → \"{e['sentence']}\"\n"
        f"    Source: {cite}  DOI:{e.get('doi', '')}\n"
    )


def print_result(result: dict, max_per_group: int = 4):
    print("=" * 72)
    print(f"CLAIM  : {result['claim']}")
    print(f"VERDICT: {result['verdict']}")
    print(f"        (supporting: {result['n_supporting']}, "
        f"refuting: {result['n_refuting']})")
    print("=" * 72)

    ev = result["evidence"]
    for group in ("SUPPORTS", "REFUTES", "NEUTRAL"):
        items = [e for e in ev if e["stance"] == group][:max_per_group]
        if not items:
            continue
        print(f"\n{group}:")
        for e in items:
            print(_fmt_evidence(e))


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:])
        print_result(verify(claim))
    else:
        print("Interactive mode — type a claim and press Enter (Ctrl+C to exit)")
        while True:
            try:
                c = input("\nClaim> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not c:
                continue
            print_result(verify(c))
