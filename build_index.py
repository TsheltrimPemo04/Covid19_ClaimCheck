import os
import re
import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "data_preprocessed.csv")
SENT_CSV = os.path.join(HERE, "sentence_index.csv")
SENT_EMB = os.path.join(HERE, "sentence_embeddings.npy")
DOC_EMB = os.path.join(HERE, "doc_embeddings.npy")
CONFIG = os.path.join(HERE, "index_config.json")

# Pick a model:
#   'sentence-transformers/all-MiniLM-L6-v2'  — fast, general, 22M params (CPU-friendly)
#   'pritamdeka/S-PubMedBert-MS-MARCO'        — biomedical, better quality (needs GPU)
#   'allenai/specter'                         — scientific paper similarity
# Default: MiniLM (works on CPU in a few minutes). Switch to PubMedBERT if you have GPU.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64
MIN_SENT_WORDS = 5          # skip fragments / section labels
MAX_SENT_WORDS = 80         # chop extremely long sentences
META_COLS = ["doc_id", "title", "journal", "doi", "publish_time", "authors"]


# ---------------------------------------------------------------
# Sentence splitter (regex-based; works without NLTK)
# ---------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Protect common abbreviations that would otherwise trigger a split
    protected = text
    for abbr in ["e.g.", "i.e.", "et al.", "vs.", "cf.", "Fig.", "Eq.",
                "No.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "Jr."]:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    parts = _SENT_SPLIT.split(protected)
    parts = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return parts


def ok_sentence(s: str) -> bool:
    n = len(s.split())
    return MIN_SENT_WORDS <= n <= MAX_SENT_WORDS


# ---------------------------------------------------------------
# Build sentence-level DataFrame
# ---------------------------------------------------------------
def build_sentence_index(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        # Prefer the lightly-cleaned abstract (keeps sentence boundaries)
        text = row.get("abstract_clean") or row.get("abstract") or ""
        # The cleaner lower-cased everything — re-use the original abstract
        # for sentence splitting to preserve capitalization cues.
        original = row.get("abstract") or text
        sents = split_sentences(original)
        sent_id = 0
        for s in sents:
            s = s.strip()
            if not ok_sentence(s):
                continue
            rows.append({
                "doc_id": row["doc_id"],
                "sent_id": sent_id,
                "sentence": s,
                "title": row.get("title", ""),
                "journal": row.get("journal", ""),
                "doi": row.get("doi", ""),
                "publish_time": row.get("publish_time", ""),
                "authors": row.get("authors", ""),
            })
            sent_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Abstracts: {len(df)}")

    print("Splitting abstracts into sentences...")
    sent_df = build_sentence_index(df)
    print(f"  Sentences kept: {len(sent_df)}")

    sent_df.to_csv(SENT_CSV, index=False)
    print(f"  Wrote {SENT_CSV}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {dim}")

    print("Encoding sentences (this takes a few minutes on CPU, <1 min on GPU)...")
    sent_embs = model.encode(
        sent_df["sentence"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,     # so cosine sim == dot product
    ).astype("float32")
    np.save(SENT_EMB, sent_embs)
    print(f"  Saved sentence embeddings: {sent_embs.shape} → {SENT_EMB}")

    # Mean-pooled abstract embedding for optional two-stage retrieval
    print("Computing abstract-level embeddings (mean of sentence embeddings)...")
    doc_ids = sent_df["doc_id"].values
    unique_docs = np.unique(doc_ids)
    doc_embs = np.zeros((len(unique_docs), dim), dtype="float32")
    for i, did in enumerate(unique_docs):
        mask = doc_ids == did
        v = sent_embs[mask].mean(axis=0)
        v /= (np.linalg.norm(v) + 1e-12)
        doc_embs[i] = v
    np.save(DOC_EMB, doc_embs)
    print(f"  Saved doc embeddings: {doc_embs.shape} → {DOC_EMB}")

    with open(CONFIG, "w") as f:
        json.dump({
            "embedding_model": EMBED_MODEL,
            "embedding_dim": dim,
            "n_sentences": int(len(sent_df)),
            "n_documents": int(len(unique_docs)),
        }, f, indent=2)
    print(f"  Wrote {CONFIG}")
    print("\nDone. Run verify_claim.py to check claims.")


if __name__ == "__main__":
    main()
