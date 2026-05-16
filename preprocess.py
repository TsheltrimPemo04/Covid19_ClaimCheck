import os
import re
import string
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# NLTK setup (optional — falls back to regex tokenizer if missing)
# ---------------------------------------------------------------
NLTK_AVAILABLE = True
try:
    import nltk
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize
    from nltk.stem import WordNetLemmatizer
    _STOPWORDS = set(_nltk_stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    def _tokenize(text): return _nltk_word_tokenize(text)
    def _lemmatize(tok): return _lemmatizer.lemmatize(tok)
except Exception:
    NLTK_AVAILABLE = False
    # Minimal built-in stopword list (good enough for retrieval preprocessing)
    _STOPWORDS = {
        "a","an","and","are","as","at","be","been","being","but","by","could",
        "did","do","does","doing","for","from","had","has","have","having","he",
        "her","here","hers","him","his","how","i","if","in","into","is","it",
        "its","itself","just","me","more","most","my","myself","of","off","on",
        "once","only","or","other","our","ours","ourselves","out","over","own",
        "same","she","should","so","some","such","than","that","the","their",
        "theirs","them","themselves","then","there","these","they","this","those",
        "through","to","too","under","until","up","very","was","we","were","what",
        "when","where","which","while","who","whom","why","will","with","you",
        "your","yours","yourself","yourselves",
    }
    def _tokenize(text): return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    def _lemmatize(tok): return tok


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "metadata.csv")     # 2022 CORD-19 release (~1M papers)
OUTPUT_CSV = os.path.join(HERE, "data_preprocessed.csv")
MIN_ABSTRACT_WORDS = 30

# Cap the number of papers kept (most recent first). Set to None for no cap.
# 30K most-recent COVID abstracts already gives ~5x more evidence than the
# old 2018 dataset and stays tractable for CPU embedding (~15 min).
MAX_ABSTRACTS = 30_000

# Add lemmatized-token columns (used by TF-IDF / BM25 baselines).
# Disable to skip the slow lemmatization pass — the embedding pipeline
# (build_index.py + verify_claim.py) doesn't need it.
ADD_TOKENIZED_COLUMNS = False

KEEP_COLS = [
    "cord_uid", "sha", "source_x", "title", "doi", "pmcid", "pubmed_id",
    "abstract", "publish_time", "authors", "journal", "url",
]

COVID_PATTERN = re.compile(
    r"covid|sars[- ]?cov[- ]?2|coronavirus|ncov|2019[- ]?ncov",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)       # URLs
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"\[[^\]]*\]", " ", text)              # [citations]
    text = re.sub(r"[^a-z0-9\s\.\,\-\;\:]", " ", text)   # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Keep negations — critical for claim verification
STOP_WORDS = _STOPWORDS - {
    "no", "not", "nor", "against", "without", "never", "none",
}

def tokenize_and_lemmatize(text: str):
    """Heavier normalization — for TF-IDF / BM25 baselines."""
    tokens = _tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation and len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [_lemmatize(t) for t in tokens]
    return tokens


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------
def main():
    print(f"Loading: {INPUT_CSV}")
    # Header sniff to figure out which of our desired columns actually exist
    head = pd.read_csv(INPUT_CSV, nrows=0)
    available = [c for c in KEEP_COLS if c in head.columns]
    print(f"  Columns being read: {available}")

    # Stream in chunks (file is up to ~1.65 GB; full load risks OOM on small machines)
    chunks = []
    total = 0
    for chunk in pd.read_csv(INPUT_CSV, usecols=available, dtype=str,
                            low_memory=False, chunksize=100_000,
                            on_bad_lines="skip"):
        total += len(chunk)
        # Pre-filter inside the chunk to keep memory low
        chunk = chunk.dropna(subset=["abstract"])
        # Quick COVID prefilter (saves a lot of memory on the 1M-row file)
        text = chunk["abstract"].fillna("") + " " + chunk.get("title", "").fillna("")
        chunk = chunk[text.str.contains(COVID_PATTERN, na=False)]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Streamed {total:,} raw rows → {len(df):,} COVID rows with abstracts")

    # 4. Fill optional text columns
    for col in ["title", "journal", "authors", "doi"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # 5. Parse dates
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    # 6. Drop duplicate abstracts
    before = len(df)
    df = df.drop_duplicates(subset=["abstract"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} duplicate abstracts")

    # 7. Sort by date (most recent first) and apply MAX_ABSTRACTS cap
    if "publish_time" in df.columns:
        df = df.sort_values("publish_time", ascending=False, na_position="last")
        df = df.reset_index(drop=True)
    if MAX_ABSTRACTS is not None and len(df) > MAX_ABSTRACTS:
        before = len(df)
        df = df.head(MAX_ABSTRACTS).reset_index(drop=True)
        print(f"  Capped to {MAX_ABSTRACTS:,} most recent papers (was {before:,})")

    # 8. Clean text
    print("  Cleaning text...")
    df["abstract_clean"] = df["abstract"].apply(clean_text)
    df["title_clean"] = df["title"].apply(clean_text) if "title" in df.columns else ""

    # 9. Drop very short abstracts (noise)
    before = len(df)
    df = df[df["abstract_clean"].str.split().str.len() >= MIN_ABSTRACT_WORDS]
    df = df.reset_index(drop=True)
    print(f"  Dropped {before - len(df)} very short abstracts (<{MIN_ABSTRACT_WORDS} words)")

    # 10. (Optional) Tokenize + lemmatize for TF-IDF / BM25 baselines
    if ADD_TOKENIZED_COLUMNS:
        print("  Tokenizing + lemmatizing (this takes a moment)...")
        df["tokens"] = df["abstract_clean"].apply(tokenize_and_lemmatize)
        df["abstract_tokens_joined"] = df["tokens"].apply(" ".join)
    else:
        print("  Skipping tokenization pass (ADD_TOKENIZED_COLUMNS=False)")

    # 11. Combined field for retrieval (title + abstract)
    if "title_clean" in df.columns:
        df["text_for_retrieval"] = (
            df["title_clean"].fillna("") + ". " + df["abstract_clean"]
        ).str.strip()
    else:
        df["text_for_retrieval"] = df["abstract_clean"]

    # 12. Assign a stable doc_id
    df.insert(0, "doc_id", np.arange(len(df)))

    # 13. Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal dataset: {len(df)} abstracts")
    print(f"Saved: {OUTPUT_CSV}")

    # 14. Quick peek
    print("\nSample rows:")
    preview_cols = [c for c in ["doc_id", "title", "publish_time", "journal"] if c in df.columns]
    print(df[preview_cols].head(3).to_string(index=False))
    print("\nAbstract length stats (words):")
    lens = df["abstract_clean"].str.split().str.len()
    print(lens.describe().round(1).to_string())


if __name__ == "__main__":
    main()
