# COVID-19 Claim Check

A claim-verification platform for COVID-19. Users enter a claim, the system retrieves the most relevant evidence sentences from a corpus of peer-reviewed research abstracts, runs each through a fact-verification model, and produces a verdict — **TRUE / FALSE / NOT ENOUGH INFO** — with cited sources.

## How it works

1. **Retrieval** — the user's claim is embedded with a sentence-transformer (`all-MiniLM-L6-v2`) and compared against pre-computed embeddings of every sentence in every abstract using cosine similarity. The top-K most similar sentences are returned as candidate evidence.
2. **Stance classification** — each `(evidence, claim)` pair is fed into `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, an NLI model trained on the FEVER fact-verification dataset, which classifies the relation as `entailment`, `contradiction`, or `neutral`.
3. **Verdict aggregation** — votes are tallied. The verdict is `TRUE` if there are at least 2 supporting sentences and they outnumber refuting ones; `FALSE` in the symmetric case; otherwise `NOT ENOUGH INFO`.

## Dataset

CORD-19 (COVID-19 Open Research Dataset) — the 2022 release containing ~1M scholarly articles. After filtering to COVID-related papers with usable abstracts and capping to the 30,000 most recent, the evidence corpus contains ~30K abstracts → ~210K sentences.

The CSV is too large for GitHub. Download `metadata.csv` from [the CORD-19 page on Kaggle](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) and place it in this folder before running `preprocess.py`.

## Quick start

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Place metadata.csv in this folder, then preprocess
python3 preprocess.py

# 3. Build the sentence-level embedding index (~15-30 min on CPU)
python3 build_index.py

# 4. Launch the web UI
python3 -m streamlit run app.py
```

Or run a single claim from the CLI:

```bash
python3 verify_claim.py "Wearing face masks reduces the transmission of SARS-CoV-2"
```

## Project structure

| File | Purpose |
| --- | --- |
| `preprocess.py` | Streams `metadata.csv`, filters to COVID papers with abstracts, dedupes, caps to 30K most recent, cleans text, writes `data_preprocessed.csv`. |
| `build_index.py` | Splits abstracts into sentences, embeds each with a sentence-transformer, saves the embedding matrix and metadata index. |
| `verify_claim.py` | Loads index + NLI model. Exposes `verify(claim)` and a CLI. |
| `app.py` | Streamlit web frontend. |
| `requirements.txt` | Python dependencies. |
| `run_in_colab.ipynb` | Notebook that runs the whole pipeline end-to-end in Google Colab. |

## Tuning

Decision thresholds live at the top of `verify_claim.py`:

- `TOP_K` — how many candidate sentences to retrieve per claim (default 15).
- `MIN_EVIDENCE` — minimum supporting/refuting sentences required for a non-NEI verdict (default 2).
- `MARGIN` — how much one stance must lead the other by (default 1).
- `SUPPORT_THRESHOLD` — NLI confidence needed to count a sentence as supporting/refuting (default 0.55).

## Limitations

The verifier reflects what's in the corpus, not absolute scientific truth. If the corpus is biased, outdated, or missing relevant papers, the verdict will be too. Treat verdicts as a starting point for further reading, not as medical advice.
