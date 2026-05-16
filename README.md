---
title: COVID-19 Claim Check
emoji: 🦠
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
short_description: Evidence-based fact verification for COVID-19 claims
---

# COVID-19 Claim Check

A scientific claim-verification platform for COVID-19. Users enter a free-text claim, and the system retrieves the most relevant evidence sentences from a corpus of peer-reviewed research abstracts, runs each through a fact-verification model, and produces a verdict — **TRUE / FALSE / NOT ENOUGH INFO** — with cited sources.

## How it works

A two-stage retrieve-and-verify pipeline:

1. **Dense retrieval.** The user's claim is embedded with `sentence-transformers/all-MiniLM-L6-v2` and compared by cosine similarity against ~298,000 pre-embedded sentences from 30,000 COVID-19 research abstracts (CORD-19, 2022 release). The top-15 most similar sentences are returned as candidate evidence.
2. **NLI-based stance classification.** Each `(evidence, claim)` pair is fed into `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` — a DeBERTa-v3 model fine-tuned on the FEVER fact-verification dataset — which classifies the relation as SUPPORTS, REFUTES, or NEUTRAL.
3. **Verdict aggregation.** Votes are tallied with conservative thresholds (≥2 sentences agreeing AND a margin over the opposing stance) to produce the final verdict.

## Quantitative results

Evaluated on a stratified 300-claim sample from the COVID-Fact benchmark (Saakyan et al., ACL 2021):

| Metric | Value |
| --- | --- |
| Selective accuracy | 57.3 % |
| Macro F1 | 50.1 % |
| Coverage (commit rate) | 41.3 % |
| REFUTED F1 | 69.0 % |
| SUPPORTED F1 | 31.2 % |

The system is calibrated to abstain rather than commit to wrong answers — a deliberate design choice for a public-health fact-checker.

## Project

This is the live demo for an undergraduate NLP project at Gyalpozhing College of Information Technology, Thimphu. Source code, evaluation script, and full report on [GitHub](https://github.com/TsheltrimPemo04/Covid19_ClaimCheck).

## Disclaimer

The verdict reflects what is present in the corpus, not absolute scientific truth. Treat verdicts as a starting point for further reading, not as medical advice.
