# Redrob Candidate Ranker

An intelligent candidate ranking system for the Senior AI Engineer role at
Redrob AI. Ranks 100,000 candidates and outputs the top 100 best-fit
candidates with scores and reasoning.

## Project Structure

| File | Purpose | Run with |
|---|---|---|
| `filter.py` | Full pipeline on all 100,000 candidates → produces `team_AlphaIntellect.csv` | `python filter.py` |
| `app.py` | Interactive Streamlit sandbox demo — same logic, runs on a 1,000-candidate sample | `streamlit run app.py` |
| `candidates_sample.jsonl` | 1,000-candidate subset for sandbox demo | — |
| `minilm_model/` | Pre-downloaded MiniLM model for offline execution | — |

Both files implement **identical scoring logic** — filters, MiniLM embeddings,
23 behavioral signals, penalties. They differ only in interface and data size.

## Setup

```bash
pip install -r requirements.txt
```

## Run — Full Reproduction (Stage 3)

```bash
# Place the full dataset in repo root
cp /path/to/candidates.jsonl ./candidates.jsonl

# Run the ranker
python filter.py
```

**Offline execution:** MiniLM model is pre-downloaded in `minilm_model/` —
ranking runs fully offline (no network calls, no GPU). Completes in ~3-9
seconds on CPU for 100,000 candidates.

## Sandbox Demo

```bash
streamlit run app.py
```

Live demo: https://redrob-candidate-ranker-alphaintellect.streamlit.app

Uses the pre-loaded `candidates_sample.jsonl` (1,000 candidates — a random
subset of the full dataset). Pipeline logic is identical to `filter.py`.
Results may differ from the full submission due to sample size — for exact
reproduction use `python filter.py` with the complete `candidates.jsonl`.

## Approach

### Stage 1 — Hard Filters
- Experience < 5 years → reject
- Career history entirely in non-product industries → reject
- Primary domain is CV/Speech/ASR → reject (JD explicitly excludes these)
- Title in non-ML roles (HR, Sales, Civil Engineer, etc.) → reject
- Honeypot detection:
  - "Expert" proficiency with < 6 months of usage → reject
  - Career tenure mathematically impossible (60+ months at a company
    joined after 2021) → reject

### Stage 2 — Semantic Relevance Score (weight: 0.45)
- MiniLM (`all-MiniLM-L6-v2`) sentence embeddings
- Cosine similarity between JD text and candidate's
  summary + headline + skills + career history descriptions

### Stage 3 — Experience Score (weight: 0.30)
- 7+ years → 1.0
- 5-7 years → 0.8
- (<5 years already filtered in Stage 1)

### Stage 4 — Behavioral Score (weight: 0.25)
Combines all 23 Redrob signals — login recency, open-to-work flag,
recruiter response rate, interview completion rate, GitHub activity,
verified email/phone, notice period, salary fit, willing to relocate,
profile completeness, saved by recruiters, offer acceptance rate,
LinkedIn connected, signup date, and more — into a single weighted score.

### Final Penalties
- Notice period > 90 days → score × 0.90
- Recruiter response rate < 0.20 → score × 0.75

### Reasoning Generation
Each of the top 100 candidates gets a factual reasoning string referencing
title, years of experience, AI-relevant skill count, and response rate —
with an honest concern flag when notice period or response rate is weak.

## Results

- Candidates passed hard filters: 1,070 / 100,000
- Top 100 honeypot rate: 0%
- Runtime: ~3-9 seconds (well under the 5-minute limit)
- Validated with `challenge/validate_submission.py` — PASS
- Top companies: Zomato, Meta, Yellow.ai, Flipkart, Paytm, Netflix, LinkedIn

## AI Tools Used

Claude (Anthropic) for debugging
