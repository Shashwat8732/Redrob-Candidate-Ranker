# Redrob Candidate Ranker

An intelligent candidate ranking system for the Senior AI Engineer role at Redrob AI.
Ranks 100,000 candidates and outputs the top 100 best-fit candidates with scores
and reasoning.

## Project Structure

This repository contains two entry points, each serving a different purpose
required by the submission spec:

| File | Purpose | How to run |
|---|---|---|
| `filter.py` | Full pipeline on all 100,000 candidates → produces `submission.csv` | `python filter.py` |
| `app.py` | Interactive Streamlit demo for the sandbox requirement — same logic, runs on a sample, lets you edit the JD and explore results | `streamlit run app.py` |

Both files implement **identical scoring logic** (filters, embeddings, behavioral
scoring, penalties). `filter.py` is the reproducible terminal script used to
generate the official submission; `app.py` is the hosted sandbox where the
ranking pipeline can be verified interactively in a browser.

## Setup

```bash
pip install -r requirements.txt
```

## Run — produces submission.csv

```bash
python filter.py
```

**Note on offline execution:** The MiniLM model (`all-MiniLM-L6-v2`) is included
in this repo under `minilm_model/`, pre-downloaded, so the ranking step runs
fully offline — no network calls, no GPU. Completes in ~3-9 seconds on CPU.

## Sandbox demo

```bash
streamlit run app.py
```

Edit the JD in the text box, click "Rank Candidates", view the top results in
an interactive table, and download `submission.csv`.

## Approach

### Stage 1 — Hard Filters
- Experience < 5 years → reject
- Career history entirely in non-product industries → reject
- Primary domain is CV/Speech/ASR → reject (JD explicitly excludes these)
- Title in non-ML roles (HR, Sales, Civil Engineer, etc.) → reject
- Honeypot detection:
  - "Expert" proficiency claimed with < 6 months of usage
  - Career tenure mathematically impossible (60+ months at a company
    joined after 2021)

### Stage 2 — Semantic Relevance Score (weight: 0.45)
- MiniLM (`all-MiniLM-L6-v2`) sentence embeddings
- Cosine similarity between the JD text and each candidate's
  summary + headline + skills + career history descriptions

### Stage 3 — Experience Score (weight: 0.30)
- 7+ years → 1.0
- 5-7 years → 0.8
- (<5 years already filtered out in Stage 1)

### Stage 4 — Behavioral Score (weight: 0.25)
Combines all 23 Redrob signals — login recency, response rate, interview
completion rate, GitHub activity, verification status, notice period,
salary fit, relocation willingness, profile completeness, and more —
into a single weighted score.

### Final Penalties
- Notice period > 90 days → score × 0.90
- Recruiter response rate < 0.20 → score × 0.75

### Reasoning Generation
Each of the top 100 candidates gets a short, factual reasoning string
referencing their title, years of experience, count of AI-relevant skills,
and response rate — with an honest flag when notice period or response
rate is a concern.

## Results

- Candidates passed hard filters: 1,070 / 100,000
- Top 100 honeypot rate: 0%
- Runtime: ~3-9 seconds (well under the 5-minute limit)
- Validated with `challenge/validate_submission.py` — PASS

## AI Tools Used
Claude (Anthropic) for debugging

