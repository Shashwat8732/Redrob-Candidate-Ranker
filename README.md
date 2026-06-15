# Redrob Candidate Ranker

An intelligent candidate ranking system for the Senior AI Engineer role at
Redrob AI. Ranks 100,000 candidates and outputs the top 100 best-fit
candidates with scores and reasoning.

## Project Structure

This repo has two entry points, each required by the submission spec:

| File | Purpose | Run with |
|---|---|---|
| `filter.py` | Full pipeline on all 100,000 candidates → produces `submission.csv` (used for Stage 3 reproduction) | `python filter.py` |
| `app.py` | Interactive Streamlit sandbox demo — same logic, runs on a small sample, lets you edit the JD and explore results | `streamlit run app.py` |

Both files implement **identical scoring logic** (filters, embeddings,
behavioral scoring, penalties). They differ only in interface and data size.

## File Paths

| File | Expects data at | Notes |
|---|---|---|
| `filter.py` | `candidates.jsonl` (repo root) | Full 100K dataset. Not included in repo (too large) — place it here for reproduction. |
| `app.py` | `candidates_sample.jsonl` (repo root) | 1000-candidate subset, included in repo for the sandbox demo. |
| both | `minilm_model/` (repo root) | Pre-downloaded MiniLM model, included in repo for offline execution. |

## Setup

```bash
pip install -r requirements.txt
```

## Run — produces submission.csv (Stage 3 reproduction)

```bash
cp /path/to/candidates.jsonl ./candidates.jsonl
python filter.py
```

**Offline execution:** The MiniLM model (`all-MiniLM-L6-v2`) is pre-downloaded
in `minilm_model/`, so ranking runs fully offline — no network calls, no GPU.
Completes in ~3-9 seconds on CPU for 100,000 candidates.

## Sandbox demo

```bash
streamlit run app.py
```

Uses the pre-loaded `candidates_sample.jsonl` (1000 candidates). Edit the JD,
click "Rank Candidates", view results in an interactive table, and download
`submission.csv`.

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
- Cosine similarity between JD text and each candidate's
  summary + headline + skills + career history descriptions

### Stage 3 — Experience Score (weight: 0.30)
- 7+ years → 1.0
- 5-7 years → 0.8

### Stage 4 — Behavioral Score (weight: 0.25)
Combines all 23 Redrob signals — login recency, response rate, interview
completion rate, GitHub activity, verification status, notice period,
salary fit, relocation willingness, profile completeness, and more —
into a weighted score.

### Final Penalties
- Notice period > 90 days → score × 0.90
- Recruiter response rate < 0.20 → score × 0.75

### Reasoning Generation
Each of the top 100 candidates gets a short, factual reasoning string
referencing title, years of experience, count of AI-relevant skills, and
response rate — with an honest concern flag when notice period or response
rate is weak.

## Results

- Candidates passed hard filters: 1,070 / 100,000
- Top 100 honeypot rate: 0%
- Runtime: ~3-9 seconds (well under the 5-minute limit)
- Validated with `challenge/validate_submission.py` — PASS

## AI Tools Used
Claude (Anthropic) for debugging
