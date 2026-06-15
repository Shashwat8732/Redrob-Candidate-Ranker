# Redrob Candidate Ranker

Ranks 100,000 candidates for a Senior AI Engineer role at Redrob AI.

## Setup
pip install -r requirements.txt

## Run
python filter.py

## Sandbox
streamlit run app.py

## Approach
- Stage 1: Hard filters (experience, industry, domain, title, honeypots)
- Stage 2: Semantic relevance (MiniLM, weight 0.45)
- Stage 3: Experience score (weight 0.30)
- Stage 4: Behavioral score - all 23 Redrob signals (weight 0.25)
- Penalties: notice period >90 days, response rate <0.20

## Results
- Passed filters: 1,070/100,000
- Honeypot rate in top 100: 0%
- Runtime: ~3 seconds
- Validated with challenge/validate_submission.py — PASS

## AI Tools Used
Claude (Anthropic) for guidance, debugging, code review.
