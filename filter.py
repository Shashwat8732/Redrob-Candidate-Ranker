import json
import csv
import numpy as np
import os
from datetime import datetime, date
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

start = time.time()

#Load
print("Loading candidates...")
candidates = []
with open("candidates.jsonl", "rt") as f:
    for line in f:
        candidates.append(json.loads(line))
print(f"Loaded: {len(candidates)}")

#Model
print("Loading model...")
if os.path.exists('./minilm_model'):
    model = SentenceTransformer('./minilm_model')
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save('./minilm_model')

JD_TEXT = """
Senior AI Engineer role at product company.
Required: embeddings, vector search, retrieval systems,
ranking, NLP, LLMs, FAISS, Pinecone, sentence transformers,
information retrieval, recommendation systems, MLOps,
production ML deployment, Python, PyTorch.
Experience: 5-9 years at product companies.
Not looking for: pure CV/Speech, service companies only.
"""
jd_embedding = model.encode([JD_TEXT])

#Filters
GOOD_INDUSTRIES = [
    'ai/ml', 'fintech', 'e-commerce', 'food delivery',
    'transportation', 'software', 'edtech', 'healthtech'
]

REJECT_DOMAINS = [
    'computer vision', 'speech recognition',
    'asr', 'object detection', 'ocr'
]

REJECT_TITLES = [
    'civil engineer', 'mechanical engineer', 'hr manager',
    'accountant', 'marketing manager', 'content writer',
    'graphic designer', 'sales', 'operations manager',
    'business analyst', 'project manager', 'customer support',
    '.net developer', 'mobile developer', 'frontend engineer',
    'devops engineer', 'qa engineer', 'backend engineer',
    'java developer', 'cloud engineer', 'data engineer',
    'full stack developer', 'software engineer'
]

def exp_ok(profile):
    return profile['years_of_experience'] >= 5

def is_service_only(career_history):
    for job in career_history:
        industry = job.get('industry', '').lower()
        for good in GOOD_INDUSTRIES:
            if good in industry:
                return False
    return True

def wrong_domain(profile):
    text = profile.get('headline', '').lower()
    text += profile.get('summary', '').lower()
    for domain in REJECT_DOMAINS:
        if domain in text:
            return True
    return False

def wrong_title(profile):
    title = profile.get('current_title', '').lower()
    for bad in REJECT_TITLES:
        if bad in title:
            return True
    return False

def is_honeypot(candidate):
    # Check 1: expert skill but < 6 months
    for skill in candidate['skills']:
        if skill['proficiency'] == 'expert':
            if skill.get('duration_months', 0) < 6:
                return True, f"expert but low months"

    # Check 2: impossible tenure
    for job in candidate['career_history']:
        start_year = int(job['start_date'][:4])
        months = job.get('duration_months', 0)
        if start_year >= 2021 and months > 60:
            return True, f"impossible tenure"

    return False, ""

def should_reject(candidate):
    p = candidate['profile']
    ch = candidate['career_history']
    if not exp_ok(p):
        return True, "exp kam"
    if is_service_only(ch):
        return True, "service only"
    if wrong_domain(p):
        return True, "wrong domain"
    if wrong_title(p):
        return True, "wrong title"
    honeypot, reason = is_honeypot(candidate)
    if honeypot:
        return True, reason
    return False, ""

#Filter
print("Filtering...")
passed = []
for c in candidates:
    reject, reason = should_reject(c)
    if not reject:
        passed.append(c)
print(f"Passed: {len(passed)} | Rejected: {len(candidates)-len(passed)}")

#Behavioral Score
REF_DATE = date(2026, 6, 1)

def behavioral_score(signals):
    
    
    # 1.Profile completeness
    pc = signals['profile_completeness_score'] / 100

    # 2.Last active
    last_active = datetime.strptime(
        signals['last_active_date'], '%Y-%m-%d').date()
    days = (REF_DATE - last_active).days
    if days < 30:       days_score = 1.0
    elif days < 90:     days_score = 0.7
    elif days < 180:    days_score = 0.4
    else:               days_score = 0.1

    # 3.Open to work
    otw = 1.0 if signals['open_to_work_flag'] else 0.3

    # 4. Profile views
    views = min(signals['profile_views_received_30d'] / 200, 1.0)

    # 5.Applications submitted
    apps = min(signals['applications_submitted_30d'] / 15, 1.0)

    # 6.Recruiter response rate
    rr = signals['recruiter_response_rate']

    # 7.Response time (fast = good)
    rt = signals['avg_response_time_hours']
    if rt <= 24:        rt_score = 1.0
    elif rt <= 72:      rt_score = 0.7
    elif rt <= 168:     rt_score = 0.4
    else:               rt_score = 0.2

    # 8.Skill assessments
    assessments = signals['skill_assessment_scores']
    if assessments:
        ac = sum(assessments.values()) / len(assessments) / 100
    else:
        ac = 0.3

    # 9.Connections
    conn = min(signals['connection_count'] / 500, 1.0)

    # 10.Endorsements
    end = min(signals['endorsements_received'] / 100, 1.0)

    # 11.Notice period
    notice = signals['notice_period_days']
    if notice <= 30:    notice_score = 1.0
    elif notice <= 60:  notice_score = 0.85
    elif notice <= 90:  notice_score = 0.65
    else:               notice_score = 0.3

    # 12. Salary match (JD budget ~40-60 LPA)
    sal = signals['expected_salary_range_inr_lpa']
    sal_mid = (sal['min'] + sal['max']) / 2
    if 20 <= sal_mid <= 60:   sal_score = 1.0
    elif sal_mid < 20:        sal_score = 0.7
    else:                     sal_score = 0.5

    # 13.Work mode 
    wm = signals['preferred_work_mode']
    wm_score = 1.0 if wm in ['hybrid', 'flexible'] else 0.8

    # 14.Willing to relocate
    reloc = 1.0 if signals['willing_to_relocate'] else 0.8

    # 15.GitHub
    github = signals['github_activity_score']
    github_score = min(github / 100, 1.0) if github > 0 else 0.2

    # 16.Search appearance
    search = min(signals['search_appearance_30d'] / 300, 1.0)

    # 17.Saved by recruiters
    saved = min(signals['saved_by_recruiters_30d'] / 20, 1.0)

    # 18.Interview completion
    icr = signals['interview_completion_rate']

    # 19.Offer acceptance
    oar = signals['offer_acceptance_rate']
    oar_score = oar if oar >= 0 else 0.5

    # 20.Verified email
    ver_email = 1.0 if signals['verified_email'] else 0.0

    # 21.Verified phone
    ver_phone = 1.0 if signals['verified_phone'] else 0.0

    # 22.LinkedIn connected
    linkedin = 1.0 if signals['linkedin_connected'] else 0.5

    # 23.Signup date
    signup = datetime.strptime(
        signals['signup_date'], '%Y-%m-%d').date()
    signup_days = (REF_DATE - signup).days
    signup_score = min(signup_days / 365, 1.0)

    # Weighted final
    score = (
        days_score   * 0.15 +
        otw          * 0.12 +
        rr           * 0.10 +
        icr          * 0.08 +
        notice_score * 0.08 +
        github_score * 0.07 +
        rt_score     * 0.06 +
        pc           * 0.05 +
        ac           * 0.05 +
        ver_email    * 0.04 +
        ver_phone    * 0.04 +
        oar_score    * 0.04 +
        saved        * 0.03 +
        conn         * 0.03 +
        end          * 0.02 +
        views        * 0.02 +
        apps         * 0.02 +
        search       * 0.02 +
        sal_score    * 0.02 +
        wm_score     * 0.02 +
        reloc        * 0.02 +
        linkedin     * 0.01 +
        signup_score * 0.01
    )

    return round(score, 4)

#Embed
texts = []
for c in passed:
    skills_text = ' '.join([s['name'] for s in c['skills']])
    career_text = ' '.join([
        job.get('description', '') + ' ' + job.get('title', '')
        for job in c['career_history']
    ])
    profile_text = (
        c['profile'].get('summary', '') + ' ' +
        c['profile'].get('headline', '') + ' ' +
        skills_text + ' ' + career_text
    )
    texts.append(profile_text)

if os.path.exists('embeddings.npy'):
    print("Loading saved embeddings...")
    embeddings = np.load('embeddings.npy')
    if len(embeddings) != len(texts):
        print("Re-embedding...")
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
        np.save('embeddings.npy', embeddings)
else:
    print("Embedding candidates...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    np.save('embeddings.npy', embeddings)
print("Embeddings ready!")

#Score
scored = []
for i, c in enumerate(passed):
    rel = float(cosine_similarity(jd_embedding, [embeddings[i]])[0][0])
    beh = behavioral_score(c['redrob_signals'])
    exp = c['profile']['years_of_experience']
    exp_score = 1.0 if exp >= 7 else 0.8 if exp >= 5 else 0.6

    final = rel*0.45 + exp_score*0.30 + beh*0.25

    #Notice period penalty
    notice = c['redrob_signals']['notice_period_days']
    if notice > 90:
        final *= 0.90
    
        #Low response rate penalty
    rr = c['redrob_signals']['recruiter_response_rate']
    if rr < 0.20:
      final *= 0.75

    scored.append((round(min(final, 1.0), 4), c))

scored.sort(key=lambda x: x[0], reverse=True)

#Top 10
print("\nTop 10:")
for score, c in scored[:10]:
    print(c['candidate_id'], score,
          c['profile']['current_title'],
          c['profile']['current_company'])

#Reasoning
AI_SKILLS = [
    'embedding', 'vector', 'retrieval', 'ranking', 'nlp',
    'llm', 'faiss', 'pinecone', 'transformer', 'mlops',
    'pytorch', 'tensorflow', 'langchain', 'rag', 'search',
    'recommendation', 'fine-tun', 'sentence', 'weaviate',
    'qdrant', 'milvus', 'information retrieval', 'opensearch',
    'elasticsearch', 'bm25', 'rerank', 'peft', 'lora'
]

def make_reasoning(c, score):
    p = c['profile']
    signals = c['redrob_signals']

    ai_count = 0
    for skill in c['skills']:
        name = skill['name'].lower()
        for a in AI_SKILLS:
            if a in name:
                ai_count += 1
                break

    notice = signals['notice_period_days']
    rr = signals['recruiter_response_rate']

    concerns = []
    if notice > 90:
        concerns.append(f"{notice}-day notice period")
    if rr < 0.40:
        concerns.append(f"low response rate ({rr})")

    base = (
        f"{p['current_title']} with {p['years_of_experience']}yr exp; "
        f"{ai_count} AI core skills"
    )

    if concerns:
        return base + "; but " + ", ".join(concerns) + "."
    else:
        return base + f"; response rate {rr}; strong overall fit."
#CSV
with open('team_AlphaIntellect.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    for rank, (score, c) in enumerate(scored[:100], start=1):
        writer.writerow([
            c['candidate_id'], rank, score,
            make_reasoning(c, score)
        ])

print("\nCSV ready: team_AlphaIntellect.csv")

#Validate
with open('team_AlphaIntellect.csv', 'r') as f:
    rows = list(csv.DictReader(f))
scores = [float(r['score']) for r in rows]
ok = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
print(f"Total rows: {len(rows)}")
print(f"Score non-increasing: {ok}")
print(f"Duplicate IDs: {len(rows) - len(set(r['candidate_id'] for r in rows))}")
print(f"\nFirst: {rows[0]}")
print(f"Last:  {rows[-1]}")

elapsed = time.time() - start
print(f"\nTotal time: {elapsed:.1f} seconds")




with open('team_AlphaIntellect.csv', 'r') as f:
    content = f.read()
print(content)

#Honeypots in top 100
honeypot_count = 0
for score, c in scored[:100]:
    found, reason = is_honeypot(c)
    if found:
        honeypot_count += 1
        print(c['candidate_id'], reason)

print(f"\nHoneypots in top 100: {honeypot_count}/100")
print(f"Rate: {honeypot_count}%")
print(f"Safe limit: <10%")