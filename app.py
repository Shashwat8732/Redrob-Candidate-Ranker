import streamlit as st
import json
import numpy as np
import os
from datetime import datetime, date
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("🎯 Redrob Candidate Ranker")
st.caption("Senior AI Engineer — Intelligent Candidate Discovery")

#Load Model
@st.cache_resource
def load_model():
    if os.path.exists('./minilm_model'):
        return SentenceTransformer('./minilm_model')
    return SentenceTransformer('all-MiniLM-L6-v2')

#Load Candidates
@st.cache_data
def load_candidates():
    candidates = []
    path = "challenge/candidates.jsonl"
    if os.path.exists(path):
        with open(path, "rt") as f:
            for line in f:
                candidates.append(json.loads(line))
    return candidates

model = load_model()
candidates = load_candidates()

st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top N candidates", 10, 100, 10)

JD_DEFAULT = """Senior AI Engineer role at product company.
Required: embeddings, vector search, retrieval systems,
ranking, NLP, LLMs, FAISS, Pinecone, sentence transformers,
information retrieval, recommendation systems, MLOps,
production ML deployment, Python, PyTorch.
Experience: 5-9 years at product companies."""

jd_text = st.text_area("Job Description", value=JD_DEFAULT, height=200)

if st.button("🚀 Rank Candidates", type="primary"):
    if not candidates:
        st.error("candidates.jsonl not found!")
        st.stop()

    with st.spinner("Running pipeline..."):

        #Filters
        GOOD_INDUSTRIES = ['ai/ml', 'fintech', 'e-commerce',
            'food delivery', 'transportation', 'software',
            'edtech', 'healthtech']
        REJECT_DOMAINS = ['computer vision', 'speech recognition',
            'asr', 'object detection', 'ocr']
        REJECT_TITLES = ['civil engineer', 'mechanical engineer',
            'hr manager', 'accountant', 'marketing manager',
            'content writer', 'graphic designer', 'sales',
            'operations manager', 'business analyst',
            'project manager', 'customer support',
            '.net developer', 'mobile developer',
            'frontend engineer', 'devops engineer',
            'qa engineer', 'backend engineer', 'java developer',
            'cloud engineer', 'data engineer',
            'full stack developer', 'software engineer']

        def should_reject(c):
            p = c['profile']
            if p['years_of_experience'] < 5:
                return True
            industries = [j.get('industry','').lower()
                         for j in c['career_history']]
            if not any(g in i for i in industries
                      for g in GOOD_INDUSTRIES):
                return True
            text = p.get('headline','').lower() + \
                   p.get('summary','').lower()
            if any(d in text for d in REJECT_DOMAINS):
                return True
            title = p.get('current_title','').lower()
            if any(b in title for b in REJECT_TITLES):
                return True
            for skill in c['skills']:
                if skill['proficiency'] == 'expert':
                    if skill.get('duration_months', 0) < 6:
                        return True
            for job in c['career_history']:
                start_year = int(job['start_date'][:4])
                months = job.get('duration_months', 0)
                if start_year >= 2021 and months > 60:
                    return True
            return False

        passed = [c for c in candidates
                  if not should_reject(c)]

        #Embed
        REF_DATE = date(2026, 6, 1)
        jd_emb = model.encode([jd_text])

        texts = []
        for c in passed:
            skills_text = ' '.join(
                [s['name'] for s in c['skills']])
            career_text = ' '.join([
                job.get('description','') + ' ' +
                job.get('title','')
                for job in c['career_history']])
            texts.append(
                c['profile'].get('summary','') + ' ' +
                c['profile'].get('headline','') + ' ' +
                skills_text + ' ' + career_text)

        embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=False)

        
       #Behavioral Score
        def behavioral_score(signals):
            pc = signals['profile_completeness_score'] / 100

            last_active = datetime.strptime(
                signals['last_active_date'], '%Y-%m-%d').date()
            days = (REF_DATE - last_active).days
            if days < 30:       days_score = 1.0
            elif days < 90:     days_score = 0.7
            elif days < 180:    days_score = 0.4
            else:               days_score = 0.1

            otw = 1.0 if signals['open_to_work_flag'] else 0.3
            views = min(signals['profile_views_received_30d'] / 200, 1.0)
            apps = min(signals['applications_submitted_30d'] / 15, 1.0)
            rr = signals['recruiter_response_rate']

            rt = signals['avg_response_time_hours']
            if rt <= 24:        rt_score = 1.0
            elif rt <= 72:      rt_score = 0.7
            elif rt <= 168:     rt_score = 0.4
            else:               rt_score = 0.2

            assessments = signals['skill_assessment_scores']
            if assessments:
                ac = sum(assessments.values()) / len(assessments) / 100
            else:
                ac = 0.3

            conn = min(signals['connection_count'] / 500, 1.0)
            end = min(signals['endorsements_received'] / 100, 1.0)

            notice = signals['notice_period_days']
            if notice <= 30:    notice_score = 1.0
            elif notice <= 60:  notice_score = 0.85
            elif notice <= 90:  notice_score = 0.65
            else:               notice_score = 0.3

            sal = signals['expected_salary_range_inr_lpa']
            sal_mid = (sal['min'] + sal['max']) / 2
            if 20 <= sal_mid <= 60:   sal_score = 1.0
            elif sal_mid < 20:        sal_score = 0.7
            else:                     sal_score = 0.5

            wm = signals['preferred_work_mode']
            wm_score = 1.0 if wm in ['hybrid', 'flexible'] else 0.8

            reloc = 1.0 if signals['willing_to_relocate'] else 0.8

            github = signals['github_activity_score']
            github_score = min(github / 100, 1.0) if github > 0 else 0.2

            search = min(signals['search_appearance_30d'] / 300, 1.0)
            saved = min(signals['saved_by_recruiters_30d'] / 20, 1.0)
            icr = signals['interview_completion_rate']

            oar = signals['offer_acceptance_rate']
            oar_score = oar if oar >= 0 else 0.5

            ver_email = 1.0 if signals['verified_email'] else 0.0
            ver_phone = 1.0 if signals['verified_phone'] else 0.0
            linkedin = 1.0 if signals['linkedin_connected'] else 0.5

            signup = datetime.strptime(
                signals['signup_date'], '%Y-%m-%d').date()
            signup_days = (REF_DATE - signup).days
            signup_score = min(signup_days / 365, 1.0)

            return round(
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
                signup_score * 0.01,
                4
            )

        #Score
        AI_SKILLS = ['embedding','vector','retrieval',
            'ranking','nlp','llm','faiss','pinecone',
            'transformer','mlops','pytorch','tensorflow',
            'langchain','rag','search','recommendation',
            'fine-tun','sentence','weaviate','qdrant',
            'milvus','information retrieval','opensearch',
            'elasticsearch','bm25','rerank','peft','lora']

        scored = []
        for i, c in enumerate(passed):
            rel = float(cosine_similarity(
                jd_emb, [embeddings[i]])[0][0])
            beh = behavioral_score(c['redrob_signals'])
            exp = c['profile']['years_of_experience']
            exp_score = (1.0 if exp>=7 else
                        0.8 if exp>=5 else 0.6)
            final = rel*0.45 + exp_score*0.30 + beh*0.25
            notice = c['redrob_signals']['notice_period_days']
            if notice > 90:
                final *= 0.90
            rr = c['redrob_signals']['recruiter_response_rate']
            if rr < 0.20:
                final *= 0.75
            ai_count = sum(1 for s in c['skills']
                if any(a in s['name'].lower()
                       for a in AI_SKILLS))
            scored.append((round(min(final,1.0),4), c, ai_count))

        scored.sort(key=lambda x: x[0], reverse=True)

    # Display
    st.success(f"✅ Done! {len(passed)} candidates scored.")

    st.subheader(f"🏆 Top {top_n} Candidates")

    rows = []
    for rank, (score, c, ai_count) in enumerate(
            scored[:top_n], start=1):
        p = c['profile']
        sig = c['redrob_signals']
        rows.append({
            "Rank": rank,
            "ID": c['candidate_id'],
            "Title": p['current_title'],
            "Company": p['current_company'],
            "Exp (yr)": p['years_of_experience'],
            "Score": score,
            "AI Skills": ai_count,
            "Response Rate": sig['recruiter_response_rate'],
            "Notice (days)": sig['notice_period_days'],
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    #Download CSV
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['candidate_id','rank','score','reasoning'])
    for rank, (score, c, ai_count) in enumerate(
            scored[:100], start=1):
        p = c['profile']
        sig = c['redrob_signals']
        notice = sig['notice_period_days']
        rr = sig['recruiter_response_rate']
        concerns = []
        if notice > 90:
            concerns.append(f"{notice}-day notice period")
        if rr < 0.40:
            concerns.append(f"low response rate ({rr})")

        base = (
            f"{p['current_title']} with "
            f"{p['years_of_experience']}yr exp; "
            f"{ai_count} AI core skills; response rate {rr}")

        reasoning = (base + "; but " + ", ".join(concerns) + "."
                     if concerns else base + "; strong overall fit.")
        writer.writerow([
            c['candidate_id'], rank, score, reasoning])

    st.download_button(
        "📥 Download submission.csv",
        output.getvalue(),
        "submission.csv",
        "text/csv")