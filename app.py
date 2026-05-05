import streamlit as st
import pandas as pd
import pdfplumber
import docx
import re
import tempfile
import os
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ATS SaaS Dashboard", layout="wide")

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    transition: transform 0.3s ease;
}
.card:hover {
    transform: scale(1.02);
}

.selected {
    border-left: 5px solid #22c55e;
    background: rgba(34,197,94,0.15);
}

.rejected {
    border-left: 5px solid #ef4444;
    background: rgba(239,68,68,0.15);
}

.top {
    border: 2px solid gold;
    box-shadow: 0 0 15px gold;
}

.skill {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 10px;
    font-size: 12px;
}

.skill-match {
    background: rgba(34,197,94,0.2);
    color: #22c55e;
}

.skill-miss {
    background: rgba(200,200,200,0.15);
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text)

def extract_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
    except:
        pass
    return text

def extract_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""

def process_file(file):
    suffix = file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.read())
        path = tmp.name

    text = extract_pdf(path) if suffix == "pdf" else extract_docx(path)

    # keep raw for preview
    raw_text = text

    os.remove(path)

    return clean_text(text), raw_text

def load_default_resumes(folder="resumes"):
    names, texts, raw_texts = [], [], []

    if not os.path.exists(folder):
        return names, texts, raw_texts

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        text = ""
        if file.endswith(".pdf"):
            text = extract_pdf(path)
        elif file.endswith(".docx"):
            text = extract_docx(path)

        if text:
            names.append(file)
            texts.append(clean_text(text))
            raw_texts.append(text)

    return names, texts, raw_texts

def compute_similarity(resumes, jd):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([jd] + resumes)
    return cosine_similarity(vectors[0:1], vectors[1:]).flatten()

def skill_analysis(resume, skills):
    matched = [s for s in skills if s in resume]
    missing = [s for s in skills if s not in resume]
    return matched, missing

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🚀 AI Resume ATS Dashboard")

# Sidebar
st.sidebar.header("⚙️ Settings")

job_desc = st.sidebar.text_area(
    "Job Description",
    "Looking for Data Analyst with Python, SQL, Machine Learning, Pandas, NumPy"
)

skills_input = st.sidebar.text_input(
    "Required Skills",
    "python, sql, machine learning, pandas, numpy"
)

skills = [s.strip().lower() for s in skills_input.split(",")]

use_default = st.sidebar.checkbox("Demo Mode", True)

# Upload
files = st.file_uploader("Upload Resumes", type=["pdf","docx"], accept_multiple_files=True)

names, texts, raw_texts = [], [], []

if files:
    for f in files:
        cleaned, raw = process_file(f)
        if cleaned:
            names.append(f.name)
            texts.append(cleaned)
            raw_texts.append(raw)
elif use_default:
    st.success("Demo Mode Active")
    n, t, r = load_default_resumes()
    names.extend(n)
    texts.extend(t)
    raw_texts.extend(r)

# -----------------------------
# MAIN
# -----------------------------
if texts:

    jd_clean = clean_text(job_desc)
    sim = compute_similarity(texts, jd_clean)

    data = []

    for i, r in enumerate(texts):
        matched, missing = skill_analysis(r, skills)
        score = sim[i]*0.7 + (len(matched)/len(skills))*0.3
        status = "Selected" if score > 0.3 else "Rejected"

        data.append({
            "Resume": names[i],
            "Score": round(score,2),
            "Status": status,
            "Matched": matched,
            "Missing": missing,
            "Raw": raw_texts[i]
        })

    df = pd.DataFrame(data).sort_values(by="Score", ascending=False)

    # TOP CANDIDATE
    top_candidate = df.iloc[0]["Resume"]

    st.success(f"🏆 Top Candidate: {top_candidate}")

    # TABLE
    st.subheader("📊 Ranking")
    st.dataframe(df[["Resume","Score","Status"]])

    # GRAPH
    fig = px.bar(df, x="Resume", y="Score", color="Status")
    st.plotly_chart(fig)

    # -----------------------------
    # CARDS
    # -----------------------------
    st.subheader("📂 Candidate Analysis")

    for _, row in df.iterrows():

        status_class = "selected" if row["Status"]=="Selected" else "rejected"
        top_class = "top" if row["Resume"] == top_candidate else ""

        st.markdown(f"<div class='card {status_class} {top_class}'>", unsafe_allow_html=True)

        title = f"### {row['Resume']} ({row['Status']})"
        if row["Resume"] == top_candidate:
            title += " 🏆"

        st.markdown(title)
        st.write(f"Score: {row['Score']}")

        # Skills
        skill_html = ""

        for s in row["Matched"]:
            skill_html += f"<span class='skill skill-match'>{s}</span>"

        for s in row["Missing"]:
            skill_html += f"<span class='skill skill-miss'>{s}</span>"

        st.markdown(skill_html, unsafe_allow_html=True)

        # RESUME PREVIEW
        with st.expander("📄 Resume Preview"):
            st.text(row["Raw"][:1500])  # show limited text

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload resumes or enable demo mode")