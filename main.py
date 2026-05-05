import os
import pandas as pd
import pdfplumber
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
RESUME_FOLDER = os.path.join(os.getcwd(), "resumes")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "outputs")

JOB_DESCRIPTION = """
Looking for Data Analyst with skills in Python, SQL, Excel,
Machine Learning, Data Visualization, Pandas, NumPy.
"""

REQUIRED_SKILLS = ["python", "sql", "machine learning", "pandas", "numpy"]

# -----------------------------
# TEXT EXTRACTION (PDF)
# -----------------------------
def extract_text_from_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"❌ Error reading PDF: {file_path} -> {e}")
        return ""

# -----------------------------
# TEXT EXTRACTION (DOCX)
# -----------------------------
def extract_text_from_docx(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return ""

        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    except Exception as e:
        print(f"❌ Invalid DOCX file: {file_path} -> {e}")
        return ""

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# LOAD RESUMES
# -----------------------------
def load_resumes(folder):
    resumes = []
    names = []

    print("📂 Checking folder:", folder)

    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        return names, resumes

    files = os.listdir(folder)
    print("📄 Files found:", files)

    for file in files:
        path = os.path.join(folder, file)
        print(f"\n🔍 Processing: {file}")

        text = ""

        if file.endswith(".pdf"):
            text = extract_text_from_pdf(path)

        elif file.endswith(".docx"):
            text = extract_text_from_docx(path)

        else:
            print(f"⚠️ Skipped unsupported file: {file}")
            continue

        if not text or text.strip() == "":
            print(f"⚠️ Empty or unreadable file skipped: {file}")
            continue

        cleaned = clean_text(text)

        if cleaned == "":
            print(f"⚠️ Cleaned text empty, skipping: {file}")
            continue

        resumes.append(cleaned)
        names.append(file)
        print(f"✅ Successfully loaded: {file}")

    return names, resumes

# -----------------------------
# SIMILARITY SCORE (TF-IDF)
# -----------------------------
def calculate_similarity(resumes, job_desc):
    vectorizer = TfidfVectorizer()

    try:
        vectors = vectorizer.fit_transform([job_desc] + resumes)
        similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        return similarity
    except Exception as e:
        print(f"❌ Error in similarity calculation: {e}")
        return []

# -----------------------------
# SKILL MATCHING SCORE
# -----------------------------
def calculate_skill_score(resume):
    score = 0
    for skill in REQUIRED_SKILLS:
        if skill in resume:
            score += 1
    return score / len(REQUIRED_SKILLS)

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    print("\n🚀 Automated Resume Screening Tool Started\n")

    job_desc_clean = clean_text(JOB_DESCRIPTION)

    names, resumes = load_resumes(RESUME_FOLDER)

    print(f"\n📊 Total valid resumes loaded: {len(resumes)}")

    if len(resumes) == 0:
        print("❌ No valid resumes found. Fix your files.")
        return

    similarity_scores = calculate_similarity(resumes, job_desc_clean)

    if len(similarity_scores) == 0:
        print("❌ Could not calculate similarity.")
        return

    # Create DataFrame
    df = pd.DataFrame({
        "Resume": names,
        "Similarity Score": similarity_scores
    })

    # Skill Score
    df["Skill Score"] = [calculate_skill_score(r) for r in resumes]

    # Final Score (Weighted)
    df["Final Score"] = df["Similarity Score"] * 0.7 + df["Skill Score"] * 0.3

    # Ranking
    df = df.sort_values(by="Final Score", ascending=False)

    # Shortlisting
    df["Status"] = df["Final Score"].apply(
        lambda x: "Shortlisted" if x >= 0.3 else "Rejected"
    )

    # Save Output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "results.csv")
    df.to_csv(output_path, index=False)

    # Display Results
    print("\n✅ FINAL RESULTS:\n")
    print(df)

    print(f"\n📁 Results saved at: {output_path}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()