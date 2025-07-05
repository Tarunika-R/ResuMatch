# resume_classifier_app.py (Streamlit UI)

import streamlit as st
import pandas as pd
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import base64
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher

# -------------------
# RESUME PARSING
# -------------------
def parse_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def parse_docx(file):
    return docx2txt.process(file)

# -------------------
# STREAMLIT UI
# -------------------
st.set_page_config(page_title="Resume Classifier + JD Match", layout="centered")
st.title("📄 Resume Job Fit Evaluator using NLP")
st.write("Upload a resume and paste a job description to evaluate match accuracy.")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description")
applied_role = st.text_input("Enter the Job Role You Are Applying For")

if uploaded_file and job_description and applied_role:
    file_text = ""
    if uploaded_file.name.endswith(".pdf"):
        file_text = parse_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        file_text = parse_docx(uploaded_file)

    if not file_text.strip():
        st.warning("Could not extract text from the document. Please upload a valid resume.")
    else:
        st.subheader("📄 Extracted Resume Text Preview")
        st.text_area("Resume Content", file_text[:1500] + ("..." if len(file_text) > 1500 else ""))

        # Load classification model
        model = joblib.load('resume_classifier.pkl')
        predicted_role = model.predict([file_text])[0]
        st.success(f"✅ Predicted Job Role from Resume: **{predicted_role}**")

        # Load embedding model
        @st.cache_resource
        def load_embedder():
            return SentenceTransformer('all-MiniLM-L6-v2')

        embedder = load_embedder()

        # Compute similarity score
        resume_emb = embedder.encode([file_text], convert_to_tensor=True)
        job_emb = embedder.encode([job_description], convert_to_tensor=True)
        score = util.pytorch_cos_sim(resume_emb, job_emb).item()

        match_percent = round(score * 100, 2)

        st.subheader("🔍 Job Match Evaluation")
        st.info(f"You have applied for the role: **{applied_role}**")

        if predicted_role.lower() == applied_role.lower():
            st.success(f"🎯 Your resume matches the applied role (**{applied_role}**).")
        else:
            st.warning(f"⚠️ Your resume seems more suitable for **{predicted_role}**, not **{applied_role}**.")

        if score >= 0.7:
            st.success(f"🟢 Good Match with Job Description! Similarity Score: **{match_percent}%**")
        else:
            st.error(f"🔴 Not a Strong Match with Job Description. Similarity Score: **{match_percent}%**")
            if st.button("What is Missing?"):
                # Local fallback for missing skills
                resume_words = set(file_text.lower().split())
                jd_words = set(job_description.lower().split())
                missing = jd_words - resume_words
                common_terms = [word for word in missing if len(word) > 4 and word.isalpha()]
                if common_terms:
                    st.markdown("### ❌ Possible Missing Keywords:")
                    for word in sorted(common_terms):
                        st.markdown(f"- {word}")
                else:
                    st.info("No significant missing terms found with simple keyword check.")
