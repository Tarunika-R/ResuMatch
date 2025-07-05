# app.py

from flask import Flask, render_template, request
import joblib
import os
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load models
model = joblib.load('model/resume_classifier.pkl')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Resume parser
def parse_resume(file):
    ext = os.path.splitext(file.filename)[1]
    if ext == ".pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif ext == ".docx":
        return docx2txt.process(file)
    return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    match_score = None
    match_feedback = None
    missing_keywords = []

    if request.method == 'POST':
        role_applied = request.form['applied_role']
        job_desc = request.form['job_desc']
        file = request.files['resume']

        resume_text = parse_resume(file)

        if not resume_text.strip():
            match_feedback = "Could not extract text from the document. Please upload a valid resume."
        else:
            prediction = model.predict([resume_text])[0]

            resume_emb = embedder.encode([resume_text], convert_to_tensor=True)
            job_emb = embedder.encode([job_desc], convert_to_tensor=True)
            score = util.pytorch_cos_sim(resume_emb, job_emb).item()
            match_score = round(score * 100, 2)

            if prediction.lower() == role_applied.lower():
                match_feedback = "Your resume matches the applied role."
            else:
                match_feedback = f"Resume seems better suited for {prediction} rather than {role_applied}."

            if score < 0.7:
                resume_words = set(resume_text.lower().split())
                jd_words = set(job_desc.lower().split())
                missing = jd_words - resume_words
                missing_keywords = sorted([word for word in missing if len(word) > 4 and word.isalpha()])

    return render_template("index.html",
                           prediction=prediction,
                           match_score=match_score,
                           match_feedback=match_feedback,
                           missing_keywords=missing_keywords)

if __name__ == '__main__':
    app.run(debug=True)
