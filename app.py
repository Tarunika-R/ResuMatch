# app.py

from flask import Flask, render_template, request, send_file
import joblib
import os
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import tempfile

app = Flask(__name__)

# Load models
model = joblib.load('model/resume_classifier.pkl')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Setup database
DB_PATH = "evaluations.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT,
                        applied_role TEXT,
                        predicted_role TEXT,
                        match_score REAL,
                        feedback TEXT
                      )''')
    conn.commit()
    conn.close()

init_db()

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

# PDF generator
def generate_feedback_pdf(predicted_role, match_score, match_feedback, missing_keywords):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resume Evaluation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Predicted Role: {predicted_role}\n")
    pdf.multi_cell(0, 10, f"Similarity Score: {match_score}%\n")
    pdf.multi_cell(0, 10, f"Feedback: {match_feedback}\n")
    if missing_keywords:
        pdf.multi_cell(0, 10, "Missing Keywords:")
        for kw in missing_keywords:
            pdf.cell(0, 10, f"- {kw}", ln=True)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp.name)
    return temp.name

# Email sender
def send_email(recipient_email, subject, body, attachment_path):
    sender_email = "taru253@gmail.com"
    sender_password = "augu zivl stvx igbk"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    with open(attachment_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
        msg.attach(part)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    match_score = None
    match_feedback = None
    missing_keywords = []

    if request.method == 'POST':
        role_applied = request.form['applied_role']
        job_desc = request.form['job_desc']
        email = request.form['email']
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

            # Save to DB
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO evaluations (email, applied_role, predicted_role, match_score, feedback) VALUES (?, ?, ?, ?, ?)",
                           (email, role_applied, prediction, match_score, match_feedback))
            conn.commit()
            conn.close()

            # Generate and send feedback PDF
            pdf_path = generate_feedback_pdf(prediction, match_score, match_feedback, missing_keywords)
            send_email(email, "Your Resume Evaluation Report", "Please find the attached report.", pdf_path)

    return render_template("index.html",
                           prediction=prediction,
                           match_score=match_score,
                           match_feedback=match_feedback,
                           missing_keywords=missing_keywords)

if __name__ == '__main__':
    app.run(debug=True)
