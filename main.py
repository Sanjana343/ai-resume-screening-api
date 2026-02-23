from fastapi import FastAPI, File, UploadFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import PyPDF2
from io import BytesIO

app = FastAPI()

# Load Job Description
with open("job_description.txt", "r") as f:
    job_desc = f.read()

@app.post("/match_resume/")
async def match_resume(file: UploadFile = File(...)):
    resume_bytes = await file.read()

if file.filename.endswith(".pdf"):
    pdf_reader = PyPDF2.PdfReader(BytesIO(resume_bytes))
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

else:
    resume_text = resume_bytes.decode("utf-8")
   
    documents = [job_desc, resume_text]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = round(float(similarity[0][0]) * 100, 2)

    return {"Match Score (%)": score}