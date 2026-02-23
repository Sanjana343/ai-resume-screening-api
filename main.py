from fastapi import FastAPI, File, UploadFile
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

import PyPDF2
from io import BytesIO

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

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
   
    job_embedding = model.encode(job_desc)
    resume_embedding = model.encode(resume_text)

    similarity = cosine_similarity (
        [job_embedding],
        [resume_embedding]
)

    score = round(float(similarity[0][0]) * 100, 2)

    return {"Match Score (%)": score}