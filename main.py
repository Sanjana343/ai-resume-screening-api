from typing import List
from fastapi import FastAPI, UploadFile, File
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import PyPDF2
from io import BytesIO

# ----------------------------
# Initialize FastAPI App
# ----------------------------
app = FastAPI()

# ----------------------------
# Load Transformer Model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Load Job Description Once
# ----------------------------
with open("job_description.txt", "r") as f:
    job_desc = f.read()

# ----------------------------
# Multi Resume Ranking Endpoint
# ----------------------------
@app.post("/rank_resume/")
async def rank_resumes(files: List[UploadFile] = File(..., description="Upload multiple resumes")):

    # Encode job description once
    job_embedding = model.encode(job_desc)

    results = []

    for file in files:

        resume_bytes = await file.read()

        # Extract text from PDF
        if file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(BytesIO(resume_bytes))
            resume_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text
        else:
            # Extract text from TXT
            resume_text = resume_bytes.decode("utf-8")

        # Generate embedding for resume
        resume_embedding = model.encode(resume_text)

        # Compute cosine similarity
        similarity = cosine_similarity(
            [job_embedding],
            [resume_embedding]
        )

        score = round(float(similarity[0][0]) * 100, 2)

        results.append({
            "resume_name": file.filename,
            "match_score": score
        })

    # Sort resumes by highest match score
    ranked_results = sorted(
        results,
        key=lambda x: x["match_score"],
        reverse=True
    )

    return {
        "ranked_resumes": ranked_results
    }