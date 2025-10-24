import json
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from leaderboard import router as leaderboard_router
from firebase_config import db

app = FastAPI()

app.include_router(leaderboard_router)

# Load model once
model = SentenceTransformer('all-mpnet-base-v2')

# Load reference data
def load_reference_data():
    qna_ref = db.collection("QnA").stream()
    questions = []
    for doc in qna_ref:
        q = doc.to_dict()
        questions.append({
            "id": q.get("id"),
            "question": q.get("question_text"),
            "answers": q.get("answers", [])
        })
    return questions

reference_data = load_reference_data()

# Precompute embeddings for all answers
for q in reference_data:
    q["embeddings"] = model.encode(q["answers"], convert_to_tensor=True)

# Input schema
class UserAnswer(BaseModel):
    question_id: int
    answer_text: str

@app.post("/evaluate_answer")
def evaluate_answer(data: UserAnswer):
    try:
        # Find the question by ID
        question = next((q for q in reference_data if q["id"] == data.question_id), None)
        if not question:
            return {"error": f"Question ID {data.question_id} not found."}

        # Encode the user's answer
        user_emb = model.encode(data.answer_text, convert_to_tensor=True)

        # Compute cosine similarity
        similarities = util.cos_sim(user_emb, question["embeddings"])[0].tolist()

        # Find best match
        best_score = max(similarities)
        best_index = similarities.index(best_score)

        return {
            "question_id": data.question_id,
            "question": question["question"],
            "best_match_sample": question["answers"][best_index],
            "similarity_score": round(best_score, 3),
            "all_scores": [
                {"sample": ans, "score": round(score, 3)}
                for ans, score in zip(question["answers"], similarities)
            ]
        }

    except Exception as e:
        return {"error": str(e)} 

# uvicorn evaluation:app --reload