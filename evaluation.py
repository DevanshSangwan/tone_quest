import json
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from leaderboard import router as leaderboard_router
from firebase_config import db
from functools import lru_cache
from typing import Optional, Dict, Any
import time

app = FastAPI()

app.include_router(leaderboard_router)

# Load model once
model = SentenceTransformer('all-mpnet-base-v2')

# Simple cache with TTL for questions
question_cache: Dict[int, Dict[str, Any]] = {}
CACHE_TTL = 300  # 5 minutes cache TTL

# Load reference data on-demand
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

def get_question_with_embeddings(question_id: int, use_cache: bool = True):
    """Load a specific question and compute its embeddings on-demand with optional caching"""
    
    # Check cache first if enabled
    if use_cache and question_id in question_cache:
        cached_data = question_cache[question_id]
        # Check if cache is still valid (not expired)
        if time.time() - cached_data["timestamp"] < CACHE_TTL:
            return cached_data["question"]
        else:
            # Remove expired cache entry
            del question_cache[question_id]
    
    # Load from Firestore
    qna_ref = db.collection("QnA").where("id", "==", question_id).stream()
    questions = []
    for doc in qna_ref:
        q = doc.to_dict()
        questions.append({
            "id": q.get("id"),
            "question": q.get("question_text"),
            "answers": q.get("answers", [])
        })
    
    if not questions:
        return None
    
    question = questions[0]
    # Compute embeddings on-demand
    question["embeddings"] = model.encode(question["answers"], convert_to_tensor=True)
    
    # Cache the result if caching is enabled
    if use_cache:
        question_cache[question_id] = {
            "question": question,
            "timestamp": time.time()
        }
    
    return question

# Input schema
class UserAnswer(BaseModel):
    question_id: int
    answer_text: str
    force_refresh: Optional[bool] = False  # Option to bypass cache

@app.post("/evaluate_answer")
def evaluate_answer(data: UserAnswer):
    try:
        # Load question with embeddings on-demand
        # Use force_refresh to bypass cache if requested
        question = get_question_with_embeddings(data.question_id, use_cache=not data.force_refresh)
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

# Additional utility endpoints for cache management
@app.get("/cache/stats")
def get_cache_stats():
    """Get cache statistics"""
    return {
        "cached_questions": len(question_cache),
        "cache_ttl_seconds": CACHE_TTL,
        "cached_question_ids": list(question_cache.keys())
    }

@app.delete("/cache/clear")
def clear_cache():
    """Clear all cached questions"""
    question_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.delete("/cache/question/{question_id}")
def clear_question_cache(question_id: int):
    """Clear cache for a specific question"""
    if question_id in question_cache:
        del question_cache[question_id]
        return {"message": f"Cache cleared for question {question_id}"}
    else:
        return {"message": f"Question {question_id} not found in cache"}

@app.get("/questions/list")
def list_all_questions():
    """List all available questions (without embeddings)"""
    try:
        questions = load_reference_data()
        return {
            "total_questions": len(questions),
            "questions": [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer_count": len(q["answers"])
                }
                for q in questions
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# uvicorn evaluation:app --reload