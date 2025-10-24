import json
from fastapi import FastAPI, HTTPException, Depends  # Added Depends and HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from leaderboard import router as leaderboard_router
from firebase_config import db
from typing import Optional, Dict, Any
import time

# --- New Imports for Auth & Caching ---
from auth import get_current_user  # Import the auth dependency
from cachetools import TTLCache     # Import a thread-safe cache
from threading import Lock          # For thread-safe cache access

app = FastAPI()
app.include_router(leaderboard_router)

# Load model once
model = SentenceTransformer('all-mpnet-base-v2')

# --- Thread-Safe Cache Setup ---
CACHE_TTL = 300  # 5 minutes cache TTL
# maxsize=100 (100 questions), ttl=300 (5 seconds)
question_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)
question_cache_lock = Lock()  # Lock to ensure thread-safe operations

# --- Data Loading ---

def load_reference_data():
    """Loads all question headers from Firestore."""
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
    """Load a specific question and compute its embeddings on-demand (thread-safe)"""
    
    if use_cache:
        # Check cache (thread-safe)
        with question_cache_lock:
            cached_question = question_cache.get(question_id)
        if cached_question:
            return cached_question
            
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
    try:
        question["embeddings"] = model.encode(question["answers"], convert_to_tensor=True)
    except Exception:
        question["embeddings"] = [] # Handle questions with no answers
    
    # Cache the result if caching is enabled (thread-safe)
    if use_cache:
        with question_cache_lock:
            question_cache[question_id] = question
            
    return question

# --- Input Schema ---

class UserAnswer(BaseModel):
    question_id: int
    answer_text: str
    force_refresh: Optional[bool] = False  # Option to bypass cache

# --- Protected Endpoint ---

@app.post("/evaluate_answer")
def evaluate_answer(
    data: UserAnswer,
    # This is the dependency that protects the endpoint.
    # If the token is invalid, it will stop here and return a 401 error.
    current_user: dict = Depends(get_current_user)
):
    try:
        # You now have the user's info!
        user_id = current_user["uid"]
        user_email = current_user.get("email")
        print(f"Authenticated user {user_id} ({user_email}) is submitting an answer.")

        # Load question with embeddings on-demand
        question = get_question_with_embeddings(data.question_id, use_cache=not data.force_refresh)
        
        if not question:
            raise HTTPException(status_code=404, detail=f"Question ID {data.question_id} not found.")
        
        # This is the corrected check.
        # We just need to check if the length of the embeddings list/tensor is zero.
        if len(question.get("embeddings", [])) == 0:
            raise HTTPException(status_code=500, detail=f"Question ID {data.question_id} has no reference answers.")

        # Encode the user's answer
        user_emb = model.encode(data.answer_text, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarities = util.cos_sim(user_emb, question["embeddings"])[0].tolist()
        
        # Find best match
        best_score = max(similarities)
        best_index = similarities.index(best_score)

        # --- THIS IS THE MISSING PIECE ---
        # Now you can connect to your other services
        
        # 1. TODO: Save to your (PostgreSQL/Firestore) database
        #    save_answer_to_db(user_id, data.question_id, best_score)
        
        # 2. TODO: Get the 'delta' and update the Redis leaderboard
        #    old_score = get_old_score_from_db(user_id, data.question_id)
        #    score_delta = best_score - old_score
        #    # r.zincrby(LEADERBOARD_KEY, score_delta, user_id)
        
        return {
            "question_id": data.question_id,
            "question": question["question"],
            "best_match_sample": question["answers"][best_index],
            "similarity_score": round(best_score, 3),
            "user_who_submitted": user_id,  # Include the user ID in the response
            "all_scores": [
                {"sample": ans, "score": round(score, 3)}
                for ans, score in zip(question["answers"], similarities)
            ]
        }
    except Exception as e:
        # Use HTTPException for proper error responses
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Utility Endpoints (Updated for new cache) ---

@app.get("/cache/stats")
def get_cache_stats():
    """Get cache statistics"""
    with question_cache_lock:
        return {
            "cached_questions": question_cache.currsize,
            "max_questions": question_cache.maxsize,
            "cache_ttl_seconds": question_cache.ttl,
            "cached_question_ids": list(question_cache.keys())
        }

@app.delete("/cache/clear")
def clear_cache():
    """Clear all cached questions"""
    with question_cache_lock:
        question_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.delete("/cache/question/{question_id}")
def clear_question_cache(question_id: int):
    """Clear cache for a specific question"""
    with question_cache_lock:
        if question_id in question_cache:
            del question_cache[question_id]
            return {"message": f"Cache cleared for question {question_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found in cache")

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
        raise HTTPException(status_code=500, detail=str(e))