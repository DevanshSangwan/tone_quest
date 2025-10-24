# leaderboard.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import redis
from typing import List, Dict, Any

router = APIRouter()

# Read from environment variables, with defaults for local dev
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
# --- Redis Connection ---
# Make sure Redis server is running locally (port 6379)
r = redis.Redis(host='localhost', port=REDIS_PORT, db=0, decode_responses=True)

LEADERBOARD_KEY = "tonequest_leaderboard"

# --- Request Models ---
class ScoreUpdate(BaseModel):
    user_id: str
    delta: float  # points to add/subtract

# --- Endpoints ---
@router.post("/submit_score")
def submit_score(data: ScoreUpdate):
    try:
        # Increment (or decrement) user's score in sorted set
        r.zincrby(LEADERBOARD_KEY, data.delta, data.user_id)
        new_score = r.zscore(LEADERBOARD_KEY, data.user_id)

        # zscore may return None or a string (because decode_responses=True)
        if new_score is None:
            raise HTTPException(status_code=404, detail="User not found after update (unexpected)")

        new_score = float(new_score)
        return {"message": "Score updated", "user_id": data.user_id, "new_score": round(new_score, 2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard/top/{n}")
def get_top_players(n: int):
    try:
        # Get top n players, highest score first
        results = r.zrevrange(LEADERBOARD_KEY, 0, n - 1, withscores=True)  # list of (user, score)
        return [
            {"rank": idx + 1, "user_id": user, "score": round(float(score), 2)}
            for idx, (user, score) in enumerate(results)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard/user/{user_id}")
def get_user_rank(user_id: str):
    try:
        # Get the user's 0-indexed rank
        rank_0_idx = r.zrevrank(LEADERBOARD_KEY, user_id)
        if rank_0_idx is None:
            raise HTTPException(status_code=404, detail="User not found in leaderboard")
        
        score = r.zscore(LEADERBOARD_KEY, user_id)
        if score is None:
            # This should be impossible if rank exists, but good to check
            raise HTTPException(status_code=404, detail="User score not found")
        score = float(score)

        # --- Updated Logic for 25 above / 24 below ---
        # Spec: 25 above + 1 (self) + 24 below = 50 players total
        # We use the 0-indexed rank to calculate the range.
        start = max(rank_0_idx - 25, 0)
        end = rank_0_idx + 24
        # --- End of Updated Logic ---

        # Fetch the players in that range
        nearby = r.zrevrange(LEADERBOARD_KEY, start, end, withscores=True)
        
        nearby_players = []
        for i, (u, s) in enumerate(nearby):
            # The absolute rank is the starting offset + list index + 1
            nearby_players.append({
                "rank": start + i + 1,
                "user_id": u,
                "score": round(float(s), 2)
            })
            
        return {
            "user_id": user_id,
            "rank": rank_0_idx + 1,  # Convert 0-index to 1-index for display
            "score": round(score, 2),
            "nearby_players": nearby_players
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
