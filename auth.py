from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from firebase_admin.auth import InvalidIdTokenError, ExpiredIdTokenError

# Import the pre-initialized auth module from your firebase_config
from firebase_config import firebase_auth 

# This tells FastAPI to look for an 'Authorization: Bearer <token>' header.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    A FastAPI dependency that verifies the Firebase ID token
    and returns the user's decoded token (payload).
    
    If the token is invalid or expired, it raises an HTTPException.
    """
    try:
        # Verify the token against the Firebase Auth API
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token
        
    except ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidIdTokenError:
        # This handles tokens that are malformed, have the wrong signature, etc.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Handle other potential errors (e.g., Firebase network issues)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error verifying token: {e}",
        )