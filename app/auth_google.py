import os
import httpx
from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from .database import SessionLocal
from .models import User
from .auth import create_access_token
from fastapi.responses import RedirectResponse

load_dotenv()

router = APIRouter()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/auth/callback")
async def google_auth(request: Request, db: Session = Depends(get_db)):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missin smthng from Google")
    
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri":GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(token_url, data=token_data)
        token_resp.raise_for_status()
        token_json = token_resp.json()
        access_token = token_json.get("access_token")

    user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(user_info_url, headers=headers)
        user_resp.raise_for_status()
        user_json = user_resp.json()
        email = user_json["email"]
        name = user_json.get("name", email.split("@")[0])

    user = db.query(User).filter_by(username=email).first()
    if not user:
        user = User(username=email, hashed_password="google_oauth")
        db.add(user)
        db.commit()
        db.refresh(user)
    
    token = create_access_token(data={"sub": user.username})
    return RedirectResponse(url=f"http://localhost:5174/auth/callback?token={token}")