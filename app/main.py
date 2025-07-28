from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import os
from starlette.middleware.sessions import SessionMiddleware
from starlette.config import Config

from app.makeup_recommender import (
    chat_with_beauty_assistant,
    generate_makeup_prompt,
    generate_ai_prompt_with_openai
)
from app.face_analysis import analyze_face
from app.runway_utils import image_to_image as generate_image_from_selfie
from app.describe_makeup import describe_makeup_from_image
# from app import auth, models  # Временно отключаем аутентификацию
from app.database import engine
from app.ingredient_checker import check_ingredients
from . import auth_google
from app.routers.generate_make import router as generate_look_router

# --- Константы и подготовка директорий ---
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- FastAPI и middleware ---
app = FastAPI(
    title="YouGlow API",
    description="API для анализа лица и подбора макияжа",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  
        "http://localhost:3000",
        "https://www.glowguide.live",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session middleware ---
config = Config(".env")
app.add_middleware(SessionMiddleware, secret_key=config("SECRET_KEY"))

# # --- Google OAuth ---
# oauth = OAuth(config)
# oauth.register(
#     name="google",
#     client_id=config("GOOGLE_CLIENT_ID"),
#     client_secret=config("GOOGLE_CLIENT_SECRET"),
#     server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
#     client_kwargs={"scope": "openid email profile"},
# )

# @app.get("/login/google")
# async def login(request: Request):
#     redirect_uri = request.url_for("auth_google_callback")
#     return await oauth.google.authorize_redirect(request, redirect_uri)

# @app.get("/auth/google/callback")
# async def auth_google_callback(request: Request):
#     token = await oauth.google.authorize_access_token(request)
#     user_info = await oauth.google.parse_id_token(request, token)
#     return {"email": user_info["email"], "name": user_info["name"]}

# --- Pydantic модели ---
class IngredientNote(BaseModel):
    name: str
    note: str

class IngredientCheckRequest(BaseModel):
    input_text: str 

class IngredientCheckResponse(BaseModel):
    comedogenic: list[IngredientNote]
    safe: list[IngredientNote]
    unknown: list[IngredientNote]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class TryOnResponse(BaseModel):
    image_url: str
    prompt_used: str

# --- Приветствие ---
@app.get("/")
def root():
    return {"message": "YouGlow API is running!!!"}

# --- Chat endpoint ---
@app.post("/beauty-chat", response_model=ChatResponse)
async def beauty_chat(request: ChatRequest):
    reply = chat_with_beauty_assistant(request.message)
    return ChatResponse(reply=reply)

# --- Makeup Recommendation ---
@app.post("/makeup-recommendation/")
async def makeup_recommendation(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    analysis = analyze_face(image_bytes)

    if "error" in analysis:
        return {"error": analysis["error"]}

    face_data = {
        "skin_tone": analysis.get("skin_tone", "unknown"),
        "average_rgb": analysis.get("average_rgb", [0, 0, 0]),
        "face_shape": analysis.get("face_shape", "unknown"),
        "eye_distance": analysis.get("eye_distance", "medium"),
    }

    main_prompt = generate_makeup_prompt(face_data)
    steps = [] 

    image_url = generate_image_from_selfie(image_bytes, prompt_text=main_prompt)

    if not image_url:
        return {"error": "Failed to generate image from Runway"}

    return {
        "filename": file.filename,
        "saved_as": filename,
        "face_data": face_data,
        "recommendation": main_prompt,
        "steps": steps,
        "image_url": image_url
    }

# --- Try On ---
@app.post("/try-on", response_model=TryOnResponse)
async def try_on(user_photo: UploadFile = File(...), makeup_reference: UploadFile = File(...)):
    user_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{user_photo.filename}")
    ref_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{makeup_reference.filename}")

    with open(user_path, "wb") as f:
        shutil.copyfileobj(user_photo.file, f)
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(makeup_reference.file, f)

    with open(user_path, "rb") as f:
        image_bytes = f.read()

    generated_prompt = describe_makeup_from_image(ref_path)
    image_url = generate_image_from_selfie(image_bytes, prompt_text=generated_prompt)

    if not image_url:
        return {
            "image_url": "https://runway.fake.image/failed.jpg",
            "prompt_used": generated_prompt
        }

    return {
        "image_url": image_url,
        "prompt_used": generated_prompt
    }

# --- Ingredient Checker ---
@app.post("/check-ingredients", response_model=IngredientCheckResponse)
async def check_ingredients_endpoint(request: IngredientCheckRequest):
    result = check_ingredients(request.input_text)
    return IngredientCheckResponse(
        comedogenic=[IngredientNote(**item) for item in result.get("comedogenic", [])],
        safe=[IngredientNote(**item) for item in result.get("safe", [])],
        unknown=[IngredientNote(**item) for item in result.get("unknown", [])]
    )

app.include_router(generate_look_router)

# Временно отключаем аутентификацию для быстрого запуска
# from app.auth import router as auth_router
# app.include_router(auth_router)  

# from app.routers.user_profile import router as user_router
# app.include_router(user_router)

# from app.database import engine
# from app import models

# models.Base.metadata.create_all(bind=engine)
