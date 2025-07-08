from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import os

from app.face_analysis import analyze_face
from app.makeup_recommender import generate_makeup_prompt, generate_ai_prompt_with_openai
from app.runway_utils import generate_video_from_image
from app.describe_makeup import describe_makeup_from_image
from app import auth, models
from app.database import engine
from . import auth_google

UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI( 
    title="YouGlow API",
    description="API для анализа лица и подбора макияжа",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)
app.include_router(auth.router)
app.include_router(auth_google.router)

@app.get("/")
def root():
    return {"message": "YouGlow API is running!!!"}

@app.post("/makeup-recommendation/")
async def makeup_recommendation(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Сохраняем файл
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Читаем байты (❗️)
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    analysis = analyze_face(image_bytes)  # ✅ теперь всё работает

    if "error" in analysis:
        return {"error": analysis["error"]}

    face_data = {
        "skin_tone": analysis.get("skin_tone", "unknown"),
        "average_rgb": analysis.get("average_rgb", [0, 0, 0]),
        "face_shape": analysis.get("face_shape", "unknown"),
        "eye_distance": analysis.get("eye_distance", "medium"),
    }

    main_prompt = generate_makeup_prompt(face_data)
    steps = []  # пока пусто


    video_url = generate_video_from_image(image_bytes, prompt_text=main_prompt)

    if not video_url:
        return {"error": "Failed to generate video from Runway"}

    return {
        "filename": file.filename,
        "saved_as": filename,
        "face_data": face_data,
        "recommendation": main_prompt,
        "steps": steps,
        "video_url": video_url
    }


class GenerationResponse(BaseModel):
    video_url: str

@app.post("/try-on", response_model=GenerationResponse)
async def try_on(
    user_photo: UploadFile = File(...),
    makeup_reference: UploadFile = File(...)
):
    user_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{user_photo.filename}")
    ref_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{makeup_reference.filename}")

    with open(user_path, "wb") as f:
        shutil.copyfileobj(user_photo.file, f)
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(makeup_reference.file, f)

    with open(user_path, "rb") as f:
        image_bytes = f.read()

    generated_prompt = describe_makeup_from_image(ref_path)
    video_url = generate_video_from_image(image_bytes, prompt_text=generated_prompt)

    if not video_url:
        return {"video_url": "https://runway.fake.video/failed.mp4"}

    return {"video_url": video_url}

# @app.post("/generate-makeup")
# async def generate_makeup_endpoint(file: UploadFile = File(...)):
#     face_data = analyze_face(await file.read())
#     if not face_data:
#         return {"error": "No face detected"}

#     base_prompt, steps = generate_makeup_prompt(face_data)
#     enhanced_prompt = generate_ai_prompt_with_openai(base_prompt)

#     return {
#         "face_data": face_data,
#         "enhanced_prompt": enhanced_prompt,
#         "steps": steps
#     }
from fastapi import FastAPI, UploadFile, File

@app.post("/generate-makeup")
async def generate_makeup_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()

    face_data = analyze_face(image_bytes)
    if not face_data:
        return {"error": "No face detected"}

    base_prompt, steps = generate_makeup_prompt(face_data)
    enhanced_prompt = generate_ai_prompt_with_openai(base_prompt)

    # 🎥 Генерация видео
    video_url = generate_video_from_image(image_bytes, prompt_text=enhanced_prompt)

    return {
        "face_data": face_data,
        "enhanced_prompt": enhanced_prompt,
        "steps": steps,
        "video_url": video_url  # <-- Вот она ссылка
    }
