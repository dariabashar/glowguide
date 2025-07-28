from openai import OpenAI
import os
from dotenv import load_dotenv
import os
import cv2
import io
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import mediapipe as mp

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ai_prompt_with_openai(user_prompt: str) -> str:
    """
    Обращается к GPT-4, чтобы сгенерировать продвинутый текст промпта по описанию макияжа.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional makeup artist telling an AI video model create makeup looks with details."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8,
    )

    return response.choices[0].message.content

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mp_face_mesh = mp.solutions.face_mesh

# ---------------- FACE ANALYSIS ---------------- #

def analyze_face(image_bytes: bytes) -> dict:
    """
    Анализирует лицо по изображению и извлекает черты: форма глаз, губ, бровей, подтон.
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return {}

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        # Eye shape
        eye_ratio = np.linalg.norm([
            (landmarks[133].x - landmarks[33].x),
            (landmarks[133].y - landmarks[33].y)
        ]) / (np.linalg.norm([
            (landmarks[159].y - landmarks[145].y),
            (landmarks[159].x - landmarks[145].x)
        ]) + 1e-6)

        eye_shape = "almond" if 2.5 < eye_ratio < 4 else ("round" if eye_ratio <= 2.5 else "monolid")

        # Undertone (по цвету щеки)
        cx = int(landmarks[234].x * w)
        cy = int(landmarks[234].y * h)
        cheek_area = image[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5]
        avg_color = np.mean(cheek_area, axis=(0, 1)) if cheek_area.size else [128, 128, 128]
        b, g, r = avg_color
        if r > g and r > b:
            undertone = "warm"
        elif b > r and b > g:
            undertone = "cool"
        else:
            undertone = "neutral"

        # Lip shape
        lip_width = np.linalg.norm([
            landmarks[291].x - landmarks[61].x,
            landmarks[291].y - landmarks[61].y
        ])
        lip_height = np.linalg.norm([
            landmarks[13].y - landmarks[14].y,
            landmarks[13].x - landmarks[14].x
        ])
        lip_shape = "heart" if lip_width / (lip_height + 1e-6) < 2 else "full"

        # Brow shape
        brow_slope = landmarks[65].y - landmarks[55].y
        brow_shape = "arched" if brow_slope < -0.02 else "straight"

        return {
            "eye_shape": eye_shape,
            "skin_type": "normal",  
            "undertone": undertone,
            "lip_shape": lip_shape,
            "brow_shape": brow_shape
        }

# ---------------- PROMPT GENERATION ---------------- #

def generate_makeup_prompt(face_data: dict) -> str:
    """
    Генерирует насыщенный промпт для Runway ML с учётом черт лица.
    """
    undertone = face_data.get("undertone", "neutral")
    eye_shape = face_data.get("eye_shape", "almond")
    skin_type = face_data.get("skin_type", "normal")
    lip_shape = face_data.get("lip_shape", "heart")
    brow_shape = face_data.get("brow_shape", "arched")

    return (
        f"Create a clean and polished everyday makeup look for a woman with {eye_shape}-shaped eyes, {lip_shape} lips, "
        f"{brow_shape} eyebrows, {skin_type} skin, and a {undertone} undertone.\n"
        "Start with a radiant medium-coverage foundation matching her skin tone. Add subtle creamy contour to define cheekbones and jawline. "
        "Apply soft peach blush to the cheeks and a golden highlighter to the high points of the face.\n"
        "Use warm-toned brown eyeshadow blended softly in the crease, and shimmery champagne on the eyelids. "
        "Add precise black eyeliner with a small wing and define lashes with lengthening mascara.\n"
        "Shape brows with a light brown pencil following the natural arch. For lips, apply a matte rose-pink lipstick with a satin finish.\n"
        "Makeup should be clearly visible and enhance natural beauty. Avoid editorial or overly glam styles. "
        "Background must be neutral, lighting should be soft and even, and the full face must be centered in the frame."
    )

BEAUTY_CHAT_SYSTEM_PROMPT = """
Ты — профессиональный визажист и консультант по уходу за кожей. Отвечай понятно, дружелюбно и по существу. 
Можешь давать советы по макияжу, выбору оттенков, типу кожи, базовому уходу и подбору продуктов. 
Не давай медицинских рекомендаций и не советуй препараты.
"""

def chat_with_beauty_assistant(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": BEAUTY_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
