from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_makeup_prompts(face_data: dict) -> tuple[str, list[dict]]:
    """
    Принимает данные лица, генерирует промпт для Runway и пошаговые инструкции.
    """
    undertone = face_data.get("undertone", "neutral")
    eye_shape = face_data.get("eye_shape", "almond")
    skin_type = face_data.get("skin_type", "normal")
    lip_shape = face_data.get("lip_shape", "heart")
    brow_shape = face_data.get("brow_shape", "arched")

    main_prompt = (
        f"Create a flattering makeup look for a user with {eye_shape} eyes, {undertone} undertone, {skin_type} skin, "
        f"{lip_shape} lips, and {brow_shape} brows. Use radiant foundation, soft contouring, peach blush, "
        f"gold-brown smokey eyeshadow, defined eyeliner, and nude glossy lips. The look should enhance natural beauty "
        "while adding elegance and freshness. Keep the lighting soft and the background neutral."
    )

    step_prompts = [
        {"title": "Step 1: Skin Prep", "description": "Cleanse and moisturize the skin. Apply a lightweight primer for smooth texture."},
        {"title": "Step 2: Foundation", "description": "Apply radiant foundation evenly using a beauty blender."},
        {"title": "Step 3: Eyes", "description": "Use gold and brown shimmer eyeshadows for a smokey effect. Add eyeliner and mascara."},
        {"title": "Step 4: Cheeks", "description": "Apply peach blush to the apples of the cheeks, blend upwards."},
        {"title": "Step 5: Lips", "description": "Finish with nude glossy lipstick."}
    ]

    return main_prompt, step_prompts

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

import cv2
import mediapipe as mp
import numpy as np
from typing import BinaryIO

mp_face_mesh = mp.solutions.face_mesh

def analyze_face(image_bytes: BinaryIO) -> dict:
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        eye_ratio = np.linalg.norm([
            landmarks[133].x - landmarks[33].x,
            landmarks[133].y - landmarks[33].y
        ]) / (np.linalg.norm([
            landmarks[159].y - landmarks[145].y,
            landmarks[159].x - landmarks[145].x
        ]) + 1e-6)

        eye_shape = "almond" if 2.5 < eye_ratio < 4 else ("round" if eye_ratio <= 2.5 else "monolid")

        cx = int(landmarks[234].x * w)
        cy = int(landmarks[234].y * h)
        cheek_area = image[cy - 5:cy + 5, cx - 5:cx + 5]
        avg_color = np.mean(cheek_area, axis=(0, 1))  # BGR
        b, g, r = avg_color
        if r > g and r > b:
            undertone = "warm"
        elif b > r and b > g:
            undertone = "cool"
        else:
            undertone = "neutral"

        lip_width = np.linalg.norm([
            landmarks[291].x - landmarks[61].x,
            landmarks[291].y - landmarks[61].y
        ])
        lip_height = np.linalg.norm([
            landmarks[13].y - landmarks[14].y,
            landmarks[13].x - landmarks[14].x
        ])
        lip_shape = "heart" if lip_width / (lip_height + 1e-6) < 2 else "full"


        brow_slope = landmarks[65].y - landmarks[55].y
        brow_shape = "arched" if brow_slope < -0.02 else "straight"

        return {
            "eye_shape": eye_shape,
            "skin_type": "normal",  
            "undertone": undertone,
            "lip_shape": lip_shape,
            "brow_shape": brow_shape
        }
