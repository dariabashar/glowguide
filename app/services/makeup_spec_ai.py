from openai import OpenAI
import os
import json
from app.schemas import MakeupSpec
from openai.types.chat import ChatCompletionMessageParam

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
Ты профессиональный визажист. Получи данные о лице клиентки (форма лица, цвет кожи, подтон и т.д.).
Верни JSON с рекомендациями по макияжу, строго в следующем формате:

{
  "foundation": {
    "tone": "...",
    "undertone": "...",
    "coverage": "..."
  },
  "blush": {
    "color": "...",
    "placement": "...",
    "finish": "..."
  },
  "eyes": {
    "shadow_color": "...",
    "liner_style": "...",
    "mascara": true
  },
  "lips": {
    "color": "...",
    "finish": "..."
  }
}

Используй нейтральный стиль. Цвета можно писать словами или hex-кодами. Не добавляй ничего лишнего.
"""

def generate_makeup_spec(face_analysis: dict) -> dict:
    useful_data = {
        "skin_tone": face_analysis.get("skin_tone"),
        "undertone": face_analysis.get("undertone"),
        "face_shape": face_analysis.get("face_shape"),
        "eye_distance": face_analysis.get("eye_distance"),
    }

    user_prompt = f"""
Вот данные о внешности клиентки:
{json.dumps(useful_data, ensure_ascii=False)}

Сгенерируй подходящий макияж в формате JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    reply = response.choices[0].message.content

    try:
        json_start = reply.find('{')
        json_str = reply[json_start:]
        parsed = json.loads(json_str)
        validated = MakeupSpec.parse_obj(parsed)
        return validated.dict()
    except Exception as e:
        return {
            "error": "Ошибка при разборе или валидации ответа",
            "exception": str(e),
            "raw_reply": reply
        }
