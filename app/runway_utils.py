import base64
import os
import io
from PIL import Image, ImageOps
from runwayml import Client, TaskFailedError
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Инициализируем клиента Runway
client = Client(api_key=os.getenv("RUNWAYML_API_SECRET"))

def prepare_image_for_runway(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Ограничения Runway: соотношение от 0.5 до 2.0
    width, height = image.size
    aspect_ratio = width / height

    if 0.5 <= aspect_ratio <= 2.0:
        return image_bytes  # ОК, возвращаем как есть

    # Если не ОК — паддим до допустимого соотношения
    new_width = width
    new_height = height

    if aspect_ratio < 0.5:
        # Слишком "высокое", добавим боковые отступы
        new_width = int(height * 0.5)
    elif aspect_ratio > 2.0:
        # Слишком "широкое", добавим отступы сверху и снизу
        new_height = int(width / 2.0)

    padded = ImageOps.pad(image, (new_width, new_height), color=(255, 255, 255))
    buf = io.BytesIO()
    padded.save(buf, format="JPEG")
    return buf.getvalue()

def generate_video_from_image(image_bytes: bytes, prompt_text: str = "Apply natural makeup"):
    # Подготовка изображения: паддинг при необходимости
    image_bytes = prepare_image_for_runway(image_bytes)

    prompt_text = prompt_text.strip()
    if len(prompt_text) > 1000:
        prompt_text = prompt_text[:1000]

    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_str}"

    print("\n=== RUNWAY DEBUG ===")
    print("Prompt:", prompt_text[:300])
    print("Image size (bytes):", len(image_bytes))

    try:
        task = client.image_to_video.create(
            model="gen4_turbo",
            prompt_image=data_uri,
            prompt_text=prompt_text,
            ratio="1280:720",
            duration=5
        ).wait_for_task_output()

        print("✅ Runway video generation success!")
        return task.output[0]

    except TaskFailedError as e:
        print("❌ The video failed to generate.")
        print("Runway task details:", e.task_details)
        return None

    except Exception as e:
        print("❌ Unexpected error during video generation:")
        print(e)
        return None
