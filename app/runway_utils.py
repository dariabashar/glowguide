import os
import io
import base64
from PIL import Image, ImageOps
from dotenv import load_dotenv
from runwayml import RunwayML, TaskFailedError

load_dotenv()
RUNWAY_API_TOKEN = os.getenv("RUNWAYML_API_SECRET")

# Инициализируем клиента Runway
client = RunwayML(api_key=RUNWAY_API_TOKEN)

# Подготовка изображения (с пэддингом по соотношению сторон)
def prepare_image_for_runway(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    aspect_ratio = width / height
    if 0.5 <= aspect_ratio <= 2.0:
        return image_bytes
    new_width = width
    new_height = height
    if aspect_ratio < 0.5:
        new_width = int(height * 0.5)
    elif aspect_ratio > 2.0:
        new_height = int(width / 2.0)
    padded = ImageOps.pad(image, (new_width, new_height), color=(255, 255, 255))
    buf = io.BytesIO()
    padded.save(buf, format="JPEG")
    return buf.getvalue()

# Основная функция генерации изображения через text_to_image с reference
def image_to_image(image_bytes: bytes, prompt_text: str = "natural makeup") -> str:
    try:
        prepared_image = prepare_image_for_runway(image_bytes)
        b64_image = base64.b64encode(prepared_image).decode("utf-8")
        image_data_uri = f"data:image/jpeg;base64,{b64_image}"

        task = client.text_to_image.create(
            model="gen4_image",
            ratio="1920:1080",  # ✅ ОБЯЗАТЕЛЬНЫЙ параметр
            prompt_text=prompt_text,
            reference_images=[{
                "uri": image_data_uri,
                "tag": "user_selfie"
            }],
        ).wait_for_task_output()

        print("✅ Runway image generated")
        return task.output[0]  # URL изображения
    except TaskFailedError as e:
        print("❌ The image failed to generate.")
        print(e.task_details)
        return None
