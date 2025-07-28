from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Получаем средний цвет с изображения
def get_average_color_from_image_url(url: str) -> tuple:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    np_img = np.array(image)
    avg_color = np_img.mean(axis=(0, 1))
    return tuple(int(x) for x in avg_color)

# Определяем категорию цвета
def rgb_to_color_label(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    if r > 230 and g > 180 and b < 140:
        return "персиковый"
    elif r > 230 and g > 100 and b > 150:
        return "розовый"
    elif r > 200 and g < 100 and b < 100:
        return "красный"
    elif r > 180 and g > 170 and b > 150:
        return "бежевый"
    return "универсальный"

# Главная функция
def get_color_label_from_image_url(image_url: str) -> dict:
    try:
        rgb = get_average_color_from_image_url(image_url)
        color_label = rgb_to_color_label(rgb)
        return {
            "rgb": rgb,
            "color_label": color_label
        }
    except Exception as e:
        return {"error": str(e)}

def classify_color(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = h * 360

    base = None
    if 0 <= h < 20 or 340 <= h <= 360:
        base = "красный"
    elif 20 <= h < 35:
        base = "персиковый"
    elif 35 <= h < 50:
        base = "коралловый"
    elif 270 <= h < 330:
        base = "розовый"
    elif 50 <= h < 70:
        base = "оранжевый"
    elif 230 <= h < 270:
        base = "лиловый"
    elif s < 0.2 and v > 0.7:
        base = "нюд"

    # Температура
    if h < 180:
        temperature = "тёплый"
    else:
        temperature = "холодный"

    # Насыщенность
    if s < 0.3:
        saturation_label = "приглушённый"
    elif s > 0.6:
        saturation_label = "насыщенный"
    else:
        saturation_label = "средний"

    return f"{temperature} {saturation_label} {base}"
