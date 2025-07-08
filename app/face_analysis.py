import mediapipe as mp
import cv2
from PIL import Image
import numpy as np
import colorsys
import io

def analyze_face(image_bytes: bytes) -> dict:
    results_data = {}
    mp_face_mesh = mp.solutions.face_mesh

    # Читаем изображение как массив байт → в OpenCV изображение
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        results_data['error'] = "Could not load image"
        return results_data

    # Анализ лица с помощью MediaPipe
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            results_data['landmarks'] = coords

            try:
                # Используем PIL для анализа цвета
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                width, height = image_pil.size
                crop_box = (
                    int(width * 0.45),
                    int(height * 0.4),
                    int(width * 0.55),
                    int(height * 0.5)
                )
                face_region = image_pil.crop(crop_box)
                pixels = np.array(face_region).reshape(-1, 3)

                avg_color = pixels.mean(axis=0)
                r, g, b = avg_color

                r_norm, g_norm, b_norm = [x / 255.0 for x in (r, g, b)]
                h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
                hue_degrees = int(h * 360)

                if 0 <= hue_degrees <= 30 or 330 <= hue_degrees <= 360:
                    tone = "warm"
                elif 180 <= hue_degrees <= 300:
                    tone = "cool"
                else:
                    tone = "neutral"

                results_data['skin_tone'] = tone
                results_data['average_rgb'] = [int(r), int(g), int(b)]
                results_data['hue'] = hue_degrees

            except Exception as e:
                results_data['skin_tone'] = "unknown"
                results_data['error'] = f"Color analysis failed: {str(e)}"

            try:
                left_eye = face_landmarks.landmark[468]
                right_eye = face_landmarks.landmark[473]
                eye_distance = abs(left_eye.x - right_eye.x)

                if eye_distance > 0.15:
                    results_data['eye_distance'] = "wide"
                elif eye_distance < 0.10:
                    results_data['eye_distance'] = "close"
                else:
                    results_data['eye_distance'] = "medium"
            except:
                results_data['eye_distance'] = "unknown"

            results_data['face_shape'] = "oval"

        else:
            results_data['error'] = "No face detected"

    return results_data

