import mediapipe as mp
import cv2
from PIL import Image
import numpy as np
import colorsys
import io

def classify_skin_tone(luminance: float) -> str:
    if luminance < 0.3:
        return "deep"
    elif luminance < 0.6:
        return "medium"
    else:
        return "light"

def classify_undertone(r, g, b) -> str:
    r_norm, g_norm, b_norm = [x / 255.0 for x in (r, g, b)]
    h, _, _ = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
    hue_degrees = int(h * 360)

    if 0 <= hue_degrees <= 30 or 330 <= hue_degrees <= 360:
        return "warm"
    elif 180 <= hue_degrees <= 300:
        return "cool"
    else:
        return "neutral"

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_face_shape(landmarks: list, image_width: int, image_height: int) -> str:
    def px(i):
        x, y, _ = landmarks[i]
        return (x * image_width, y * image_height)

    top = px(10)
    chin = px(152)
    left_cheek = px(234)
    right_cheek = px(454)
    left_jaw = px(130)
    right_jaw = px(359)
    left_forehead = px(127)
    right_forehead = px(356)

    face_length = euclidean(top, chin)
    forehead_width = euclidean(left_forehead, right_forehead)
    cheekbone_width = euclidean(left_cheek, right_cheek)
    jaw_width = euclidean(left_jaw, right_jaw)

    ratio_length_to_width = face_length / cheekbone_width
    jaw_to_cheek = jaw_width / cheekbone_width
    forehead_to_jaw = forehead_width / jaw_width

    if ratio_length_to_width >= 1.5 and abs(forehead_to_jaw - 1) < 0.15:
        return "long"
    elif abs(cheekbone_width - jaw_width) < 20 and abs(forehead_width - jaw_width) < 20:
        return "square"
    elif forehead_width > jaw_width and forehead_to_jaw > 1.2:
        return "heart"
    elif cheekbone_width > forehead_width and cheekbone_width > jaw_width and forehead_to_jaw < 0.9:
        return "diamond"
    elif ratio_length_to_width < 1.1:
        return "round"
    else:
        return "oval"

def analyze_face(image_bytes: bytes) -> dict:
    results_data = {}
    mp_face_mesh = mp.solutions.face_mesh

    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not load image"}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {"error": "No face detected"}

        face_landmarks = results.multi_face_landmarks[0]
        coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
        results_data['landmarks'] = coords 

        try:
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
            results_data["average_rgb"] = [int(r), int(g), int(b)]

            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            results_data["skin_tone"] = classify_skin_tone(luminance)
            results_data["undertone"] = classify_undertone(r, g, b)

        except Exception as e:
            results_data["skin_tone"] = "unknown"
            results_data["undertone"] = "unknown"
            results_data["error"] = f"Color analysis failed: {str(e)}"

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

        results_data["face_shape"] = detect_face_shape(
            coords, image_width=image.shape[1], image_height=image.shape[0]
        )

    return results_data

