from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from app.schemas import MakeupSpec

from app.face_analysis import analyze_face
from app.services.makeup_spec_ai import generate_makeup_spec
from app.services.prompt_builder import build_prompt_from_spec
from app.runway_utils import image_to_image

router = APIRouter()

@router.post("/generate-look")
async def generate_ideal_makeup(
    image: UploadFile = File(...),
    lang: Optional[str] = Form("en")
):
    try:
        image_bytes = await image.read()
        face_data = analyze_face(image_bytes)

        spec_dict = generate_makeup_spec(face_data)
        spec = MakeupSpec.parse_obj(spec_dict)
        prompt = build_prompt_from_spec(spec)

        image_url = image_to_image(image_bytes, prompt)

        return {
            "image_url": image_url,
            "prompt": prompt,
            "spec": spec_dict
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
