from fastapi import APIRouter, UploadFile, File, Depends
from .face_analysis import analyze_face
from .makeup_recommender import generate_makeup_prompts
from .runway_utils import generate_video_from_image

router = APIRouter()

@router.post("/generate-makeup")
async def generate_makeup(file: UploadFile = File(...)):
    face_data = await analyze_face(file)
    main_prompt, step_prompts = generate_makeup_prompts(face_data)

    main_video_url = await generate_video_from_image(prompt=main_prompt)

    steps = []
    for step in step_prompts:
        video_url = await generate_video_from_image(prompt=step["description"])
        steps.append({**step, "video_url": video_url})

        return {
            "main_video": main_video_url,
            "steps": steps,
            "face_info": face_data
        }
