import base64
import os
from runwayml import Client, TaskFailedError
from dotenv import load_dotenv

load_dotenv()

client = Client(api_key=os.getenv("RUNWAYML_API_SECRET"))

def generate_video_from_image(image_bytes: bytes, prompt_text: str = "Apply natural makeup"):
    prompt_text = prompt_text.strip()
    if len(prompt_text) > 1000:
        prompt_text = prompt_text[:1000]

    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_str}"

    try:
        task = client.image_to_video.create(
            model="gen4_turbo",
            prompt_image=data_uri,
            prompt_text=prompt_text,
            ratio="1280:720",
            duration=5
        ).wait_for_task_output()

        return task.output[0]
    except TaskFailedError as e:
        print("The video failed to generate.")
        print(e.task_details)
        return None
    except Exception as e:
        print("Unexpected error:", e)
        return None
