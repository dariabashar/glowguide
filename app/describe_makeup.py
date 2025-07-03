from openai import OpenAI
import base64
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def describe_makeup_from_image(image_path: str) -> str:
    with open(image_path, "rb") as img:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a makeup expert. Describe the makeup style on the photo in detail using natural language so it can be used as a prompt for AI makeup generation."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(img.read()).decode()}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
    return response.choices[0].message.content
