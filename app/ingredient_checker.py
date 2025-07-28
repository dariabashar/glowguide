import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
Ты — эксперт по косметическим ингредиентам. 
Раздели список ингредиентов на три группы:
1. Комедогенные
2. Безопасные
3. Неизвестные или требующие осторожности

Добавь пояснение к каждому ингредиенту.
Выводи строго в JSON-формате:

{
  "comedogenic": [{"name": "название", "note": "пояснение"}],
  "safe": [{"name": "название", "note": "пояснение"}],
  "unknown": [{"name": "название", "note": "пояснение"}]
}
"""

def check_ingredients(input_text: str):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Вот список ингредиентов:\n{input_text.strip()}"}
            ]
        )
        content = response.choices[0].message.content

        return json.loads(content)

    except Exception as e:
        print("❌ Ошибка при работе с OpenAI:", e)
        return {
            "comedogenic": [],
            "safe": [],
            "unknown": [str(e)]
        }
