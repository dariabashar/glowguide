from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel
from typing import List

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True


class FoundationSpec(BaseModel):
    tone: str  
    undertone: str  
    coverage: Optional[str] = "medium"

class BlushSpec(BaseModel):
    color: str  
    placement: str
    finish: Optional[str] = "natural"

class EyeSpec(BaseModel):
    shadow_color: str
    liner_style: str
    mascara: bool

class LipSpec(BaseModel):
    color: str
    finish: str

class MakeupSpec(BaseModel):
    foundation: FoundationSpec
    blush: BlushSpec
    eyes: EyeSpec
    lips: LipSpec

class IngredientNote(BaseModel):
    name: str
    note: str

class IngredientCheckRequest(BaseModel):
    input_text: str 

class IngredientCheckResponse(BaseModel):
    comedogenic: list[IngredientNote]
    safe: list[IngredientNote]
    unknown: list[IngredientNote]

class UserUpdate(BaseModel):
    username: str