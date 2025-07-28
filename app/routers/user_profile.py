from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app import models, schemas
from app.auth import get_current_user
from app.database import get_db

router = APIRouter()

@router.put("/users/update")
def update_user_profile(
    user_update: schemas.UserUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Заново загружаем пользователя из текущей сессии:
    user_in_db = db.query(models.User).filter_by(id=current_user.id).first()

    if user_update.username != user_in_db.username:
        existing_user = db.query(models.User).filter_by(username=user_update.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")

    user_in_db.username = user_update.username
    db.commit()
    db.refresh(user_in_db)

    return {"message": "Profile updated successfully"}
