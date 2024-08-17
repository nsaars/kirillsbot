from datetime import datetime
from database.create_connection import session

from models.consultations import Consultation


def create_consultation(user_id: int, summary: str, date: datetime):
    new_consultation = Consultation(user_id=user_id, summary=summary, date=date)
    session.add(new_consultation)
    session.commit()
    session.refresh(new_consultation)  # нужен ли он тут?
    return new_consultation
