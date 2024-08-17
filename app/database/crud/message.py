from models.messages import Message
from database.create_connection import session


def create_message(user_id: int, role: str, text: str, response_type: str = None):
    new_message = Message(user_id=user_id, role=role, text=text, response_type=response_type)
    session.add(new_message)
    session.commit()
    session.refresh(new_message)  # нужен ли он тут?
    return new_message
