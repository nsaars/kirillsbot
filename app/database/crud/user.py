from models.users import User
from database.create_connection import session


def get_user_by_telegram_id(telegram_id: int) -> User:
    return session.query(User).filter(User.telegram_id == telegram_id).first()


def get_user_by_id(user_id: int) -> User:
    return session.query(User).filter(User.id == user_id).first()


def create_user(telegram_id: int, telegram_username: str, telegram_name: str):
    user = get_user_by_telegram_id(telegram_id=telegram_id)
    if not user:
        new_user = User(telegram_id=telegram_id, telegram_username=telegram_username, telegram_name=telegram_name)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        return new_user
    return user
