from database.create_session import session_scope
from models.users import User


def get_user_by_telegram_id(db, telegram_id: int) -> bool:
    return db.query(User).filter(User.telegram_id == telegram_id).first()


def create_user(telegram_id: int, telegram_username: str, telegram_name: str):
    with session_scope() as db:
        if not get_user_by_telegram_id(db, telegram_id=telegram_id):
            new_user = User(telegram_id=telegram_id, telegram_username=telegram_username, telegram_name=telegram_name)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            return new_user
