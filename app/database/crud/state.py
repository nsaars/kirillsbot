from database.create_connection import session
from models.states import State


def get_state_by_user_id(user_id: int):
    return session.query(State).filter(State.user_id == user_id).first()


def create_state(user_id: int, title: str, data: dict):
    state = get_state_by_user_id(user_id=user_id)
    if not state:
        new_state = State(user_id=user_id, title=title, data=data)
        session.add(new_state)
        session.commit()
        session.refresh(new_state)  # нужен ли он тут?
        return new_state
    return state


def update_state(state_id: int, update_data: dict) -> State:
    state = session.query(State).filter_by(id=state_id).one()

    for key, value in update_data.items():
        if hasattr(state, key):
            setattr(state, key, value)
    session.commit()
    session.refresh(state)  # нужен ли он тут?
    return state
