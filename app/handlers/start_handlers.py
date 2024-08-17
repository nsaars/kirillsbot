from aiogram import types
from aiogram.fsm.context import FSMContext

from database.crud.state import create_state, update_state
from database.crud.user import create_user

from data.config import ADMIN


async def start_handler(message: types.Message, state: FSMContext):
    from_user = message.from_user

    db_user = create_user(from_user.id, from_user.username, from_user.full_name)
    db_state = create_state(db_user.id, await state.get_state(), await state.get_data())

    greeting_text = f"Здравствуйте, {from_user.full_name}, я виртуальный ассистент компании Quasar79.\n" \
                    f"Расскажите, пожалуйста, для чего вам нужен ИИ ассистент/чат бот?\n\n" \
                    f"Например: 'Мне нужен чат бот для моего отеля, чтоб он отвечал клиентам на вопросы и бронировал " \
                    f"для них номера.' "

    await message.answer(greeting_text)

    await state.update_data({'history': [('assistant', greeting_text)], 'db_state_id': db_state.id,
                             'db_user_id': db_user.id})
    await state.set_state('ai_conversation')
    update_state(db_state.id, {'title': 'ai_conversation', 'data': await state.get_data()})  # todo: custom fsm context
    await message.bot.send_message(ADMIN, f"@{from_user.username} ({from_user.full_name}) написал в бота.")
