from aiogram import types
from aiogram.fsm.context import FSMContext
from datetime import datetime
from database.user import create_user

from data.config import ADMIN

from utils.google_docs import add_text_to_document

from data.config import DOCUMENT_ID


async def start_handler(message: types.Message, state: FSMContext):
    from_user = message.from_user

    await message.answer(f"Здравствуйте, {from_user.full_name}, наш менеджер уже получил заявку и скоро свяжется с Вами!")
    await message.answer("Пока что можете задавать вопросы нашему ии чат боту. Просто напишите свой вопрос.")
    await state.set_state('ai_conversation')
    await message.bot.send_message(ADMIN, f"Новая заявка от @{from_user.username} ({from_user.full_name}).")

    create_user(from_user.id, from_user.username, from_user.full_name)
    add_text_to_document(DOCUMENT_ID, f"Заявка: @{from_user.username} ({from_user.full_name})."
                                      f" Дата подачи заявки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                         index=None, service=None)