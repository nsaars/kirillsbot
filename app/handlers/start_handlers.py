from aiogram import types
from aiogram.fsm.context import FSMContext
from datetime import datetime
from database.user import create_user

from data.config import ADMIN

from utils.google_docs import add_text_to_document

from data.config import DOCUMENT_ID


async def start_handler(message: types.Message, state: FSMContext):
    from_user = message.from_user
    create_user(from_user.id, from_user.username, from_user.full_name)
    add_text_to_document(DOCUMENT_ID, f"Заявка: @{from_user.username} ({from_user.full_name})."
                                      f" Дата подачи заявки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                         index=None, service=None)
    await message.answer("Наш менеджер уже получил заявку и скоро свяжется с Вами!")
    await message.bot.send_message(ADMIN, f"Новая заявка от @{from_user.username} ({from_user.full_name}).")
