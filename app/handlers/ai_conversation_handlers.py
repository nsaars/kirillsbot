from aiogram import types
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext

from utils.ai_assistant.ai_assistant import AiChain


async def ai_conversation_handler(message: types.Message, state: FSMContext):
    if message.content_type != "text":
        await message.answer("Меня настроили отвечать только на текстовые сообщения :)")
        return

    history: list = (await state.get_data()).get('history') or []
    response: str = await AiChain.get_proper_response(message.text, history)
    await state.update_data({'history': history + [('user', message.text), ('assistant', response)]})
    await message.answer(response, parse_mode=ParseMode.MARKDOWN)
