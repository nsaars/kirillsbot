from aiogram import types

from utils.ai_assistant.ai_assistant import ai


async def ai_conversation_handler(message: types.Message):
    await message.answer(ai.get_response(message.text))
