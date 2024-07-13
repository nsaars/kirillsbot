from aiogram import Router, F
from aiogram.filters import CommandStart, StateFilter

from handlers.start_handlers import start_handler


# Import your handle

def prepare_router() -> Router:

    router = Router()

    router.message.register(start_handler, CommandStart())

    return router
