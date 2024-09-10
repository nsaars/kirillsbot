import asyncio

import orjson
from aiogram import Bot, Dispatcher
from aiogram.client.bot import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

import handlers
from data import config
from database.crud.state import get_all_states
from database.crud.user import get_all_users


def setup_handlers(dp: Dispatcher) -> None:
    dp.include_router(handlers.prepare_router())


def setup_middlewares(dp: Dispatcher) -> None:
    pass


async def setup_aiogram(dp: Dispatcher) -> None:
    setup_handlers(dp)
    setup_middlewares(dp)
    await set_all_states(dp)


async def aiogram_on_startup_polling(dispatcher: Dispatcher, bot: Bot) -> None:
    await setup_aiogram(dispatcher)


async def aiogram_on_shutdown_polling(dispatcher: Dispatcher, bot: Bot) -> None:
    await bot.session.close()
    await dispatcher.storage.close()



async def set_all_states(dp): #  temp function
    states = get_all_states()
    users = get_all_users()
    for state in states:
        for user_ in users:
            if user_.id == state.user_id:
                user = user_
                break
        else:
            continue
        user_storage_key = StorageKey(bot.id, user.telegram_id, user.telegram_id)
        user_state = FSMContext(storage=dp.storage,
                                key=user_storage_key)
        await user_state.set_state(state.title)
        await user_state.set_data(state.data)


if __name__ == "__main__":
    session = AiohttpSession(
        json_loads=orjson.loads,
    )

    bot = Bot(
        token=config.BOT_TOKEN,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )

    storage = MemoryStorage()

    dp = Dispatcher(
        storage=storage,
    )

    dp.startup.register(aiogram_on_startup_polling)
    dp.shutdown.register(aiogram_on_shutdown_polling)
    asyncio.run(dp.start_polling(bot))
