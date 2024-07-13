from types import MappingProxyType
from typing import TypeVar, List, Dict, Union
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackGame,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LoginUrl,
)

from keyboards.keyboard_utils import schema_generator

A = TypeVar("A", bound=CallbackData)


class InlineConstructor:
    aliases = MappingProxyType({"cb": "callback_data"})
    available_properties = (
        "text",
        "callback_data",
        "url",
        "login_url",
        "switch_inline_query",
        "switch_inline_query_current_chat",
        "callback_game",
        "pay",
    )
    properties_amount = 2

    @staticmethod
    def create_kb(
        actions: List[Dict[str, Union[str, bool, A, LoginUrl, CallbackGame]]],
        schema: List[int],
    ) -> InlineKeyboardMarkup:
        btns: List[InlineKeyboardButton] = []

        for action in actions:
            data: Dict[str, Union[str, bool, A, LoginUrl, CallbackGame]] = {}

            for alias, real_name in InlineConstructor.aliases.items():
                if alias in action:
                    action[real_name] = action[alias]
                    del action[alias]

            for key in action:
                if key in InlineConstructor.available_properties:
                    if len(data) < InlineConstructor.properties_amount:
                        data[key] = action[key]
                    else:
                        break

            if "callback_data" in data and isinstance(data["callback_data"], CallbackData):
                data["callback_data"] = data["callback_data"].pack()

            if "pay" in data:
                if btns and data["pay"]:
                    raise ValueError("Платежная кнопка должна идти первой в клавиатуре")
                data["pay"] = action["pay"]

            if len(data) != InlineConstructor.properties_amount:
                raise ValueError("Недостаточно данных для создания кнопки")

            btns.append(InlineKeyboardButton(**data))  # type: ignore

        inline_keyboard = schema_generator.create_keyboard_layout(btns, schema)
        return InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
