from collections.abc import Sequence
from types import MappingProxyType
from aiogram.types import KeyboardButton, KeyboardButtonPollType, ReplyKeyboardMarkup
from keyboards.keyboard_utils import schema_generator


class DefaultConstructor:
    aliases = MappingProxyType({
        "contact": "request_contact",
        "location": "request_location",
        "poll": "request_poll",
    })
    available_properties = (
        "text", "request_contact", "request_location", "request_poll",
        "request_user", "request_chat", "web_app",
    )
    properties_amount = 1

    @staticmethod
    def create_kb(
            actions: Sequence[str | dict[str, str | bool | KeyboardButtonPollType]],
            schema: Sequence[int],
            resize_keyboard: bool = True,
            selective: bool = False,
            one_time_keyboard: bool = False,
            is_persistent: bool = True,
    ) -> ReplyKeyboardMarkup:
        btns: list[KeyboardButton] = []

        for action in actions:
            if isinstance(action, str):
                action = {"text": action}

            button_data = {
                DefaultConstructor.aliases.get(k, k): v
                for k, v in action.items()
                if k in DefaultConstructor.available_properties
            }

            if len(button_data) < DefaultConstructor.properties_amount:
                raise ValueError("Недостаточно данных для создания кнопки")

            btns.append(KeyboardButton(**button_data))  # type: ignore

        return ReplyKeyboardMarkup(
            resize_keyboard=resize_keyboard,
            selective=selective,
            one_time_keyboard=one_time_keyboard,
            is_persistent=is_persistent,
            keyboard=schema_generator.create_keyboard_layout(btns, schema),
        )
