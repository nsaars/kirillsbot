import aiogram.types
from .consts import DefaultConstructor


class BasicButtons(DefaultConstructor):

    @staticmethod
    def back() -> aiogram.types.ReplyKeyboardMarkup:
        return BasicButtons.create_kb(["◀️Назад"], [1])

    @staticmethod
    def cancel() -> aiogram.types.ReplyKeyboardMarkup:
        return BasicButtons.create_kb(["🚫 Отмена"], [1])

    @staticmethod
    def back_n_cancel() -> aiogram.types.ReplyKeyboardMarkup:
        return BasicButtons.create_kb(["◀️Назад", "🚫 Отмена"], [1, 1])

    @staticmethod
    def confirmation(add_back: bool = False, add_cancel: bool = False) -> aiogram.types.ReplyKeyboardMarkup:
        btns = []
        schema = []

        if add_cancel:
            btns.append("🚫 Отмена")
            schema.append(1)

        btns.append("✅Подтвердить")
        schema.append(1)

        if add_back:
            btns.append("◀️Назад")
            schema.append(1)

        return BasicButtons.create_kb(btns, schema)

    @staticmethod
    def skip(add_back: bool = False, add_cancel: bool = False) -> aiogram.types.ReplyKeyboardMarkup:
        btns = ["▶️Пропустить"]
        schema = [1]

        if add_back:
            btns.append("◀️Назад")
            schema.append(1)

        if add_cancel:
            btns.append("🚫 Отмена")
            schema.append(1)

        return BasicButtons.create_kb(btns, schema)

    @staticmethod
    def yes(add_back: bool = False, add_cancel: bool = False) -> aiogram.types.ReplyKeyboardMarkup:
        btns = ["✅Да"]
        schema = [1]

        if add_back:
            btns.append("◀️Назад")
            schema.append(1)

        if add_cancel:
            btns.append("🚫 Отмена")
            schema.append(1)

        return BasicButtons.create_kb(btns, schema)

    @staticmethod
    def no(add_back: bool = False, add_cancel: bool = False) -> aiogram.types.ReplyKeyboardMarkup:
        btns = ["❌Нет"]
        schema = [1]

        if add_back:
            btns.append("◀️Назад")
            schema.append(1)

        if add_cancel:
            btns.append("🚫 Отмена")
            schema.append(1)

        return BasicButtons.create_kb(btns, schema)

    @staticmethod
    def yes_n_no(add_back: bool = False, add_cancel: bool = False) -> aiogram.types.ReplyKeyboardMarkup:
        btns = ["✅Да", "❌Нет"]
        schema = [2]

        if add_back:
            btns.append("◀️Назад")
            schema.append(1)

        if add_cancel:
            btns.append("🚫 Отмена")
            schema.append(1)

        return BasicButtons.create_kb(btns, schema)
