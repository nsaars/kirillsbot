import aiogram
from bot.keyboards.default.consts import DefaultConstructor

TEXT_CONTACT_REGISTRATION = "Отправить контакт"
TEXT_EMPLOYEE = "Я сотрудник"


def get_contact_registration() -> aiogram.types.ReplyKeyboardMarkup:
    actions = [
        {"text": TEXT_CONTACT_REGISTRATION, "request_contact": True},
        {"text": TEXT_EMPLOYEE}
    ]
    return DefaultConstructor.create_kb(actions, [1, 1])
