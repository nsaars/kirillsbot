from datetime import datetime

from data.config import ADMIN, DOCUMENT_ID
from database.crud.consultation import create_consultation
from database.crud.user import get_user_by_id
from utils.ai_assistant.ai_chain import AiChain
from utils.google_docs import add_text_to_document


async def send_consultation_request(bot, user_id, history, date, time):
    date_time = datetime.strptime(f'{date} {time}', '%Y-%m-%d %H:%M')
    summary = (await AiChain.helpers.get_chat_summary(history)).get('summary').content

    create_consultation(user_id, summary, date_time)

    user = get_user_by_id(user_id)

    text = f"Заявка на консультацию от @{user.telegram_username} ({user.telegram_name}).\n\n" \
           f"Дата подачи заявки: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" \
           f"Желаемая дата консультации: {date_time.strftime('%Y-%m-%d %H:%M')}\n\n" \
           f"Основная информация полученная ботом:\n\n{summary}"

    add_text_to_document(DOCUMENT_ID, f"{text}\n\n\n", index=None, service=None)
    await bot.send_message(ADMIN, text)
