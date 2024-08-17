import asyncio
import json
from typing import List, Tuple

from utils.ai_assistant.ai_assistants import AiQuestionAnswering, AiHelpers
from utils.ai_assistant.functions import date_time_filter


class AiChain:
    decision = None
    responses = {
        'contact_response': "Я могу предоставить Вам контакт нашего Co-founder, CFO [Кирилла Добрели]"
                            "(https://t.me/BitMarkt).",
        'next_question_response': "Я бы не прочь поговорить на отвлечённые темы, но я на работе. Если у вас"
                                  " есть вопросы по поводу чат-ботов или их функциональности, буду рад помочь!"
    }
    qa = AiQuestionAnswering()
    helpers = AiHelpers()

    @classmethod
    async def get_responses(cls, text: str, history: List[Tuple[str, str]] = None):
        tasks = [
            asyncio.create_task(cls.helpers.get_message_type(text, history)),
            asyncio.create_task(cls.qa.get_question_response(text, history)),
            asyncio.create_task(cls.qa.get_default_response(text, history)),
            asyncio.create_task(cls.qa.get_bad_words_response(text, history)),
            asyncio.create_task(cls.qa.get_humor_response(text, history)),
        ]

        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            yield result

    @classmethod
    async def get_proper_response(cls, text: str, history: List[Tuple[str, str]]):
        cls.responses = {
            'contact_response': "Я могу предоставить Вам контакт нашего Co-founder, CFO [Кирилла Добрели]"
                                "(https://t.me/BitMarkt).",
            'next_question_response': "Я бы не прочь поговорить на отвлечённые темы, но я на работе. Если у вас"
                                      " есть вопросы по поводу чат-ботов или их функциональности, буду рад помочь!"
        }
        cls.decision = None

        async for response in cls.get_responses(text, history):
            key, message = list(response.items())[0]

            additional_kwargs = message.__getattribute__('additional_kwargs')
            if additional_kwargs:
                tool_calls = additional_kwargs.get('tool_calls')
                if tool_calls and tool_calls[0].get('type') == 'function':
                    function = tool_calls[0].get('function')
                    if function.get('name') == 'schedule_consultation':
                        success, text_response = await date_time_filter(**json.loads(function.get('arguments')))

                        return {'text': text_response, 'success': success, 'schedule_consultation_kwargs': json.loads(function.get('arguments')), 'type': 'schedule_consultation'}

            message_text = message.content
            if key == 'type':
                print(message_text)
                if message_text in ('1', '4', '6'):
                    cls.decision = 'question_response'
                elif message_text == '3':
                    cls.decision = 'humor_response'
                elif message_text == '5':
                    cls.decision = 'bad_words_response'
                elif message_text == '7':
                    cls.decision = 'contact_response'
                    return {'text': cls.responses[cls.decision], 'type': cls.decision}
                elif message_text == '9':
                    cls.decision = 'next_question_response'
                    return {'text': cls.responses[cls.decision], 'type': cls.decision}
                else:
                    cls.decision = 'default_response'
            else:
                cls.responses.update({key: message_text})
                if cls.decision and cls.responses.get(cls.decision):
                    return {'text': cls.responses[cls.decision], 'type': cls.decision}
