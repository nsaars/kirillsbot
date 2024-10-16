import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
def get_tools(date_string, time_string):
        return [
        {
            "type": "function",
            "function": {
                "name": "schedule_consultation",
                "description": "Получить желаемые клиенту дату и время для бесплатной 30 минутной онлайн консультации. "
                               "Тебе нужно получить точную дату и время от клиента. Убедись, что клиент назвал "
                               "дату и время, не придумывай ничего от себя."
                               "Консультации проводятся с 13 по 18, каждый день, кроме среды и воскресенья."
                               f"Сегодня {date_string}. Учитывай что время сейчас {time_string}. "
                               f"Обращай внимание на текущий день чем соглашаться на консультацию или "
                               f"предлагать время для консультации.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "format": "%Y-%m-%d", "description": "Желаемая дата для консультации."},
                        "time": {"type": "string", "format": "%H:%M", "description": "Желаемое время для консультации."},
                    },
                    "required": ["date", "time"],
                    "additionalProperties": False,
                },
            }
        }
    ]
