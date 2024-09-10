import json
import os

from environs import Env

env = Env()
env.read_env()

BOT_TOKEN: str = env.str('BOT_TOKEN')
DATABASE_URL = env.str('DATABASE_URL')
BOT_ID: str = BOT_TOKEN.split(":")[0]
ADMIN = env.int('ADMIN')
CREDENTIALS = json.loads(env.str('CREDENTIALS'))
DOCUMENT_ID = env.str('DOCUMENT_ID')
LANGCHAIN_TRACING_V2 = env.str('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = env.str('LANGCHAIN_API_KEY')
OPENAI_API_KEY = env.str('OPENAI_API_KEY')