from environs import Env

env = Env()
env.read_env()

BOT_TOKEN: str = env.str("BOT_TOKEN")
DATABASE_URL = env.str("DATABASE_URL")
BOT_ID: str = BOT_TOKEN.split(":")[0]

ADMIN = env.str("ADMIN")
