import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ADMIN_ID = 123456789  # вставьте сюда свой Telegram ID (число)

SIMILARITY_THRESHOLD = 0.7
TOP_K = 3
