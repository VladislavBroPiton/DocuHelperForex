import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ID администратора (целое число)
ADMIN_ID = 123456789  # Замените на свой ID

SIMILARITY_THRESHOLD = 0.7
TOP_K = 1  # для чистоты ответа оставляем один результат
