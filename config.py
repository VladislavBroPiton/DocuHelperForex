import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ID администратора (ваш Telegram ID)
ADMIN_ID = 123456789  # ЗАМЕНИТЕ НА ВАШ ID (можно узнать у бота @userinfobot)

SIMILARITY_THRESHOLD = 0.7
TOP_K = 1  # Оставляем 1, потому что мы вычленяем лучшее предложение
