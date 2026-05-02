import os

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # будет ключ OpenRouter
# Мы будем использовать OpenRouter API, поэтому base_url и модель:
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_MODEL = "deepseek/deepseek-r1:free"  # бесплатная модель

SIMILARITY_THRESHOLD = 0.7  # порог похожести
TOP_K = 3