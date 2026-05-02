import os
from dotenv import load_dotenv

load_dotenv()

# --- ОСНОВНЫЕ НАСТРОЙКИ ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# --- НАСТРОЙКИ OPENROUTER ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "microsoft/phi-3-medium-128k-instruct:free"

# --- НАСТРОЙКИ ДЛЯ ПОИСКА ---
SIMILARITY_THRESHOLD = 0.7
TOP_K = 3
