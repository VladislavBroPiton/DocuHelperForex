import asyncio
import logging
import os
import re
from aiohttp import web
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
import asyncpg
from config import BOT_TOKEN, SIMILARITY_THRESHOLD, TOP_K, DATABASE_URL, COHERE_API_KEY
from database import Database

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
db = Database()

# ---------- КЭШ ЭМБЕДДИНГОВ ----------
embedding_cache = {}
CACHE_MAX_SIZE = 100  # ограничим количество записей, чтобы не переполнить память

def get_cached_embedding(text: str):
    return embedding_cache.get(text)

def set_cached_embedding(text: str, embedding: list):
    if len(embedding_cache) >= CACHE_MAX_SIZE:
        # удаляем первый (старейший) элемент
        first_key = next(iter(embedding_cache))
        del embedding_cache[first_key]
    embedding_cache[text] = embedding

def escape_md(text: str) -> str:
    """Экранирует специальные символы Telegram MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# -------------- ФУНКЦИИ ДЛЯ ВЫДЕЛЕНИЯ ПРЕДЛОЖЕНИЙ --------------
def split_sentences(text: str) -> list[str]:
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def word_set(text: str) -> set:
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)

def score_sentence(query: str, sentence: str) -> float:
    query_words = word_set(query)
    sentence_words = word_set(sentence)
    if not query_words:
        return 0.0
    overlap = len(query_words & sentence_words)
    return overlap / len(query_words)

def get_best_sentence(query: str, sentences: list[str]) -> tuple[str, float]:
    best = ""
    best_score = 0.0
    for sent in sentences:
        score = score_sentence(query, sent)
        if score > best_score:
            best_score = score
            best = sent
    return best, best_score

# -------------- ФОРМАТИРОВАНИЕ ОТВЕТА --------------
def format_answer(sentence: str, source: str) -> str:
    """Оформляет ответ в красивом формате MarkdownV2."""
    # Убираем лишние пробелы и переносы
    sentence = sentence.strip()
    # Если предложение содержит двоеточие, возможно это термин
    # Но пока просто выделим жирным первое слово (если это не слишком длинное)
    # Для простоты – добавим красивый заголовок
    parts = sentence.split(':', 1)
    if len(parts) > 1 and len(parts[0]) < 40:
        # Есть двоеточие, первая часть – термин
        term = parts[0].strip()
        definition = parts[1].strip()
        formatted = f"*{escape_md(term)}*: {escape_md(definition)}"
    else:
        formatted = escape_md(sentence)
    source_escaped = escape_md(source)
    return f"{formatted}\n\n📚 *Источник:* {source_escaped}"

# ---------- КЛАВИАТУРА ----------
kb_buttons = [
    [KeyboardButton(text="📈 Что такое спред?"), KeyboardButton(text="⚖️ Правило 1%")],
    [KeyboardButton(text="📊 Торговые стратегии"), KeyboardButton(text="🛡️ Как управлять рисками?")],
    [KeyboardButton(text="📚 Что такое форекс?"), KeyboardButton(text="🔧 Основные термины")]
]
start_keyboard = ReplyKeyboardMarkup(keyboard=kb_buttons, resize_keyboard=True)

COHERE_URL = "https://api.cohere.ai/v1/embed"

async def get_embedding(text: str) -> list:
    # Проверяем кэш
    cached = get_cached_embedding(text)
    if cached is not None:
        logging.debug(f"Cache hit for: {text[:30]}...")
        return cached
    logging.debug(f"Cache miss for: {text[:30]}...")
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "texts": [text],
        "model": "embed-multilingual-v3.0",
        "input_type": "search_query"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(COHERE_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Cohere API error: {resp.status} - {error_text}")
            data = await resp.json()
            embedding = data["embeddings"][0]
            set_cached_embedding(text, embedding)
            return embedding

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    # Красивое приветствие с эмодзи, используем MarkdownV2, но экранируем отдельные символы
    greeting = (
        "🤖 *Добро пожаловать в Forex Assistant\\!*\n\n"
        "Я — бот, который помогает разбираться в трейдинге на валютном рынке\\. "
        "Мои знания основаны на проверенных источниках, и я ищу точные ответы на ваши вопросы\\.\n\n"
        "✨ *Что я умею:*\n"
        "• Отвечать на вопросы по трейдингу простым языком\n"
        "• Находить определения терминов, стратегии, психологию торговли\n"
        "• Давать ссылку на источник знаний\n\n"
        "💬 *Как спросить?*\n"
        "Просто напишите вопрос или выберите один из вариантов ниже\\.\n\n"
        "📌 *Примеры:*\n"
        "`Что такое пипс?`\n"
        "`Как управлять рисками?`\n"
        "`Какие бывают торговые сессии?`"
    )
    # В MarkdownV2 нужно экранировать '.', '!', '?', но мы уже экранировали вручную.
    # Используем parse_mode="MarkdownV2"
    await message.answer(greeting, parse_mode="MarkdownV2", reply_markup=start_keyboard)

@dp.message()
async def handle_message(message: types.Message):
    query = message.text.strip()
    if not query:
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        query_embedding = await get_embedding(query)
        query_emb_str = str(query_embedding)

        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch(f"""
            SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
            FROM documents_chunks
            WHERE 1 - (embedding <=> $1::vector) > {SIMILARITY_THRESHOLD}
            ORDER BY similarity DESC
            LIMIT {TOP_K}
        """, query_emb_str)

        if not rows:
            rows = await conn.fetch("""
                SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
                FROM documents_chunks
                WHERE 1 - (embedding <=> $1::vector) > 0.3
                ORDER BY similarity DESC
                LIMIT $2
            """, query_emb_str, TOP_K)

        await conn.close()

        if not rows:
            answer = "❌ *Не найдено.* Попробуйте переформулировать вопрос или спросить о другом термине."
            await message.answer(answer, parse_mode="MarkdownV2")
            await db.log_query(message.from_user.id, message.from_user.username, query, answer)
            return

        # Берём первый (самый релевантный) чанк
        chunk_text = rows[0]["chunk_text"]
        source = rows[0]["source"]

        # Разбиваем на предложения и выбираем лучшее
        sentences = split_sentences(chunk_text)
        best_sentence, score = get_best_sentence(query, sentences)

        if best_sentence and score > 0:
            final_answer = best_sentence
        else:
            # fallback – весь чанк, но коротко
            final_answer = chunk_text[:500]

        # Форматируем ответ
        formatted = format_answer(final_answer, source)
        await message.answer(formatted, parse_mode="MarkdownV2")

        # Логируем
        await db.log_query(message.from_user.id, message.from_user.username, query, final_answer)

    except Exception as e:
        logging.error(f"Ошибка в handle_message: {e}")
        await message.answer("⚠️ *Произошла внутренняя ошибка.* Попробуйте позже.", parse_mode="MarkdownV2")
        await db.log_query(message.from_user.id, message.from_user.username, query, f"ERROR: {e}")

# --- Вебхук ---
async def webhook_handler(request):
    update = await request.json()
    await dp.feed_update(bot, types.Update(**update))
    return web.Response()

async def on_startup():
    await db.connect()
    await db.create_tables()
    await bot.delete_webhook(drop_pending_updates=True)
    webhook_url = f"{os.getenv('RENDER_EXTERNAL_URL')}/webhook"
    await bot.set_webhook(webhook_url)
    logging.info(f"Webhook set to {webhook_url}")

async def main():
    app = web.Application()
    app.router.add_post("/webhook", webhook_handler)
    app.router.add_get("/", lambda req: web.Response(text="OK"))
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    await on_startup()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
