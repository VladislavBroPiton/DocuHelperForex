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

def escape_md(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# -------------- НОВЫЕ ФУНКЦИИ ДЛЯ ВЫДЕЛЕНИЯ ПРЕДЛОЖЕНИЙ --------------
def split_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения по . ! ? (учитывает многоточие)"""
    # Заменяем переносы строк на пробелы
    text = text.replace('\n', ' ')
    # Регулярное выражение для разделения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Удаляем пустые и слишком короткие (меньше 10 символов)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def word_set(text: str) -> set:
    """Возвращает множество слов в нижнем регистре, игнорируя знаки препинания."""
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)

def score_sentence(query: str, sentence: str) -> float:
    """Оценка схожести предложения с запросом (пересечение слов / длина запроса)"""
    query_words = word_set(query)
    sentence_words = word_set(sentence)
    if not query_words:
        return 0.0
    overlap = len(query_words & sentence_words)
    return overlap / len(query_words)

def get_best_sentence(query: str, sentences: list[str]) -> tuple[str, float]:
    """Возвращает лучшее предложение и его оценку."""
    best = ""
    best_score = 0.0
    for sent in sentences:
        score = score_sentence(query, sent)
        if score > best_score:
            best_score = score
            best = sent
    return best, best_score

# ----------------------------------------------------------------

kb_buttons = [
    [KeyboardButton(text="📈 Что такое спред?"), KeyboardButton(text="⚖️ Правило 1%")],
    [KeyboardButton(text="📊 Торговые стратегии"), KeyboardButton(text="🛡️ Как управлять рисками?")],
    [KeyboardButton(text="📚 Что такое форекс?"), KeyboardButton(text="🔧 Основные термины")]
]
start_keyboard = ReplyKeyboardMarkup(keyboard=kb_buttons, resize_keyboard=True)

COHERE_URL = "https://api.cohere.ai/v1/embed"

async def get_embedding(text: str) -> list:
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
            return data["embeddings"][0]

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer(
        "🤖 Привет! Я бот-помощник по трейдингу Forex.\n"
        "Задайте вопрос в свободной форме или выберите один из вариантов ниже:",
        reply_markup=start_keyboard
    )

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
            answer = "По вашему вопросу ничего не найдено. Попробуйте переформулировать."
            await message.answer(answer)
            await db.log_query(message.from_user.id, message.from_user.username, query, answer)
            return

        # Берём первый (самый релевантный) чанк
        chunk_text = rows[0]["chunk_text"]
        source = rows[0]["source"]

        # Разбиваем чанк на предложения и выбираем лучшее
        sentences = split_sentences(chunk_text)
        best_sentence, score = get_best_sentence(query, sentences)

        # Если лучшее предложение найдено и оценка > 0 – отвечаем им
        if best_sentence and score > 0:
            final_answer = best_sentence
        else:
            # Если по какой-то причине не вычленили, возвращаем весь чанк (но обрезаем до 500 символов)
            final_answer = chunk_text[:500]

        # Экранируем Markdown
        answer_text = escape_md(final_answer)
        source_escaped = escape_md(source)
        await message.answer(f"{answer_text}\n\n📚 *Источник:* {source_escaped}", parse_mode="MarkdownV2")

        # Логируем
        await db.log_query(message.from_user.id, message.from_user.username, query, final_answer)

    except Exception as e:
        logging.error(f"Ошибка в handle_message: {e}")
        await message.answer("Извините, произошла внутренняя ошибка. Попробуйте позже.")
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
