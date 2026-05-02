import asyncio
import logging
import os
import re
from aiohttp import web
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.exceptions import TelegramBadRequest
import asyncpg
from functools import lru_cache
from config import BOT_TOKEN, SIMILARITY_THRESHOLD, TOP_K, DATABASE_URL, COHERE_API_KEY, ADMIN_ID
from database import Database

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
db = Database()

# ------------------- ФУНКЦИИ -------------------
def escape_md(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# Кэш эмбеддингов
embedding_cache = {}

async def get_embedding_cached(text: str) -> str:
    if text in embedding_cache:
        return embedding_cache[text]
    embedding = await get_embedding_raw(text)
    emb_str = str(embedding)
    if len(embedding_cache) > 200:
        first_key = next(iter(embedding_cache))
        del embedding_cache[first_key]
    embedding_cache[text] = emb_str
    return emb_str

async def get_embedding_raw(text: str) -> list:
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
        async with session.post("https://api.cohere.ai/v1/embed", headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Cohere API error: {resp.status} - {error_text}")
            data = await resp.json()
            return data["embeddings"][0]

def split_sentences(text: str) -> list[str]:
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def word_set(text: str) -> set:
    return set(re.findall(r'\b\w+\b', text.lower()))

def score_sentence(query: str, sentence: str) -> float:
    qw = word_set(query)
    sw = word_set(sentence)
    if not qw:
        return 0.0
    overlap = len(qw & sw)
    return overlap / len(qw)

def get_best_sentence(query: str, sentences: list[str]) -> tuple[str, float]:
    best = ""
    best_score = 0.0
    for sent in sentences:
        sc = score_sentence(query, sent)
        if sc > best_score:
            best_score = sc
            best = sent
    return best, best_score

# ------------------- КЛАВИАТУРА -------------------
kb_buttons = [
    [KeyboardButton(text="📈 Что такое спред?"), KeyboardButton(text="⚖️ Правило 1%")],
    [KeyboardButton(text="📊 Торговые стратегии"), KeyboardButton(text="🛡️ Как управлять рисками?")],
    [KeyboardButton(text="📚 Что такое форекс?"), KeyboardButton(text="🔧 Основные термины")]
]
start_keyboard = ReplyKeyboardMarkup(keyboard=kb_buttons, resize_keyboard=True)

# ------------------- ОБРАБОТЧИКИ -------------------
@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    welcome_text = (
        "*🤖 Привет, трейдер\\!*\n\n"
        "Я *DocuHelper Forex* — твой интеллектуальный помощник по трейдингу\\.\n"
        "Я обучен на десятках статей и книг, чтобы отвечать на вопросы о:\n"
        "• фундаментальном и техническом анализе\n"
        "• управлении капиталом и психологии\n"
        "• торговых стратегиях и индикаторах\n\n"
        "Просто напиши свой вопрос в свободной форме или выбери один из вариантов ниже 👇"
    )
    await message.answer(welcome_text, parse_mode="MarkdownV2", reply_markup=start_keyboard)

@dp.message(Command("stats"))
async def stats_cmd(message: types.Message):
    # Проверка, что отправитель — администратор
    if message.from_user.id != ADMIN_ID:
        await message.answer("⛔ У вас нет доступа к этой команде.")
        return
    stats = await db.get_stats()
    top_text = "\n".join([f"{i+1}. `{q}` ({cnt})" for i, (q, cnt) in enumerate(stats['top_queries'])])
    answer = (
        f"📊 *Статистика бота*\n\n"
        f"📋 *Всего вопросов:* {stats['total']}\n"
        f"📅 *За сегодня:* {stats['today']}\n"
        f"📆 *За неделю:* {stats['week']}\n"
        f"👥 *Уникальных пользователей:* {stats['unique_users']}\n"
        f"👍 *Полезных ответов:* {stats['useful']}\n"
        f"👎 *Неполезных:* {stats['not_useful']}\n\n"
        f"🔥 *Топ‑5 запросов:*\n{top_text}"
    )
    await message.answer(answer, parse_mode="MarkdownV2")

@dp.message()
async def handle_message(message: types.Message):
    query = message.text.strip()
    if not query:
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        query_emb_str = await get_embedding_cached(query)

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
            answer = "🤔 *Не нашёл информацию.* Попробуй переформулировать вопрос или спросить что-то другое."
            await message.answer(escape_md(answer), parse_mode="MarkdownV2")
            await db.log_query(message.from_user.id, message.from_user.username, query, answer)
            return

        chunk_text = rows[0]["chunk_text"]
        source = rows[0]["source"]
        sentences = split_sentences(chunk_text)
        best_sentence, score = get_best_sentence(query, sentences)

        if best_sentence and score > 0.3:
            final_answer = best_sentence
        else:
            final_answer = chunk_text[:300] + ("..." if len(chunk_text) > 300 else "")

        header = f"*📘 Ответ на ваш запрос:*\n"
        footer = f"\n\n📚 *Источник:* `{source}`"
        full_message = header + escape_md(final_answer) + footer

        # Inline-кнопки обратной связи
        inline_kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="👍 Полезно", callback_data=f"feedback_useful:{query}"),
             InlineKeyboardButton(text="👎 Не полезно", callback_data=f"feedback_notuseful:{query}")]
        ])

        await message.answer(full_message, parse_mode="MarkdownV2", reply_markup=inline_kb)

        # Логируем вопрос и ответ (без оценки)
        await db.log_query(message.from_user.id, message.from_user.username, query, final_answer)

    except Exception as e:
        logging.error(f"Ошибка в handle_message: {e}")
        await message.answer("⚠️ *Извините, произошла внутренняя ошибка.* Попробуйте позже.", parse_mode="MarkdownV2")
        await db.log_query(message.from_user.id, message.from_user.username, query, f"ERROR: {e}")

# ------------------- ОБРАБОТКА ОБРАТНОЙ СВЯЗИ -------------------
@dp.callback_query(lambda c: c.data and c.data.startswith("feedback_"))
async def handle_feedback(callback: types.CallbackQuery):
    # Извлекаем тип и исходный вопрос
    parts = callback.data.split(":", 1)
    if len(parts) != 2:
        await callback.answer("Ошибка", show_alert=False)
        return
    action, query_text = parts
    rating = 1 if action == "feedback_useful" else -1

    # Сохраняем в БД
    await db.save_feedback(callback.from_user.id, query_text, rating)

    # Убираем кнопки, чтобы пользователь не нажал повторно
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Спасибо за обратную связь!", show_alert=False)

# ------------------- ВЕБХУК -------------------
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
