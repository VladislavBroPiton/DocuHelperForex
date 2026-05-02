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

# --- КЛАВИАТУРА ---
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
        "🤖 *Привет! Я бот-помощник по трейдингу Forex.*\n"
        "Задайте вопрос в свободной форме или выберите один из вариантов ниже:",
        parse_mode="MarkdownV2",
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
        rows = await conn.fetch("""
            SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
            FROM documents_chunks
            WHERE 1 - (embedding <=> $1::vector) > $2
            ORDER BY similarity DESC
            LIMIT $3
        """, query_emb_str, SIMILARITY_THRESHOLD, TOP_K)

        if not rows:
            rows = await conn.fetch("""
                SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
                FROM documents_chunks
                WHERE 1 - (embedding <=> $1::vector) > 0.3
                ORDER BY similarity DESC
                LIMIT $3
            """, query_emb_str, TOP_K)

        await conn.close()

        if not rows:
            answer = "По вашему вопросу ничего не найдено. Попробуйте переформулировать."
            await message.answer(answer)
            await db.log_query(message.from_user.id, message.from_user.username, query, answer)
            return

        # Формируем ответ
        answer_parts = []
        for i, row in enumerate(rows, 1):
            text = row["chunk_text"]
            source = row["source"]
            answer_parts.append(f"*Результат {i}:*\n{text}\n📚 *Источник:* {source}")
        answer = "\n\n".join(answer_parts)

        # Экранирование спецсимволов для MarkdownV2
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        answer = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', answer)
        await message.answer(answer, parse_mode="MarkdownV2")

        # Логируем вопрос и ответ
        await db.log_query(message.from_user.id, message.from_user.username, query, answer)

    except Exception as e:
        logging.error(f"Ошибка в handle_message: {e}")
        await message.answer("Извините, произошла внутренняя ошибка. Попробуйте позже.")
        await db.log_query(message.from_user.id, message.from_user.username, query, "ERROR: " + str(e))

# --- Вебхук ---
async def webhook_handler(request):
    update = await request.json()
    await dp.feed_update(bot, types.Update(**update))
    return web.Response()

async def on_startup():
    await db.connect()
    await db.create_tables()  # убедитесь, что create_tables создаёт и queries_log
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
