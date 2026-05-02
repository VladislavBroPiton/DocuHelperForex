import asyncio
import logging
import os
import re
from aiohttp import web
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import asyncpg
from config import BOT_TOKEN, SIMILARITY_THRESHOLD, TOP_K, DATABASE_URL, COHERE_API_KEY

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

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
                logging.error(f"Cohere API error: {resp.status} - {error_text}")
                raise Exception(f"Cohere API error: {resp.status}")
            data = await resp.json()
            return data["embeddings"][0]

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer(
        "Привет! Я AI-помощник по трейдингу Forex. Я ищу ответы в своей базе знаний и показываю самые подходящие фрагменты.\n"
        "Задайте вопрос, например: «Что такое спред?» или «пипс»."
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
        word_count = len(query.split())
        threshold = 0.35 if word_count <= 2 else SIMILARITY_THRESHOLD

        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch("""
            SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
            FROM documents_chunks
            WHERE 1 - (embedding <=> $1::vector) > $2
            ORDER BY similarity DESC
            LIMIT $3
        """, query_emb_str, threshold, TOP_K)

        if not rows and word_count <= 2:
            rows = await conn.fetch("""
                SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
                FROM documents_chunks
                WHERE 1 - (embedding <=> $1::vector) > 0.2
                ORDER BY similarity DESC
                LIMIT $3
            """, query_emb_str, TOP_K)

        await conn.close()

        if not rows:
            await message.answer("По вашему вопросу ничего не найдено. Попробуйте переформулировать.")
            return

        # Формируем ответ из найденных фрагментов
        answer_parts = []
        for i, row in enumerate(rows, 1):
            text = row["chunk_text"]
            source = row["source"]
            answer_parts.append(f"*Результат {i}:*\n{text}\n📚 *Источник:* {source}")
        answer = "\n\n".join(answer_parts)

        # Экранируем спецсимволы для Telegram MarkdownV2
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        answer = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', answer)
        await message.answer(answer, parse_mode="MarkdownV2")

    except Exception as e:
        logging.error(f"Ошибка в handle_message: {e}")
        await message.answer("Извините, произошла внутренняя ошибка. Попробуйте позже.")

# --- Вебхук ---
async def webhook_handler(request):
    update = await request.json()
    await dp.feed_update(bot, types.Update(**update))
    return web.Response()

async def on_startup():
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
