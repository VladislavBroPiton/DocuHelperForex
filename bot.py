import asyncio
import logging
import os
import re
from aiohttp import web
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import asyncpg
from config import (
    BOT_TOKEN, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    OPENROUTER_MODEL, SIMILARITY_THRESHOLD, TOP_K,
    DATABASE_URL, COHERE_API_KEY
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

LLM_MODEL = OPENROUTER_MODEL
logging.info(f"Using LLM model: {LLM_MODEL}")

# Константы для Cohere
COHERE_URL = "https://api.cohere.ai/v1/embed"

async def get_embedding(text: str) -> list:
    """Получает эмбеддинг от Cohere."""
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

async def ask_llm_with_context(query: str, context_chunks: list) -> str:
    """Генерация ответа через OpenRouter с запасным вариантом."""
    if not context_chunks:
        return "Извините, я не нашёл информацию по вашему вопросу."

    context = "\n\n---\n\n".join([chunk[0] for chunk in context_chunks])
    messages = [
        {"role": "system", "content": "Ты — эксперт по трейдингу Forex. Отвечай, используя только контекст. Если ответа нет в контексте, скажи: 'Не знаю, в документации нет'."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
    ]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{OPENROUTER_BASE_URL}/chat/completions"
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logging.error(f"OpenRouter API error: {resp.status} - {error_text}")
                    raise Exception(f"OpenRouter API error: {resp.status}")
                data = await resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                source = context_chunks[0][1]

                # --- ЭКРАНИРОВАНИЕ ДЛЯ MarkdownV2 ---
                escape_chars = r'_*[]()~`>#+-=|{}.!'
                answer_escaped = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', answer)
                source_escaped = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', source)

                return f"{answer_escaped}\n\n📚 *Источник:* {source_escaped}"

    except Exception as e:
        logging.error(f"LLM error: {e}")
        # ЗАПАСНОЙ ВАРИАНТ: возвращаем найденный фрагмент текста
        fallback_text = context_chunks[0][0][:600]
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        fallback_escaped = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', fallback_text)
        source_escaped = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', context_chunks[0][1])
        return f"🔍 *Найдено в базе знаний:*\n\n{fallback_escaped}\n\n📚 *Источник:* {source_escaped}"

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer(
        "Привет! Я AI-помощник по трейдингу Forex. Задавайте вопросы, и я постараюсь найти ответ в моей базе знаний.\n\n"
        "Примеры: «Что такое спред?», «Управление рисками», «пипс»"
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

        similar = [(row["chunk_text"], row["source"], row["similarity"]) for row in rows]
        await conn.close()

        if not similar:
            await message.answer("По вашему вопросу ничего не найдено. Попробуйте переформулировать.")
            return

        answer = await ask_llm_with_context(query, similar)
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
