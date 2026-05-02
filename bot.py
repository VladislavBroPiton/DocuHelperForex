import asyncio
import logging
import os
from aiohttp import web
import openai
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import asyncpg
from config import BOT_TOKEN, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, SIMILARITY_THRESHOLD, TOP_K, DATABASE_URL, COHERE_API_KEY

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

openai.api_key = OPENROUTER_API_KEY
openai.base_url = OPENROUTER_BASE_URL

COHERE_URL = "https://api.cohere.ai/v1/embed"

async def get_embedding(text: str) -> list:
    """Получает векторное представление текста через Cohere API."""
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "texts": [text],
        "model": "embed-multilingual-v3.0",
        "input_type": "search_query"  # для поисковых запросов
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(COHERE_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Cohere API error: {resp.status} - {error_text}")
            data = await resp.json()
            return data["embeddings"][0]

async def ask_llm_with_context(query: str, context_chunks: list) -> str:
    """Формирует ответ, используя LLM (OpenRouter) и контекст из базы знаний."""
    if not context_chunks:
        return "Извините, я не нашёл информацию по вашему вопросу."
    context = "\n\n---\n\n".join([chunk[0] for chunk in context_chunks])
    messages = [
        {"role": "system", "content": "Ты — эксперт по трейдингу Forex. Отвечай, используя только контекст. Если ответа нет в контексте, скажи: 'Не знаю, в документации нет'."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
    ]
    try:
        response = await openai.ChatCompletion.acreate(
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        source = context_chunks[0][1]
        return f"{answer}\n\n📚 *Источник:* {source}"
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "Ошибка при генерации ответа. Попробуйте позже."

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer(
        "Привет! Я AI-помощник по трейдингу Forex. Задавайте вопросы о трейдинге, и я постараюсь найти ответ в моей базе знаний."
    )

@dp.message()
async def handle_message(message: types.Message):
    query = message.text.strip()
    if not query:
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        # 1. Получаем эмбеддинг вопроса
        query_embedding = await get_embedding(query)
        query_emb_str = str(query_embedding)

        # 2. Динамически выбираем порог схожести
        #    Если запрос очень короткий (1–2 слова), снижаем порог, чтобы находить даже отдалённые совпадения
        word_count = len(query.split())
        if word_count <= 2:
            threshold = 0.35   # низкий порог для коротких запросов (например, "форекс")
        else:
            threshold = SIMILARITY_THRESHOLD   # обычный порог из config.py

        conn = await asyncpg.connect(DATABASE_URL)

        # 3. Поиск похожих чанков
        rows = await conn.fetch("""
            SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
            FROM documents_chunks
            WHERE 1 - (embedding <=> $1::vector) > $2
            ORDER BY similarity DESC
            LIMIT $3
        """, query_emb_str, threshold, TOP_K)

        similar = [(row["chunk_text"], row["source"], row["similarity"]) for row in rows]

        # Дополнительно: если ничего не нашлось, но запрос короткий - повторить с ещё более низким порогом?
        if not similar and word_count <= 2:
            # Пробуем ещё раз с порогом 0.2
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
            await message.answer("По вашему вопросу ничего не найдено.")
            return

        # 4. Генерация ответа через LLM
        answer = await ask_llm_with_context(query, similar)
        await message.answer(answer, parse_mode="Markdown")

    except Exception as e:
        logging.error(f"Ошибка в handle_message: {e}")
        await message.answer("Извините, произошла внутренняя ошибка.")

# --- Вебхук (без изменений) ---
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
