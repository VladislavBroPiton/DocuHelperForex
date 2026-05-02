import asyncio
import logging
import os
from aiohttp import web
import openai
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import asyncpg
from config import BOT_TOKEN, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, SIMILARITY_THRESHOLD, TOP_K, DATABASE_URL, HF_TOKEN

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

openai.api_key = OPENROUTER_API_KEY
openai.base_url = OPENROUTER_BASE_URL

HF_EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

async def get_embedding(text: str) -> list:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(HF_EMBEDDING_URL, headers=headers, json={"inputs": text}) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"HF API error: {resp.status} - {error_text}")
            embedding = await resp.json()
            return embedding

async def ask_llm_with_context(query: str, context_chunks: list) -> str:
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
        similar = [(row["chunk_text"], row["source"], row["similarity"]) for row in rows]
        await conn.close()
        if not similar:
            await message.answer("По вашему вопросу ничего не найдено.")
            return
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
