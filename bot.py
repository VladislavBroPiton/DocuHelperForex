import asyncio
import logging
import os
from aiohttp import web
import openai
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from config import BOT_TOKEN, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, SIMILARITY_THRESHOLD, TOP_K
from database import Database
import aiohttp

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
db = Database()

# Настраиваем OpenAI клиент для использования OpenRouter
openai.api_key = OPENAI_API_KEY
openai.base_url = OPENAI_BASE_URL

# --- Эмбеддинги также через бесплатный API (как в index_docs)
EMBEDDING_API_URL = "https://api.lightweightembeddings.com/v1/embeddings"
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"

async def get_embedding(text: str) -> list:
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text
        }
        async with session.post(EMBEDDING_API_URL, json=payload) as resp:
            data = await resp.json()
            return data["data"][0]["embedding"]

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
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        source = context_chunks[0][1]
        return f"{answer}\n\n📚 Источник: {source}"
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "Ошибка при генерации ответа. Попробуйте позже."

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer("Привет! Я AI-помощник по трейдингу Forex. Задавайте вопросы.")

@dp.message()
async def handle_message(message: types.Message):
    query = message.text.strip()
    if not query:
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        emb = await get_embedding(query)
    except Exception as e:
        await message.answer("Ошибка обработки запроса.")
        logging.error(f"Embedding error: {e}")
        return
    similar = await db.find_similar_chunks(emb, SIMILARITY_THRESHOLD, TOP_K)
    if not similar:
        await message.answer("По вашему вопросу ничего не найдено.")
        return
    answer = await ask_llm_with_context(query, similar)
    await message.answer(answer, parse_mode="Markdown")

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