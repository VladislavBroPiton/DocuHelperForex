import asyncio
import logging
import os
import re
from aiohttp import web
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import asyncpg
from config import BOT_TOKEN, ADMIN_ID, SIMILARITY_THRESHOLD, TOP_K, DATABASE_URL, COHERE_API_KEY
from database import Database

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
db = Database()

# ---------- Экранирование ----------
def escape_md(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# ---------- Кэш эмбеддингов ----------
embedding_cache = {}

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

async def get_embedding_cached(text: str) -> str:
    if text in embedding_cache:
        logging.info(f"Cache hit: {text[:30]}...")
        return embedding_cache[text]
    logging.info(f"Cache miss: {text[:30]}...")
    emb = await get_embedding_raw(text)
    emb_str = str(emb)
    if len(embedding_cache) > 200:
        first_key = next(iter(embedding_cache))
        del embedding_cache[first_key]
    embedding_cache[text] = emb_str
    return emb_str

# ---------- Выделение предложений ----------
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
    return len(qw & sw) / len(qw)

def get_best_sentence(query: str, sentences: list[str]) -> tuple[str, float]:
    best = ""
    best_score = 0.0
    for s in sentences:
        sc = score_sentence(query, s)
        if sc > best_score:
            best_score = sc
            best = s
    return best, best_score

# ---------- Клавиатура ----------
kb_buttons = [
    [KeyboardButton(text="📈 Что такое спред?"), KeyboardButton(text="⚖️ Правило 1%")],
    [KeyboardButton(text="📊 Торговые стратегии"), KeyboardButton(text="🛡️ Как управлять рисками?")],
    [KeyboardButton(text="📚 Что такое форекс?"), KeyboardButton(text="🔧 Основные термины")]
]
start_keyboard = ReplyKeyboardMarkup(keyboard=kb_buttons, resize_keyboard=True)

# ---------- Команда /stats ----------
@dp.message(Command("stats"))
async def stats_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        await message.answer("⛔ Доступно только администратору.")
        return
    stats = await db.get_stats_for_admin()
    text = (
        f"📊 *Статистика бота*\n\n"
        f"📌 Всего вопросов: {stats['total_queries']}\n"
        f"📌 За последние 24ч: {stats['today_queries']}\n"
        f"👥 Уникальных пользователей: {stats['unique_users']}\n"
        f"👍 Полезных отзывов: {stats['positive_feedback']}\n"
        f"👎 Неполезных: {stats['negative_feedback']}\n\n"
        f"🔥 *Топ‑5 запросов:*\n"
    )
    for i, (q, cnt) in enumerate(stats['top_queries'], 1):
        text += f"{i}. `{escape_md(q)}` – {cnt}\n"
    await message.answer(text, parse_mode="MarkdownV2")

# ---------- Ответ на сообщения ----------
@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    welcome = (
        "*🤖 Привет, трейдер\\!*\n\n"
        "Я *DocuHelper Forex* — твой интеллектуальный помощник по трейдингу\\.\n"
        "Задай вопрос в свободной форме или выбери вариант ниже 👇"
    )
    await message.answer(welcome, parse_mode="MarkdownV2", reply_markup=start_keyboard)

@dp.message()
async def handle_message(message: types.Message):
    query = message.text.strip()
    if not query:
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        emb_str = await get_embedding_cached(query)
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch(f"""
            SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
            FROM documents_chunks
            WHERE 1 - (embedding <=> $1::vector) > {SIMILARITY_THRESHOLD}
            ORDER BY similarity DESC
            LIMIT {TOP_K}
        """, emb_str)

        if not rows:
            rows = await conn.fetch("""
                SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
                FROM documents_chunks
                WHERE 1 - (embedding <=> $1::vector) > 0.3
                ORDER BY similarity DESC
                LIMIT $2
            """, emb_str, TOP_K)

        await conn.close()

        if not rows:
            answer_msg = "🤔 *Не нашёл информации.* Попробуй переформулировать."
            await message.answer(escape_md(answer_msg), parse_mode="MarkdownV2")
            await db.log_query(message.from_user.id, message.from_user.username, query, answer_msg)
            return

        chunk = rows[0]["chunk_text"]
        source = rows[0]["source"]
        sentences = split_sentences(chunk)
        best, score = get_best_sentence(query, sentences)

        if best and score > 0.3:
            final = best
        else:
            final = chunk[:300] + ("..." if len(chunk) > 300 else "")

        header = "*📘 Ответ на ваш запрос:*\n"
        footer = f"\n\n📚 *Источник:* `{source}`"
        full_answer = header + escape_md(final) + footer

        # Создаём кнопки обратной связи
        callback_data_good = f"feedback_good|{escape_md(query)[:50]}|{escape_md(final)[:80]}"
        callback_data_bad = f"feedback_bad|{escape_md(query)[:50]}|{escape_md(final)[:80]}"
        # Обрежем слишком длинные строки, чтобы не превысить лимит CallbackQuery (64 байта).
        # На самом деле 64 байта — это почти ничего, поэтому лучше хранить только ID вопроса.
        # Мы пойдём простым путём: сохраним в БД feedback связку с последним вопросом пользователя.
        # Но для простоты используем локальную переменную. Проблема больших данных в callback_data обходится.
        # Альтернатива: сохранить временный токен. Но для MVP можно сделать так:
        # В callback_data положим просто "good" или "bad", а в обработчике получим последний вопрос из логов.
        # Реализуем проще: callback_data = f"feedback_{msg_id}" и хранить связь вопрос-ответ в словаре.
        # Однако для красоты я сделаю упрощённую версию: кнопки отправляют просто "👍" и "👎", а в колбэке мы дёргаем последний лог по user_id.
        # Это надёжнее, чем пихать текст в callback_data.
        
        # Создадим кнопки с простыми данными
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="👍 Полезно", callback_data="feedback_positive"),
             InlineKeyboardButton(text="👎 Не полезно", callback_data="feedback_negative")]
        ])
        await message.answer(full_answer, parse_mode="MarkdownV2", reply_markup=keyboard)
        
        # Логируем в queries_log (сохраняем вопрос и ответ)
        await db.log_query(message.from_user.id, message.from_user.username, query, final)

        # Сохраняем последний ответ для обратной связи: в словарь, чтобы в колбэке знать, к какому ответу относится отзыв
        # Используем глобальный словарь, но из-за асинхронности не страшно.
        if not hasattr(bot, "last_answer_cache"):
            bot.last_answer_cache = {}
        bot.last_answer_cache[message.from_user.id] = {"query": query, "answer": final}

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await message.answer("⚠️ *Внутренняя ошибка.* Попробуйте позже.", parse_mode="MarkdownV2")
        await db.log_query(message.from_user.id, message.from_user.username, query, f"ERROR: {e}")

# ---------- Обработчик кнопок обратной связи ----------
@dp.callback_query()
async def feedback_callback(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    data = callback.data
    # Получим последний вопрос-ответ для этого пользователя (из кэша)
    last = getattr(bot, "last_answer_cache", {}).get(user_id)
    if not last:
        await callback.answer("Не удалось определить вопрос. Попробуйте ещё раз.")
        return
    query = last["query"]
    answer = last["answer"]
    if data == "feedback_positive":
        feedback_type = True
        await callback.answer("Спасибо за положительный отзыв! 👍")
    elif data == "feedback_negative":
        feedback_type = False
        await callback.answer("Спасибо за обратную связь, мы постараемся улучшить ответы.")
    else:
        await callback.answer()
        return
    await db.save_feedback(user_id, query, answer, feedback_type)
    # Удаляем кнопки у сообщения, чтобы нельзя было повторно проголосовать
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

# ---------- Вебхук ----------
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
