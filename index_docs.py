import asyncio
import aiohttp
import openai
from config import OPENAI_API_KEY, OPENAI_BASE_URL, DATABASE_URL
from database import Database

# Настраиваем OpenAI-клиент для работы через OpenRouter (для генерации ответов не нужен в этом скрипте)
# Но для эмбеддингов мы будем использовать другой сервис.

# Бесплатный эмбеддинг-сервис без ключа
EMBEDDING_API_URL = "https://api.lightweightembeddings.com/v1/embeddings"
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"  # подходящая бесплатная модель

async def get_embedding(text: str) -> list:
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text
        }
        async with session.post(EMBEDDING_API_URL, json=payload) as resp:
            data = await resp.json()
            # data.data[0].embedding
            return data["data"][0]["embedding"]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

async def main():
    db = Database()
    await db.connect()
    await db.create_tables()
    await db.delete_all_chunks()

    with open("forex_knowledge.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text(full_text, chunk_size=400, overlap=50)
    print(f"Найдено {len(chunks)} фрагментов")

    for i, chunk in enumerate(chunks):
        emb = await get_embedding(chunk)
        await db.insert_chunk(chunk, emb, source="forex_knowledge.txt")
        print(f"Загружен {i+1}/{len(chunks)}")

    await db.close()
    print("Готово!")

if __name__ == "__main__":
    asyncio.run(main())