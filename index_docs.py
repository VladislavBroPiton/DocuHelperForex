import asyncio
import aiohttp
import asyncpg
import os
import glob
from config import DATABASE_URL

COHERE_URL = "https://api.cohere.ai/v1/embed"

async def get_embedding(text: str, api_key: str) -> list:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "texts": [text],
        "model": "embed-multilingual-v3.0",
        "input_type": "search_document"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(COHERE_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Cohere API error: {resp.status} - {error_text}")
            data = await resp.json()
            return data["embeddings"][0]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

async def main():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise Exception("COHERE_API_KEY не задан")

    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    await conn.execute("DROP TABLE IF EXISTS documents_chunks;")
    await conn.execute("""
        CREATE TABLE documents_chunks (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding vector(1024),
            source VARCHAR(255)
        );
    """)

    # Находим все .txt файлы в папке knowledge
    txt_files = glob.glob("knowledge/*.txt")
    if not txt_files:
        raise Exception("Нет файлов .txt в папке knowledge/")

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        print(f"Обработка файла: {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = chunk_text(full_text, chunk_size=400, overlap=50)
        print(f"  Найдено {len(chunks)} фрагментов")

        for i, chunk in enumerate(chunks):
            emb = await get_embedding(chunk, api_key)
            emb_str = str(emb)
            await conn.execute(
                "INSERT INTO documents_chunks (chunk_text, embedding, source) VALUES ($1, $2::vector, $3)",
                chunk, emb_str, filename
            )
            print(f"  Загружен {i+1}/{len(chunks)}")

    await conn.close()
    print("Готово! Индексация завершена.")

if __name__ == "__main__":
    asyncio.run(main())
