import asyncio
import aiohttp
import asyncpg
from config import DATABASE_URL

# Бесплатный эндпоинт LightweightEmbeddings (без токена)
EMBEDDING_API_URL = "https://api.lightweightembeddings.com/v1/embeddings"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

async def get_embedding(text: str) -> list:
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text
        }
        async with session.post(EMBEDDING_API_URL, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Embedding API error: {resp.status} - {error_text}")
            data = await resp.json()
            # Ответ LightweightEmbeddings: {"data": [{"embedding": [...]}]}
            embedding = data["data"][0]["embedding"]
            return embedding

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    await conn.execute("DROP TABLE IF EXISTS documents_chunks;")
    await conn.execute("""
        CREATE TABLE documents_chunks (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding vector(384),
            source VARCHAR(255)
        );
    """)

    with open("forex_knowledge.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text(full_text, chunk_size=400, overlap=50)
    print(f"Найдено {len(chunks)} фрагментов")

    for i, chunk in enumerate(chunks):
        emb = await get_embedding(chunk)
        emb_str = str(emb)
        await conn.execute(
            "INSERT INTO documents_chunks (chunk_text, embedding, source) VALUES ($1, $2::vector, $3)",
            chunk, emb_str, "forex_knowledge.txt"
        )
        print(f"Загружен {i+1}/{len(chunks)}")

    await conn.close()
    print("Готово!")

if __name__ == "__main__":
    asyncio.run(main())
