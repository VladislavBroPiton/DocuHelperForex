import asyncio
from sentence_transformers import SentenceTransformer
import asyncpg
from config import DATABASE_URL

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

async def get_embedding(text: str) -> list:
    return model.encode(text).tolist()

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
    # Включаем расширение vector (если ещё не включено)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Удаляем старую таблицу, чтобы не было конфликта размерности
    await conn.execute("DROP TABLE IF EXISTS documents_chunks;")
    # Создаём новую таблицу с векторами размерности 384
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
