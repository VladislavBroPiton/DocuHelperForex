import asyncio
from sentence_transformers import SentenceTransformer
import asyncpg
from config import DATABASE_URL

# Загружаем бесплатную локальную модель (она скачается один раз, ~120 МБ)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

async def get_embedding(text: str) -> list:
    # Модель возвращает вектор (список чисел)
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
    # Создаём таблицу, если её нет (размерность вектора 384)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents_chunks (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding vector(384),
            source VARCHAR(255)
        );
    """)
    # Очищаем старые данные
    await conn.execute("TRUNCATE documents_chunks;")

    with open("forex_knowledge.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text(full_text, chunk_size=400, overlap=50)
    print(f"Найдено {len(chunks)} фрагментов")

    for i, chunk in enumerate(chunks):
        emb = await get_embedding(chunk)
        await conn.execute(
            "INSERT INTO documents_chunks (chunk_text, embedding, source) VALUES ($1, $2, $3)",
            chunk, emb, "forex_knowledge.txt"
        )
        print(f"Загружен {i+1}/{len(chunks)}")

    await conn.close()
    print("Готово!")

if __name__ == "__main__":
    asyncio.run(main())
