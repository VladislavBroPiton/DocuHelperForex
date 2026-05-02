import asyncpg
from config import DATABASE_URL

class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        """Создаёт пул соединений с базой данных."""
        self.pool = await asyncpg.create_pool(DATABASE_URL)

    async def close(self):
        """Закрывает пул соединений."""
        if self.pool:
            await self.pool.close()

    async def create_tables(self):
        """Создаёт все необходимые таблицы, если они не существуют."""
        async with self.pool.acquire() as conn:
            # Включаем расширение vector (если ещё не включено)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Таблица для хранения фрагментов знаний (используется index_docs.py)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1024),
                    source VARCHAR(255)
                );
            """)
            
            # Таблица для логов вопросов и ответов
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS queries_log (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    username TEXT,
                    query_text TEXT,
                    answer_text TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

    # --- Методы для работы с логами ---
    async def log_query(self, user_id: int, username: str, query_text: str, answer_text: str = None):
        """Сохраняет вопрос пользователя и ответ бота в таблицу queries_log."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO queries_log (user_id, username, query_text, answer_text, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, username, query_text, answer_text)

    # --- Методы для работы с чанками (используются в index_docs.py, но можно вызывать и из бота) ---
    async def delete_all_chunks(self):
        """Удаляет все записи из таблицы documents_chunks (перед переиндексацией)."""
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE documents_chunks;")

    async def insert_chunk(self, chunk_text: str, embedding: list, source: str):
        """Вставляет один фрагмент текста с его векторным представлением."""
        emb_str = str(embedding)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO documents_chunks (chunk_text, embedding, source)
                VALUES ($1, $2::vector, $3)
            """, chunk_text, emb_str, source)

    async def find_similar_chunks(self, query_embedding: list, threshold: float, top_k: int):
        """Возвращает список кортежей (chunk_text, source, similarity) для похожих фрагментов."""
        emb_str = str(query_embedding)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
                FROM documents_chunks
                WHERE 1 - (embedding <=> $1::vector) > $2
                ORDER BY similarity DESC
                LIMIT $3
            """, emb_str, threshold, top_k)
            return [(row["chunk_text"], row["source"], row["similarity"]) for row in rows]
