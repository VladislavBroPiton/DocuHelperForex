import asyncpg
from config import DATABASE_URL

class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(DATABASE_URL)

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def create_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Таблица для фрагментов знаний
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1024),
                    source VARCHAR(255)
                );
            """)
            # Таблица для логов вопросов/ответов
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
            # Таблица для обратной связи
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    query_text TEXT,
                    answer_text TEXT,
                    rating INT,  -- 1 = полезно, 0 = не полезно
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

    # --- Логирование вопросов ---
    async def log_query(self, user_id: int, username: str, query_text: str, answer_text: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO queries_log (user_id, username, query_text, answer_text, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, username, query_text, answer_text)

    # --- Обратная связь ---
    async def save_feedback(self, user_id: int, query_text: str, answer_text: str, rating: int):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feedback (user_id, query_text, answer_text, rating, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, query_text, answer_text, rating)

    # --- Статистика для администратора ---
    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            # Общее количество вопросов
            total_queries = await conn.fetchval("SELECT COUNT(*) FROM queries_log")
            # Вопросов за сегодня
            today_queries = await conn.fetchval(
                "SELECT COUNT(*) FROM queries_log WHERE created_at::date = CURRENT_DATE"
            )
            # Вопросов за неделю
            week_queries = await conn.fetchval(
                "SELECT COUNT(*) FROM queries_log WHERE created_at > NOW() - INTERVAL '7 days'"
            )
            # Уникальные пользователи
            unique_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM queries_log")
            # Топ-5 самых частых запросов (нормализованных)
            top_queries = await conn.fetch("""
                SELECT query_text, COUNT(*) as cnt
                FROM queries_log
                GROUP BY query_text
                ORDER BY cnt DESC
                LIMIT 5
            """)
            # Количество положительных и отрицательных отзывов
            positive = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE rating = 1")
            negative = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE rating = 0")

            return {
                "total_queries": total_queries,
                "today_queries": today_queries,
                "week_queries": week_queries,
                "unique_users": unique_users,
                "top_queries": [(row["query_text"], row["cnt"]) for row in top_queries],
                "positive_feedback": positive,
                "negative_feedback": negative,
            }

    # --- Методы для работы с чанками (используются в index_docs.py) ---
    async def delete_all_chunks(self):
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE documents_chunks;")

    async def insert_chunk(self, chunk_text: str, embedding: list, source: str):
        emb_str = str(embedding)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO documents_chunks (chunk_text, embedding, source)
                VALUES ($1, $2::vector, $3)
            """, chunk_text, emb_str, source)

    async def find_similar_chunks(self, query_embedding: list, threshold: float, top_k: int):
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
