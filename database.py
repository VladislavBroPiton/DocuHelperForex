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
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1024),
                    source VARCHAR(255)
                );
            """)
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
            # Новая таблица для обратной связи
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    query_text TEXT,
                    answer_text TEXT,
                    rating INTEGER,  -- 1 = полезно, 0 = не полезно
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

    async def log_query(self, user_id: int, username: str, query_text: str, answer_text: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO queries_log (user_id, username, query_text, answer_text, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, username, query_text, answer_text)

    async def save_feedback(self, user_id: int, query_text: str, answer_text: str, rating: int):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feedback (user_id, query_text, answer_text, rating, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, query_text, answer_text, rating)

    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            total_queries = await conn.fetchval("SELECT COUNT(*) FROM queries_log")
            today = await conn.fetchval("SELECT COUNT(*) FROM queries_log WHERE DATE(created_at) = CURRENT_DATE")
            week = await conn.fetchval("SELECT COUNT(*) FROM queries_log WHERE created_at > NOW() - INTERVAL '7 days'")
            unique_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM queries_log")
            # Топ-5 запросов
            top_queries = await conn.fetch("""
                SELECT query_text, COUNT(*) as cnt
                FROM queries_log
                GROUP BY query_text
                ORDER BY cnt DESC
                LIMIT 5
            """)
            top_list = [(row["query_text"], row["cnt"]) for row in top_queries]

            # Обратная связь
            total_feedback = await conn.fetchval("SELECT COUNT(*) FROM feedback")
            positive = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE rating = 1")
            negative = total_feedback - positive if total_feedback else 0

            return {
                "total_queries": total_queries,
                "today_queries": today,
                "week_queries": week,
                "unique_users": unique_users,
                "top_queries": top_list,
                "total_feedback": total_feedback,
                "positive_feedback": positive,
                "negative_feedback": negative,
            }

    # Остальные методы (delete_all_chunks, insert_chunk, find_similar_chunks) если используются, оставьте как есть.
