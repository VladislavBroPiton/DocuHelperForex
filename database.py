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
            # Таблица для обратной связи
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    query_text TEXT,
                    answer_text TEXT,
                    feedback_type BOOLEAN,  -- TRUE = полезно, FALSE = не полезно
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

    async def log_query(self, user_id: int, username: str, query_text: str, answer_text: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO queries_log (user_id, username, query_text, answer_text, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, username, query_text, answer_text)

    async def save_feedback(self, user_id: int, query_text: str, answer_text: str, feedback_type: bool):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feedback (user_id, query_text, answer_text, feedback_type, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, query_text, answer_text, feedback_type)

    async def get_stats_for_admin(self) -> dict:
        async with self.pool.acquire() as conn:
            # Общее количество вопросов
            total_queries = await conn.fetchval("SELECT COUNT(*) FROM queries_log;")
            # Вопросы за последние 24 часа
            today_queries = await conn.fetchval("""
                SELECT COUNT(*) FROM queries_log
                WHERE created_at > NOW() - INTERVAL '1 day';
            """)
            # Уникальные пользователи за всё время
            unique_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM queries_log;")
            # Топ-5 запросов (без учёта регистра, исключая слишком короткие)
            top_queries = await conn.fetch("""
                SELECT lower(query_text) as q, COUNT(*) as cnt
                FROM queries_log
                WHERE LENGTH(query_text) > 3
                GROUP BY q
                ORDER BY cnt DESC
                LIMIT 5;
            """)
            # Количество полезных и неполезных отзывов
            positive = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE feedback_type = TRUE;")
            negative = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE feedback_type = FALSE;")
            
            return {
                "total_queries": total_queries,
                "today_queries": today_queries,
                "unique_users": unique_users,
                "top_queries": [(row["q"], row["cnt"]) for row in top_queries],
                "positive_feedback": positive,
                "negative_feedback": negative,
            }
