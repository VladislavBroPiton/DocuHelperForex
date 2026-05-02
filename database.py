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
            # Включаем векторное расширение
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
            
            # Таблица логов вопросов
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
            
            # Таблица обратной связи
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    query_text TEXT,
                    rating INTEGER,   -- 1 = полезно, 0 = не полезно
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
    async def save_feedback(self, user_id: int, query_text: str, rating: int):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feedback (user_id, query_text, rating, created_at)
                VALUES ($1, $2, $3, NOW())
            """, user_id, query_text, rating)

    # --- Статистика для админа ---
    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            total_queries = await conn.fetchval("SELECT COUNT(*) FROM queries_log")
            today_queries = await conn.fetchval(
                "SELECT COUNT(*) FROM queries_log WHERE created_at::date = CURRENT_DATE"
            )
            week_queries = await conn.fetchval(
                "SELECT COUNT(*) FROM queries_log WHERE created_at > NOW() - INTERVAL '7 days'"
            )
            unique_users = await conn.fetchval(
                "SELECT COUNT(DISTINCT user_id) FROM queries_log"
            )
            # Топ-5 запросов (без учета регистра, простой подсчет)
            top_queries = await conn.fetch("""
                SELECT LOWER(query_text) as q, COUNT(*) as cnt
                FROM queries_log
                GROUP BY LOWER(query_text)
                ORDER BY cnt DESC
                LIMIT 5
            """)
            # Оценки полезности
            total_feedback = await conn.fetchval("SELECT COUNT(*) FROM feedback")
            positive_feedback = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE rating = 1")
            negative_feedback = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE rating = 0")
            positive_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0

            return {
                "total_queries": total_queries,
                "today_queries": today_queries,
                "week_queries": week_queries,
                "unique_users": unique_users,
                "top_queries": [(row["q"], row["cnt"]) for row in top_queries],
                "feedback_total": total_feedback,
                "positive": positive_feedback,
                "negative": negative_feedback,
                "positive_rate": round(positive_rate, 1)
            }

    # --- Работа с чанками (для index_docs.py, если нужно) ---
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
