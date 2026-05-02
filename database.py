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
        """Создаёт все таблицы, если их нет."""
        async with self.pool.acquire() as conn:
            # Расширение vector
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Таблица чанков знаний
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

            # Таблица для обратной связи
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    query_text TEXT,
                    rating INTEGER,  -- 1 = полезно, 0 или -1 = не полезно
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Проверка, что колонка rating существует (если таблица создана ранее без неё)
            await self._ensure_feedback_column(conn)

    async def _ensure_feedback_column(self, conn):
        """Если колонка rating отсутствует, добавляет её."""
        # Проверяем наличие колонки
        result = await conn.fetchrow("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='feedback' AND column_name='rating'
        """)
        if not result:
            await conn.execute("ALTER TABLE feedback ADD COLUMN rating INTEGER;")
            print("Добавлена колонка rating в таблицу feedback")

    # --- Методы для логов ---
    async def log_query(self, user_id: int, username: str, query_text: str, answer_text: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO queries_log (user_id, username, query_text, answer_text, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """, user_id, username, query_text, answer_text)

    # --- Методы для обратной связи ---
    async def save_feedback(self, user_id: int, query_text: str, rating: int):
        """Сохраняет оценку ответа (rating: 1 = полезно, -1 = не полезно)."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feedback (user_id, query_text, rating, created_at)
                VALUES ($1, $2, $3, NOW())
            """, user_id, query_text, rating)

    # --- Методы для статистики ---
    async def get_stats(self) -> dict:
        """Возвращает словарь со статистикой."""
        async with self.pool.acquire() as conn:
            # Общее количество вопросов
            total = await conn.fetchval("SELECT COUNT(*) FROM queries_log")

            # Вопросов за сегодня
            today = await conn.fetchval("SELECT COUNT(*) FROM queries_log WHERE created_at::date = CURRENT_DATE")

            # За неделю
            week = await conn.fetchval("SELECT COUNT(*) FROM queries_log WHERE created_at > NOW() - INTERVAL '7 days'")

            # Уникальные пользователи
            unique_users = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM queries_log")

            # Топ-5 запросов (приводим к нижнему регистру, удаляем лишние пробелы)
            top_queries = await conn.fetch("""
                SELECT LOWER(TRIM(query_text)) as q, COUNT(*) as cnt
                FROM queries_log
                GROUP BY q
                ORDER BY cnt DESC
                LIMIT 5
            """)
            top_list = [(row['q'], row['cnt']) for row in top_queries]

            # Статистика по feedback: сколько полезных/неполезных
            useful = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE rating = 1")
            not_useful = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE rating = -1")

            return {
                "total": total,
                "today": today,
                "week": week,
                "unique_users": unique_users,
                "top_queries": top_list,
                "useful": useful,
                "not_useful": not_useful
            }

    # --- Методы для работы с чанками (для index_docs.py) ---
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
