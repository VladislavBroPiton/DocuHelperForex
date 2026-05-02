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
                    embedding vector(1536),
                    source VARCHAR(255)
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding 
                ON documents_chunks 
                USING ivfflat (embedding vector_cosine_ops);
            """)

    async def insert_chunk(self, text: str, embedding: list, source: str = "forex_knowledge.txt"):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO documents_chunks (chunk_text, embedding, source) VALUES ($1, $2, $3)",
                text, embedding, source
            )

    async def delete_all_chunks(self):
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE documents_chunks;")

    async def find_similar_chunks(self, query_embedding: list, threshold: float, top_k: int):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chunk_text, source, 1 - (embedding <=> $1::vector) AS similarity
                FROM documents_chunks
                WHERE 1 - (embedding <=> $1::vector) > $2
                ORDER BY similarity DESC
                LIMIT $3
            """, query_embedding, threshold, top_k)
            return [(row["chunk_text"], row["source"], row["similarity"]) for row in rows]