import os
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# .strip() guards against a trailing newline/space in the host's secret value,
# which otherwise makes psycopg2 read e.g. sslmode="require\n" and fail.
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip() or None

# ---- Connection pool (single, app-lifetime) ----
_pool: Optional[ThreadedConnectionPool] = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not set in environment variables")
        _pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("DB_POOL_MAX", "10")),
            dsn=DATABASE_URL,
            cursor_factory=RealDictCursor,
        )
    return _pool


@contextmanager
def get_cursor(commit: bool = False):
    """Borrow a pooled connection and yield a cursor, returning it afterwards."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db():
    """Initialize database tables and indexes (idempotent)."""
    with get_cursor(commit=True) as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id VARCHAR(255) NOT NULL,
                title VARCHAR(500) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")
    print("Database initialized successfully")


def generate_title_from_message(content: str) -> str:
    """Fast, dependency-free fallback title from the first user message."""
    if not content:
        return "New Chat"
    words = content.strip().split()
    if len(words) <= 6:
        return " ".join(words)
    return " ".join(words[:6]) + "..."


class ChatRepository:
    @staticmethod
    def create_chat(user_id: str, title: str) -> Dict[str, Any]:
        with get_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO chats (user_id, title)
                VALUES (%s, %s)
                RETURNING id, user_id, title, created_at, updated_at;
                """,
                (user_id, title),
            )
            return dict(cur.fetchone())

    @staticmethod
    def get_chats(user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM chats
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s;
                """,
                (user_id, limit, offset),
            )
            return [dict(row) for row in cur.fetchall()]

    @staticmethod
    def get_chat(chat_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM chats
                WHERE id = %s AND user_id = %s;
                """,
                (chat_id, user_id),
            )
            result = cur.fetchone()
            return dict(result) if result else None

    @staticmethod
    def update_chat_title(chat_id: str, user_id: str, title: str) -> Optional[Dict[str, Any]]:
        with get_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE chats
                SET title = %s, updated_at = NOW()
                WHERE id = %s AND user_id = %s
                RETURNING id, user_id, title, created_at, updated_at;
                """,
                (title, chat_id, user_id),
            )
            result = cur.fetchone()
            return dict(result) if result else None

    @staticmethod
    def delete_chat(chat_id: str, user_id: str) -> bool:
        with get_cursor(commit=True) as cur:
            cur.execute(
                """
                DELETE FROM chats
                WHERE id = %s AND user_id = %s
                RETURNING id;
                """,
                (chat_id, user_id),
            )
            return cur.fetchone() is not None


class MessageRepository:
    @staticmethod
    def _user_owns_chat(cur, chat_id: str, user_id: str) -> bool:
        cur.execute("SELECT 1 FROM chats WHERE id = %s AND user_id = %s;", (chat_id, user_id))
        return cur.fetchone() is not None

    @staticmethod
    def create_message(chat_id: str, user_id: str, role: str, content: str) -> Optional[Dict[str, Any]]:
        """Insert a message only if the user owns the chat."""
        with get_cursor(commit=True) as cur:
            if not MessageRepository._user_owns_chat(cur, chat_id, user_id):
                return None
            cur.execute(
                """
                INSERT INTO messages (chat_id, role, content)
                VALUES (%s, %s, %s)
                RETURNING id, chat_id, role, content, created_at;
                """,
                (chat_id, role, content),
            )
            result = cur.fetchone()
            cur.execute("UPDATE chats SET updated_at = NOW() WHERE id = %s;", (chat_id,))
            return dict(result)

    @staticmethod
    def get_messages(chat_id: str, user_id: str) -> List[Dict[str, Any]]:
        with get_cursor() as cur:
            if not MessageRepository._user_owns_chat(cur, chat_id, user_id):
                return []
            cur.execute(
                """
                SELECT id, chat_id, role, content, created_at
                FROM messages
                WHERE chat_id = %s
                ORDER BY created_at ASC;
                """,
                (chat_id,),
            )
            return [dict(row) for row in cur.fetchall()]
