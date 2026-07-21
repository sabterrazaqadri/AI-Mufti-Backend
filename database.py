import os
import re
import time
import uuid
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
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
        # Neon's pooler endpoint occasionally drops/resets the connection or
        # has a transient DNS blip; retry a few times before giving up.
        last_exc = None
        for attempt in range(5):
            try:
                _pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=int(os.getenv("DB_POOL_MAX", "10")),
                    dsn=DATABASE_URL,
                    cursor_factory=RealDictCursor,
                )
                break
            except psycopg2.OperationalError as exc:
                last_exc = exc
                time.sleep(20)
        else:
            raise last_exc
    return _pool


@contextmanager
def get_cursor(commit: bool = False):
    """Borrow a pooled connection and yield a cursor, returning it afterwards."""
    pool = _get_pool()
    conn = pool.getconn()
    # Neon (serverless) closes idle connections, so a pooled handle can be dead
    # by the time it's borrowed again — swap dead handles for fresh ones.
    while conn.closed:
        pool.putconn(conn, close=True)
        conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        if not conn.closed:
            conn.rollback()
        raise
    finally:
        pool.putconn(conn, close=bool(conn.closed))


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
        # Citations shown alongside an assistant reply. Added after the table
        # existed, so it is an idempotent ALTER rather than part of the CREATE.
        cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS sources JSONB;")
        # Answers a user chose to publish. Kept separate from `messages` on
        # purpose: publishing is a deliberate act with its own lifetime, and a
        # published page must survive the chat (and account) it came from.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public_answers (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                slug VARCHAR(220) UNIQUE NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources JSONB NOT NULL,
                views INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_public_answers_created "
            "ON public_answers(created_at DESC);"
        )
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
    def create_message(
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Insert a message only if the user owns the chat.

        `sources` are the citations shown with an assistant reply; storing them
        is what lets a reopened chat still show where its rulings came from."""
        with get_cursor(commit=True) as cur:
            if not MessageRepository._user_owns_chat(cur, chat_id, user_id):
                return None
            cur.execute(
                """
                INSERT INTO messages (chat_id, role, content, sources)
                VALUES (%s, %s, %s, %s)
                RETURNING id, chat_id, role, content, sources, created_at;
                """,
                (chat_id, role, content, Json(sources) if sources else None),
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
                SELECT id, chat_id, role, content, sources, created_at
                FROM messages
                WHERE chat_id = %s
                ORDER BY created_at ASC;
                """,
                (chat_id,),
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Published answers
# ---------------------------------------------------------------------------
# A user can publish an exchange to a public, indexable page. Only answers that
# actually carry citations are ever publishable — an unsourced answer must not
# become a permanent page with our name on it.

# Arabic-script punctuation sits inside the U+0600–U+06FF block, so it survives a
# naive "keep Arabic letters" filter and ends up in the URL. Strip it explicitly.
_SLUG_PUNCT = re.compile(r"[،؛؟۔٪-٭۝]")
_SLUG_STRIP = re.compile(r"[^\w؀-ۿ\s-]", re.UNICODE)
_SLUG_SPACE = re.compile(r"[\s_]+")


def slugify_question(text: str, max_len: int = 70) -> str:
    """URL slug from the question itself.

    Urdu and Arabic letters are KEPT rather than stripped: percent-encoded
    Unicode URLs are handled correctly by search engines and are far more
    meaningful to this audience than a transliteration would be.
    """
    s = _SLUG_PUNCT.sub(" ", (text or "").strip().lower())
    s = _SLUG_STRIP.sub("", s)
    s = _SLUG_SPACE.sub("-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0] or s[:max_len]
    return s or "masla"


class PublicAnswerRepository:
    @staticmethod
    def publish(question: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert, resolving slug collisions with a short suffix."""
        base = slugify_question(question)
        with get_cursor(commit=True) as cur:
            for attempt in range(6):
                slug = base if attempt == 0 else f"{base}-{uuid.uuid4().hex[:5]}"
                try:
                    cur.execute(
                        """
                        INSERT INTO public_answers (slug, question, answer, sources)
                        VALUES (%s, %s, %s, %s)
                        RETURNING slug, question, answer, sources, created_at;
                        """,
                        (slug, question, answer, Json(sources)),
                    )
                    return dict(cur.fetchone())
                except psycopg2.errors.UniqueViolation:
                    cur.connection.rollback()
            raise RuntimeError("Could not allocate a unique slug")

    @staticmethod
    def get(slug: str) -> Optional[Dict[str, Any]]:
        with get_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE public_answers SET views = views + 1
                WHERE slug = %s
                RETURNING slug, question, answer, sources, views, created_at;
                """,
                (slug,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    @staticmethod
    def list(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT slug, question, created_at
                FROM public_answers
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s;
                """,
                (limit, offset),
            )
            return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def count() -> int:
        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM public_answers;")
            return int(cur.fetchone()["n"])
