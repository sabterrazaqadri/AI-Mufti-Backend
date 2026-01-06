import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    """Get database connection"""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not set in environment variables")
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    """Initialize database tables"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Enable UUID extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

            # Create chats table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id VARCHAR(255) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Create messages table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")

        conn.commit()
        print("Database initialized successfully")
    finally:
        conn.close()


def generate_title_from_message(content: str) -> str:
    """Generate a short title from the first user message using AI logic or simple heuristics"""
    if not content:
        return "New Chat"

    # We want a human-readable title, max 4-6 words
    words = content.strip().split()
    if len(words) <= 6:
        return " ".join(words)

    # Simple heuristic: take the first 6 words
    title = " ".join(words[:6])
    if len(words) > 6:
        title += "..."

    return title


class ChatRepository:
    """Repository for chat operations"""

    @staticmethod
    def create_chat(user_id: str, title: str) -> Dict[str, Any]:
        """Create a new chat"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chats (user_id, title)
                    VALUES (%s, %s)
                    RETURNING id, user_id, title, created_at, updated_at;
                """, (user_id, title))
                result = cur.fetchone()
                conn.commit()
                return dict(result)
        finally:
            conn.close()

    @staticmethod
    def get_chats(user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all chats for a user, ordered by updated_at DESC"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, title, created_at, updated_at
                    FROM chats
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s;
                """, (user_id, limit, offset))
                return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def get_chat(chat_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chat by ID, ensuring it belongs to the user"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, title, created_at, updated_at
                    FROM chats
                    WHERE id = %s AND user_id = %s;
                """, (chat_id, user_id))
                result = cur.fetchone()
                return dict(result) if result else None
        finally:
            conn.close()

    @staticmethod
    def update_chat_title(chat_id: str, user_id: str, title: str) -> Optional[Dict[str, Any]]:
        """Update chat title"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE chats
                    SET title = %s, updated_at = NOW()
                    WHERE id = %s AND user_id = %s
                    RETURNING id, user_id, title, created_at, updated_at;
                """, (title, chat_id, user_id))
                result = cur.fetchone()
                conn.commit()
                return dict(result) if result else None
        finally:
            conn.close()

    @staticmethod
    def delete_chat(chat_id: str, user_id: str) -> bool:
        """Delete a chat and all its messages"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM chats
                    WHERE id = %s AND user_id = %s
                    RETURNING id;
                """, (chat_id, user_id))
                result = cur.fetchone()
                conn.commit()
                return result is not None
        finally:
            conn.close()


class MessageRepository:
    """Repository for message operations"""

    @staticmethod
    def create_message(chat_id: str, role: str, content: str) -> Dict[str, Any]:
        """Create a new message"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (chat_id, role, content)
                    VALUES (%s, %s, %s)
                    RETURNING id, chat_id, role, content, created_at;
                """, (chat_id, role, content))
                result = cur.fetchone()

                # Update chat's updated_at
                cur.execute("""
                    UPDATE chats SET updated_at = NOW() WHERE id = %s;
                """, (chat_id,))

                conn.commit()
                return dict(result)
        finally:
            conn.close()

    @staticmethod
    def get_messages(chat_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a chat, verifying user owns the chat"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                # First verify user owns the chat
                cur.execute("""
                    SELECT id FROM chats WHERE id = %s AND user_id = %s;
                """, (chat_id, user_id))
                if not cur.fetchone():
                    return []

                # Get messages
                cur.execute("""
                    SELECT id, chat_id, role, content, created_at
                    FROM messages
                    WHERE chat_id = %s
                    ORDER BY created_at ASC;
                """, (chat_id,))
                return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
