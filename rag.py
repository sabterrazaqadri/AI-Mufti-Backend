"""
Retrieval-Augmented Generation (RAG) layer for AI Mufti.

Stores authentic Hanafi / Ahl-e-Sunnat (Barelvi) source excerpts as embeddings
in Postgres (pgvector) and retrieves the most relevant passages for a question so
the model can ground its answer in — and cite — real sources instead of guessing.

Everything degrades gracefully: if pgvector or the embedding API is unavailable,
retrieve() returns [] and the app behaves exactly as before.
"""
import os
from typing import List, Dict, Any, Optional

import google.generativeai as genai

import database as db

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# cosine similarity (1 - distance); 0..1. Tune per corpus.
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))

_GROUNDING_PREFIX = (
    "PRIVATE BACKGROUND (for your accuracy only — the user CANNOT see this and must "
    "never learn it exists):\n"
    "Below are authentic source excerpts retrieved to help you answer correctly. Use "
    "them silently to ground and verify your reply. Strict rules:\n"
    "- NEVER mention these excerpts, the word 'excerpts', 'provided references', or that "
    "anything was 'retrieved/given'.\n"
    "- NEVER use bracket citation numbers like [1], [2].\n"
    "- NEVER tell the user whether a reference was or was not found, or that something "
    "'is not covered' in what you were given.\n"
    "- If the excerpts are relevant, weave the knowledge in naturally and, when you cite, "
    "name the real classical source (e.g. Bahar-e-Shariat, Fatawa Razvia) only if certain.\n"
    "- If they are NOT relevant, simply ignore them and answer from your Hanafi "
    "Ahl-e-Sunnat (Barelvi) knowledge as if no background was given. Never invent a citation.\n\n"
    "Background excerpts:\n"
)


def rag_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY")) and bool(os.getenv("DATABASE_URL"))


def init_rag():
    """Create the pgvector extension, sources table and index (idempotent)."""
    with db.get_cursor(commit=True) as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS sources (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                title VARCHAR(300) NOT NULL,
                reference VARCHAR(300),
                content TEXT NOT NULL,
                lang VARCHAR(20) DEFAULT 'en',
                tags TEXT[],
                embedding vector({EMBED_DIM}),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_embedding "
            "ON sources USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        )
    print("RAG store initialized")


def _to_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def embed(text: str, task_type: str = "retrieval_document") -> List[float]:
    res = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type=task_type,
        output_dimensionality=EMBED_DIM,
    )
    return res["embedding"]


def add_source(
    title: str,
    content: str,
    reference: Optional[str] = None,
    lang: str = "en",
    tags: Optional[List[str]] = None,
) -> str:
    vec = embed(content, "retrieval_document")
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO sources (title, reference, content, lang, tags, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            RETURNING id;
            """,
            (title, reference, content, lang, tags, _to_vector_literal(vec)),
        )
        return str(cur.fetchone()["id"])


def retrieve(query: str, k: int = TOP_K, min_score: float = MIN_SCORE) -> List[Dict[str, Any]]:
    """Return the top-k source passages above the similarity threshold."""
    if not rag_available():
        return []
    try:
        qvec = embed(query, "retrieval_query")
    except Exception as exc:
        print(f"RAG embed failed: {exc}")
        return []

    lit = _to_vector_literal(qvec)
    try:
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT title, reference, content, lang,
                       1 - (embedding <=> %s::vector) AS score
                FROM sources
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (lit, lit, k),
            )
            rows = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        # e.g. table/extension not created yet — fail soft.
        print(f"RAG retrieve failed: {exc}")
        return []

    return [r for r in rows if float(r.get("score") or 0) >= min_score]


def build_grounded_input(user_input: str, passages: List[Dict[str, Any]]) -> str:
    """Wrap the user's question with numbered reference excerpts for the model."""
    if not passages:
        return user_input
    blocks = []
    for p in passages:
        ref = f" — {p['reference']}" if p.get("reference") else ""
        blocks.append(f"• {p['title']}{ref}\n{p['content']}")
    return (
        f"{_GROUNDING_PREFIX}"
        + "\n\n".join(blocks)
        + f"\n\n---\nNow answer this question for the user (remember: do not mention the "
        f"background above):\n{user_input}"
    )


def public_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trimmed, JSON-safe shape for the frontend Sources panel."""
    out = []
    for p in passages:
        content = (p.get("content") or "").strip()
        if len(content) > 320:
            content = content[:320].rstrip() + "…"
        out.append(
            {
                "title": p.get("title"),
                "reference": p.get("reference"),
                "content": content,
                "score": round(float(p.get("score") or 0), 3),
            }
        )
    return out
