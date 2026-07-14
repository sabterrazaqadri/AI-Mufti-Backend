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
from dotenv import load_dotenv

import database as db

# Configure the key explicitly: without this, standalone scripts (ingest_book.py,
# verify_rag.py) fall back to genai's auto-config, which prefers any ambient
# GOOGLE_API_KEY in the shell over the project's .env GEMINI_API_KEY.
load_dotenv()
_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
if _api_key:
    genai.configure(api_key=_api_key)

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
# cosine similarity (1 - distance); 0..1. Measured on this corpus: genuine fiqh
# masail score ~0.75-0.80 against the books, while off-topic queries (weather,
# sports, crypto) top out ~0.66. 0.70 cleanly separates them so we only answer
# when a real source backs the question and otherwise refuse ("koi mustanad hawala nahi").
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.70"))

_GROUNDING_PREFIX = (
    "PRIVATE BACKGROUND (for your accuracy only — the user CANNOT see this and must "
    "never learn it exists):\n"
    "Below are authentic source excerpts retrieved from the AI Mufti library to answer "
    "this question. You MUST base your ruling ONLY on these excerpts. Strict rules:\n"
    "- Ground the ENTIRE answer in these excerpts. Do NOT add any ruling, condition, verse, "
    "hadith, or detail from your own memory that is not supported by them.\n"
    "- NEVER mention these excerpts, the words 'excerpts'/'sources'/'library', 'provided "
    "references', or that anything was 'retrieved/given'.\n"
    "- NEVER use bracket citation numbers like [1], [2].\n"
    "- When you cite a source, copy its reference line EXACTLY as given (book, jild/hissa, "
    "bab, safha/masla number) — never alter it, never add a page/volume number of your own.\n"
    "- If the excerpts only PARTLY cover the question, answer the covered part from them and, "
    "for the uncovered part, plainly say you have no mustanad reference for it right now — "
    "do NOT fill the gap from memory and do NOT invent a citation.\n\n"
    "Background excerpts:\n"
)

# Used when retrieval WAS attempted for a substantive question but nothing relevant was
# found: the model must refuse rather than answer a fiqh mas'ala from unverified memory.
_NO_SOURCE_DIRECTIVE = (
    "PRIVATE BACKGROUND (the user CANNOT see this and must never learn it exists):\n"
    "No authentic source excerpt was found in the AI Mufti library for this question.\n"
    "RULES:\n"
    "- If this is a substantive Islamic question or mas'ala, you MUST NOT answer it from "
    "your own memory and MUST NOT give any ruling, verse, hadith, or citation. Reply with "
    "ONLY the refusal below, in the SAME language/script the user used:\n"
    "    • Urdu script: معذرت، میرے پاس اس وقت اس مسئلے پر کوئی مستند حوالہ نہیں۔\n"
    "    • Roman Urdu: Muazrat, mere paas is waqt is mas'ale par koi mustanad hawala nahi.\n"
    "    • English: Sorry, I do not have an authentic reference on this matter right now.\n"
    "- EXCEPTION — the refusal does NOT apply if the user's message is only a greeting/salam, "
    "a thank-you, an identity question (your name or creator), or a meta/language follow-up "
    "(e.g. 'in urdu', 'explain'). In those cases respond normally per your other instructions.\n"
    "- Do NOT mention sources, a library, retrieval, or that anything was or wasn't found — "
    "say nothing beyond the refusal sentence itself.\n\n"
    "---\nUser's message:\n"
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
        request_options={"timeout": 12},  # query-time: fail soft fast, don't hang /chat
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


def add_source_with_vector(
    title: str,
    content: str,
    vec: List[float],
    reference: Optional[str] = None,
    lang: str = "ur",
    tags: Optional[List[str]] = None,
) -> str:
    """Insert a source whose embedding was already computed (used by bulk book
    ingestion, which batches many chunks into one embedding API call)."""
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


def embed_batch(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Embed many texts in one API request (Gemini batches lists natively)."""
    res = genai.embed_content(
        model=EMBED_MODEL,
        content=texts,
        task_type=task_type,
        output_dimensionality=EMBED_DIM,
    )
    return res["embedding"]


# Roman-Urdu/English words (eid, musafir, wudu...) sit far from the Urdu-script
# book text in embedding space, so Latin-script queries also search with an
# Urdu-script rewrite from a cheap model (its own free-tier quota bucket).
REWRITE_MODEL = os.getenv("RAG_REWRITE_MODEL", "gemini-2.5-flash-lite")
_REWRITE_PROMPT = (
    "Rewrite this Islamic fiqh question in formal Urdu script using classical "
    "fiqhi terminology (the wording a Hanafi fiqh book like Bahar-e-Shariat would "
    "use). Reply with ONLY the rewritten Urdu question, nothing else.\n\nQuestion: "
)
_LATIN_RE = None  # lazy-compiled


def _rewrite_query_to_urdu(query: str) -> Optional[str]:
    global _LATIN_RE
    import re
    if _LATIN_RE is None:
        _LATIN_RE = re.compile(r"[A-Za-z]{3,}")
    if not _LATIN_RE.search(query):
        return None  # already Urdu/Arabic script
    try:
        model = genai.GenerativeModel(
            REWRITE_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0, max_output_tokens=200
            ),
        )
        # Hard timeout: this runs BEFORE the chat response starts streaming, and
        # on 429 the SDK otherwise retries internally for minutes, hanging /chat.
        resp = model.generate_content(
            _REWRITE_PROMPT + query, request_options={"timeout": 8}
        )
        text = (resp.text or "").strip()
        return text or None
    except Exception as exc:
        print(f"RAG query rewrite failed (soft): {exc}")
        return None


def _search(qvec: List[float], k: int) -> List[Dict[str, Any]]:
    lit = _to_vector_literal(qvec)
    with db.get_cursor() as cur:
        # ivfflat's default probes=1 checks a single cluster and can miss (or
        # return zero) results; probe more lists for good recall at this scale.
        cur.execute("SET LOCAL ivfflat.probes = %s;", (int(os.getenv("RAG_PROBES", "16")),))
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
        return [dict(r) for r in cur.fetchall()]


def retrieve(query: str, k: int = TOP_K, min_score: float = MIN_SCORE) -> List[Dict[str, Any]]:
    """Return the top-k source passages above the similarity threshold.

    Latin-script queries are searched twice — as written and as an Urdu-script
    rewrite — and the results merged by best score."""
    if not rag_available():
        return []

    queries = [query]
    rewritten = _rewrite_query_to_urdu(query)
    if rewritten:
        queries.append(rewritten)

    merged: Dict[str, Dict[str, Any]] = {}
    for q in queries:
        try:
            rows = _search(embed(q, "retrieval_query"), k)
        except Exception as exc:
            # e.g. embed quota or table/extension not created yet — fail soft.
            print(f"RAG retrieve failed: {exc}")
            continue
        for r in rows:
            key = (r.get("reference") or "") + (r.get("content") or "")[:80]
            if key not in merged or float(r["score"]) > float(merged[key]["score"]):
                merged[key] = r

    rows = sorted(merged.values(), key=lambda r: float(r.get("score") or 0), reverse=True)
    return [r for r in rows[:k] if float(r.get("score") or 0) >= min_score]


def build_grounded_input(
    user_input: str,
    passages: List[Dict[str, Any]],
    retrieval_attempted: bool = False,
) -> str:
    """Wrap the user's question with reference excerpts for the model.

    - passages present  → ground the answer strictly in them.
    - no passages, retrieval WAS attempted → instruct the model to refuse (no
      mustanad reference), so fiqh masail are never answered from raw memory.
    - no passages, retrieval NOT attempted (meta/short follow-up) → pass through
      untouched so language/expand follow-ups still work.
    """
    if not passages:
        if retrieval_attempted:
            return f"{_NO_SOURCE_DIRECTIVE}{user_input}"
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
