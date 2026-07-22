"""
Retrieval-Augmented Generation (RAG) layer for AI Mufti.

Stores authentic Hanafi / Ahl-e-Sunnat (Barelvi) source excerpts as embeddings
in Postgres (pgvector) and retrieves the most relevant passages for a question so
the model can ground its answer in — and cite — real sources instead of guessing.

Everything degrades gracefully: if pgvector or the embedding API is unavailable,
retrieve() returns [] and the app behaves exactly as before.
"""
import os
import re
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
    "- Excerpts are retrieved by similarity, so a NON-Islamic question (space, sports, "
    "trivia) can still pull some in. If the question is not an Islamic matter, ignore them "
    "and give the out-of-scope reply instead — never stretch an excerpt to fit.\n"
    "- If the excerpts only PARTLY cover the question, answer the covered part from them and, "
    "for the uncovered part, plainly say you have no mustanad reference for it right now — "
    "do NOT fill the gap from memory and do NOT invent a citation.\n\n"
    "Background excerpts:\n"
)

# Used when retrieval WAS attempted for a substantive question but nothing relevant was
# found. The model must NOT answer a fiqh mas'ala from unverified memory — but "no source"
# is not the right reply to every unanswerable message, so classify FIRST and only fall
# through to the mustanad refusal for genuine Islamic masail.
_NO_SOURCE_DIRECTIVE = (
    "PRIVATE BACKGROUND (the user CANNOT see this and must never learn it exists):\n"
    "No authentic source excerpt was found in the AI Mufti library for this message.\n"
    "Work through these cases IN ORDER and use the FIRST one that fits:\n"
    "\n"
    "CASE A — greeting/salam, thanks, identity question (your name or creator), or a "
    "meta/language follow-up ('in urdu', 'explain', 'aur batao'):\n"
    "    → Respond normally per your other instructions. No refusal.\n"
    "\n"
    "CASE B — the message is NOT an Islamic matter at all (space/NASA, science trivia, "
    "sports, politics, coding, general knowledge, personal chit-chat):\n"
    "    → Do NOT say you lack a reference. Say instead that you only cover Islamic "
    "matters, in the user's own language/script:\n"
    "    • Urdu script: معذرت، میں صرف اسلامی مسائل پر علم رکھتا ہوں۔\n"
    "    • Roman Urdu: Muazrat, main sirf Islami masail par ilm rakhta hun.\n"
    "    • English: Sorry, I only have knowledge about Islamic matters.\n"
    "    → BUT if the topic genuinely has an Islamic angle covered by Qur'an/Hadith "
    "(creation of the universe, the heavens, the moon splitting, etc.), you may address "
    "THAT angle — and only if a background excerpt supports it, which here it does not, "
    "so use the CASE D refusal for that part.\n"
    "\n"
    "CASE C — the message asks for a COUNT or STATISTIC over a text: how many times a word "
    "appears in the Qur'an, number of letters/words/ayat, longest or shortest surah, "
    "word-frequency, or any 'kitni baar / how many times' tally:\n"
    "    → You genuinely cannot compute this: you look up passages, you do not count or "
    "search text mechanically, and you are not permitted to research it yourself. Say so "
    "plainly and honestly in the user's language/script, e.g.:\n"
    "    • Urdu script: معذرت، میں قرآن یا حدیث میں الفاظ کی گنتی نہیں کر سکتا؛ میں مستند "
    "کتابوں کے حوالے سے مسائل کا جواب دیتا ہوں۔\n"
    "    • Roman Urdu: Muazrat, main Quran ya Hadees me alfaz ki ginti nahi kar sakta; main "
    "mustanad kitabon ke hawale se masail ka jawab deta hun.\n"
    "    • English: Sorry, I cannot count word occurrences in the Qur'an or Hadith; I answer "
    "masail from authentic books.\n"
    "    → NEVER guess or state a number, not even approximately. If the user insists that "
    "you search it yourself, repeat that you may only answer from authenticated books.\n"
    "\n"
    "CASE D — a substantive Islamic question or mas'ala:\n"
    "    → You MUST NOT answer from your own memory and MUST NOT give any ruling, verse, "
    "hadith, or citation. Reply with ONLY this refusal, in the user's language/script:\n"
    "    • Urdu script: معذرت، میرے پاس اس وقت اس مسئلے پر کوئی مستند حوالہ نہیں۔\n"
    "    • Roman Urdu: Muazrat, mere paas is waqt is mas'ale par koi mustanad hawala nahi.\n"
    "    • English: Sorry, I do not have an authentic reference on this matter right now.\n"
    "\n"
    "In every case: do NOT mention sources, a library, retrieval, or that anything was or "
    "wasn't found — say nothing beyond the reply itself.\n"
    "The three wordings above are examples, not the whole list. If the user wrote in ANY "
    "other language (Hindi, Bengali, Turkish, Arabic, Persian, Indonesian, French, "
    "Pashto, …), give that same sentence in THAT language and script — never fall back to "
    "English just because the user's language is not listed here.\n\n"
    "---\nUser's message:\n"
)

# Used when retrieval was deliberately SKIPPED (bare meta follow-up like "in urdu").
# Without this the model sees no excerpts and the ACCURACY rule makes it refuse a
# follow-up it should simply act on.
_PASSTHROUGH_DIRECTIVE = (
    "PRIVATE BACKGROUND (the user CANNOT see this and must never learn it exists):\n"
    "This message is a meta/language instruction about your PREVIOUS answer, not a new "
    "mas'ala. No source lookup was performed and NONE IS NEEDED. Do NOT give the "
    "'no mustanad reference' refusal here. Simply re-give, translate, or expand your "
    "previous answer on the SAME topic as instructed, keeping its original citations "
    "exactly as you already stated them.\n\n"
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
    "Transliterate/translate this question into formal Urdu script, using classical "
    "Hanafi fiqhi terminology where the question is genuinely about fiqh (the wording a "
    "book like Bahar-e-Shariat would use).\n"
    "STRICT RULES:\n"
    "- Keep the SAME question. Never invent, substitute, or 'correct' the subject. If a "
    "word is unfamiliar or a proper noun (a name, place, organisation, brand), transliterate "
    "it as-is — do NOT replace it with a similar-sounding Islamic term.\n"
    "- If the question is NOT about Islam/fiqh at all (space, science, sports, technology, "
    "general knowledge), reply with exactly: NONE\n"
    "Reply with ONLY the rewritten Urdu question, or NONE. Nothing else.\n\nQuestion: "
)
# The rewrite is fed to vector search, so a drifted rewrite silently retrieves sources
# for a DIFFERENT question ("Nasa" → "نبیذ" once matched nabeez passages at 0.76 and
# produced a confidently wrong grounding). Discard rewrites that drift this far from
# the original in embedding space.
REWRITE_MIN_SIM = float(os.getenv("RAG_REWRITE_MIN_SIM", "0.75"))

# Arabic script, incl. the Urdu/Persian extensions and presentation forms.
_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _is_arabic_script(text: str) -> bool:
    """True when the query is already written in the corpus's own script."""
    letters = _LETTER_RE.findall(text)
    if not letters:
        return False
    arabic = sum(1 for c in letters if _ARABIC_SCRIPT_RE.match(c))
    return arabic / len(letters) >= 0.5


def _rewrite_query_to_urdu(query: str) -> Optional[str]:
    # Rewrite ANY non-Arabic-script query, not just Latin ones. The embeddings are
    # multilingual, so Hindi/Bengali/Turkish do retrieve unaided — but measured on
    # this corpus they land around 0.67–0.74, straddling the 0.70 threshold, and
    # the Urdu rewrite is what reliably pushes a real question over it. The old
    # [A-Za-z] trigger silently skipped every non-Latin script.
    if _is_arabic_script(query):
        return None
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
        if not text or text.upper().strip(".!؟? ") == "NONE":
            return None
        return text
    except Exception as exc:
        print(f"RAG query rewrite failed (soft): {exc}")
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _search(qvec: List[float], k: int) -> List[Dict[str, Any]]:
    lit = _to_vector_literal(qvec)
    with db.get_cursor() as cur:
        # ivfflat's default probes=1 checks a single cluster and can miss (or
        # return zero) results; probe more lists for good recall at this scale.
        cur.execute("SET LOCAL ivfflat.probes = %s;", (int(os.getenv("RAG_PROBES", "16")),))
        cur.execute(
            """
            SELECT title, reference, content, lang, tags[1] AS slug, tags[2] AS jild_tag,
                   tags[3] AS page_tag,
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

    try:
        qvecs = [(query, embed(query, "retrieval_query"))]
    except Exception as exc:
        print(f"RAG retrieve failed: {exc}")
        return []

    rewritten = _rewrite_query_to_urdu(query)
    if rewritten:
        try:
            rvec = embed(rewritten, "retrieval_query")
            # Guard against a hallucinated rewrite: if it no longer means what the
            # user asked, searching with it retrieves confident, wrong sources.
            sim = _cosine(qvecs[0][1], rvec)
            if sim >= REWRITE_MIN_SIM:
                qvecs.append((rewritten, rvec))
            else:
                print(f"RAG rewrite discarded (drift {sim:.3f}): {rewritten!r}")
        except Exception as exc:
            print(f"RAG rewrite embed failed (soft): {exc}")

    merged: Dict[str, Dict[str, Any]] = {}
    for _q, qv in qvecs:
        try:
            rows = _search(qv, k)
        except Exception as exc:
            # e.g. table/extension not created yet — fail soft.
            print(f"RAG retrieve failed: {exc}")
            continue
        for r in rows:
            key = (r.get("reference") or "") + (r.get("content") or "")[:80]
            if key not in merged or float(r["score"]) > float(merged[key]["score"]):
                merged[key] = r

    rows = sorted(merged.values(), key=lambda r: float(r.get("score") or 0), reverse=True)
    return [r for r in rows[:k] if float(r.get("score") or 0) >= min_score]


# ---------------------------------------------------------------------------
# Open-ended "browse" requests
# ---------------------------------------------------------------------------
# "Hadees sunao" / "koi dua batao" carry no topic, so there is nothing for vector
# search to match on and they never clear MIN_SCORE — the user got a "no mustanad
# reference" refusal even though whole hadith books sit in the table. These are
# served by pulling a random passage from the relevant books instead.

_HADEES_TAGS = ("miraat-ul-manajeeh", "anwaar-ul-hadees", "kalam-e-raza-me-ahadees-ke-jalwe")
_QURAN_TAGS = ("al-quran-ul-kareem",)

# "give me one" phrasing, Roman Urdu / Urdu script / English.
_BROWSE_ASK_RE = re.compile(
    r"(sunao|sunao?n|sunaao|sunaye|sunaiye|suna\s*d[eo]|batao|bataao|bataye|bataiye|"
    r"bata\s*d[eo]|share|tell\s+me|give\s+me|koi|kuch|ek\s|"
    r"سناؤ|سنائیں|سنائیے|سنا\s*د|بتاؤ|بتائیں|بتائیے|بتا\s*د|کوئی|کچھ)",
    re.IGNORECASE,
)
# (topic pattern, book tags to draw from, substring the passage must contain)
_BROWSE_TOPICS = (
    (re.compile(r"(hadees|hadith|hadis|ahadees|حدیث|احادیث)", re.IGNORECASE), _HADEES_TAGS, None),
    (re.compile(r"(ayat|aayat|qur'?an|quran|آیت|قرآن)", re.IGNORECASE), _QURAN_TAGS, None),
    (re.compile(r"(dua|du'?a|دعا)", re.IGNORECASE), None, "دعا"),
    (re.compile(r"(naseehat|nasihat|advice|نصیحت)", re.IGNORECASE), None, "نصیحت"),
)


def browse_intent(query: str):
    """Return (tags, needle) when the message is a topic-less 'sunao/batao' request.

    None means it is a real, topical question — let vector search handle it."""
    q = query.strip()
    words = q.split()
    if len(words) > 8:
        return None  # long enough to carry a real topic
    if len(words) > 3 and not _BROWSE_ASK_RE.search(q):
        return None
    for pat, tags, needle in _BROWSE_TOPICS:
        if pat.search(q):
            return tags, needle
    return None


def browse(tags, needle: Optional[str], k: int = 3) -> List[Dict[str, Any]]:
    """Random passages from the given books — similarity is not meaningful here."""
    if not rag_available():
        return []
    where, params = [], []
    if tags:
        where.append("tags && %s")
        params.append(list(tags))
    if needle:
        where.append("content LIKE %s")
        params.append(f"%{needle}%")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    params.append(k)
    try:
        with db.get_cursor() as cur:
            cur.execute(
                f"SELECT title, reference, content, lang, tags[1] AS slug, "
                f"tags[2] AS jild_tag, tags[3] AS page_tag, 1.0 AS score "
                f"FROM sources{clause} ORDER BY random() LIMIT %s;",
                params,
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        print(f"RAG browse failed: {exc}")
        return []


def build_grounded_input(
    user_input: str,
    passages: List[Dict[str, Any]],
    retrieval_attempted: bool = False,
) -> str:
    """Wrap the user's question with reference excerpts for the model.

    - passages present  → ground the answer strictly in them.
    - no passages, retrieval WAS attempted → instruct the model to refuse (no
      mustanad reference), so fiqh masail are never answered from raw memory.
    - no passages, retrieval NOT attempted (meta follow-up) → tell the model no
      lookup was needed, so language/expand follow-ups still work instead of
      tripping the "no mustanad reference" refusal.
    """
    if not passages:
        if retrieval_attempted:
            return f"{_NO_SOURCE_DIRECTIVE}{user_input}"
        return f"{_PASSTHROUGH_DIRECTIVE}{user_input}"
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


def _digits(tag: Optional[str]) -> Optional[int]:
    """'jild-2' → 2, 'page_0048.txt' → 48. None when the tag is missing/odd."""
    if not tag:
        return None
    d = re.sub(r"\D", "", tag)
    return int(d) if d else None


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
                # Book/volume/page, so a citation can deep-link to the exact
                # original page it was taken from.
                "slug": p.get("slug"),
                "jild": _digits(p.get("jild_tag")),
                "page": _digits(p.get("page_tag")),
                "score": round(float(p.get("score") or 0), 3),
            }
        )
    return out
