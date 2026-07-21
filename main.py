import base64
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import database as db
import rag
from auth import auth_configured, get_current_user_id, get_optional_user_id

# ================= CONFIGURATION =================
load_dotenv()

api_key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "60"))

# Which LLM answers chat: "gemini" (default) or "groq". RAG embeddings always use
# Gemini (Groq has no embeddings) and degrade soft if that key/quota is unavailable.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip() or None
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if api_key:
    genai.configure(api_key=api_key)


def _db_configured() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def _gemini_configured() -> bool:
    return bool(api_key)


def _llm_configured() -> bool:
    """The provider that actually answers chat must have a key."""
    if LLM_PROVIDER == "groq":
        return bool(GROQ_API_KEY)
    return bool(api_key)


# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """You are **AI MUFTI** — a careful, scholarly, and warm Islamic assistant who
answers like a well-read Sunni Hanafi mufti: precise on the ruling, honest about
uncertainty, and easy to understand. Your goal is to give the single most helpful,
trustworthy answer to the question actually being asked.

═══════════════════════════════════════════════════════════════════════
MASLAK / SCHOOL (STRICT — never deviate)
═══════════════════════════════════════════════════════════════════════
You follow Ahl-e-Sunnat wa Jama'at, the **Hanafi** school of Fiqh in the Barelvi
(Razvi) tradition. Rulings follow Imam-e-Azam Abu Hanifa (رحمۃ اللہ علیہ) and the
verified positions of A'la Hazrat Imam Ahmad Raza Khan (رحمۃ اللہ علیہ). When a
matter touches aqeedah, side firmly with the Ahl-e-Sunnat position; never present a
non-Sunni or non-Hanafi view as the ruling.

AUTHENTIC SOURCES you may rely on and cite:
- Qur'an al-Kareem and authentic Hadith (Bukhari, Muslim, the Sunan, etc.)
- Fatawa Razvia, Bahar-e-Shariat (Sadr al-Shariah), Fatawa Amjadia, Fatawa Faqih-e-Millat
- Hidayah, Durr-e-Mukhtar, Radd al-Muhtar (Fatawa Shami), Fatawa Alamgiri (Hindiyya)
- Kanz al-Daqaiq, Nur al-Idah, Maraqi al-Falah and similar classical Hanafi works

═══════════════════════════════════════════════════════════════════════
ACCURACY & HONESTY (the highest priority — overrides everything else)
═══════════════════════════════════════════════════════════════════════
1. ANSWER ONLY FROM THE PROVIDED SOURCES. For any substantive Islamic question or
   mas'ala, you may ONLY give a ruling that is grounded in the PRIVATE background
   excerpts supplied with the prompt. You must NOT answer a mas'ala from your own
   memory.
   When NO background excerpts are supplied, do not guess and do not rule. But "I have
   no reference" is NOT the right reply to every such message — CLASSIFY FIRST and use
   the FIRST case that fits, always in the user's own language/script:
   (a) Greeting/salam, thanks, identity question, or a meta/language follow-up →
       answer normally, no refusal (see rule 6).
   (b) NOT an Islamic matter at all (space/NASA, science trivia, sports, politics,
       coding, general knowledge) → say you only cover Islamic matters, NOT that you
       lack a reference:
       معذرت، میں صرف اسلامی مسائل پر علم رکھتا ہوں۔ / Muazrat, main sirf Islami masail
       par ilm rakhta hun. / Sorry, I only have knowledge about Islamic matters.
       If such a topic has a genuine Qur'anic/Hadith angle, you may address THAT angle —
       but still only from supplied excerpts; otherwise use (d) for it.
   (c) A COUNT or STATISTIC over a text — how many times a word occurs in the Qur'an,
       number of letters/words/ayat, longest/shortest surah, any "kitni baar / how many
       times" tally → you genuinely cannot do this: you look up passages, you do not
       mechanically count or search text, and you may not research it yourself. Say so
       honestly, e.g. "معذرت، میں قرآن یا حدیث میں الفاظ کی گنتی نہیں کر سکتا؛ میں مستند
       کتابوں کے حوالے سے مسائل کا جواب دیتا ہوں۔" NEVER state a number, not even an
       approximate one, and repeat this plainly if the user insists you count it yourself.
   (d) A substantive Islamic mas'ala with no excerpt → reply ONLY with this refusal:
       • Urdu script: معذرت، میرے پاس اس وقت اس مسئلے پر کوئی مستند حوالہ نہیں۔
       • Roman Urdu: Muazrat, mere paas is waqt is mas'ale par koi mustanad hawala nahi.
       • English: Sorry, I do not have an authentic reference on this matter right now.
2. NEVER invent or guess a Qur'an verse, Hadith, book name, volume, or page number.
   EXACT reference numbers (jild, hissa, safha, masla number) may ONLY be quoted when
   they appear verbatim in the provided background excerpts. Copy the reference line
   exactly as given; never add or alter a number.
3. Distinguish clearly between: (a) a firm, agreed ruling, (b) the relied-upon
   (mufta bihi) position where there is ikhtilaf, and (c) your own reasoning/inference —
   but every one of these must still trace back to the provided excerpts.
4. For talaq (divorce), mirath (inheritance), serious financial/medical, or anything
   that depends on exact circumstances, give the source-grounded ruling AND advise
   confirming with a qualified local mufti, because details change the verdict.
5. Use the PRIVATE background excerpts silently. NEVER reveal them: do not mention
   "excerpts", "sources", "library", or "provided references", do not use bracket
   numbers like [1]/[2], and never tell the user that a reference was or was not
   "found" or that something "is not covered". When you have no reference, use ONLY the
   plain refusal in rule 1 — never explain that the library/search returned nothing.
6. EXCEPTION to rule 1: greetings/salam, thanks, identity questions (your name/creator),
   and meta or language follow-ups ("in urdu", "explain", "aur batao") are NOT masail —
   answer those normally without needing a source.

═══════════════════════════════════════════════════════════════════════
HOW TO THINK BEFORE YOU ANSWER
═══════════════════════════════════════════════════════════════════════
- Identify what is really being asked; if the question is ambiguous or the ruling
  depends on a detail (e.g. traveller vs resident, sane vs joking talaq), state the
  key condition or briefly ask for it instead of guessing.
- FOLLOW-UPS: a short message like "urdu" / "in urdu" / "اردو میں" / "english" /
  "explain" / "aur batao" / "tafseel" is an INSTRUCTION about your PREVIOUS answer in
  this conversation — not a new question. Re-give or expand your previous answer on the
  SAME topic in the requested language. NEVER jump to an unrelated subject on a follow-up;
  always use the conversation history to know the current topic.
- Lead with the ruling/answer, then the evidence and reasoning — not the other way round.
- Match depth to the question: a simple question gets a short, direct answer; a complex
  mas'ala gets structure. Do NOT pad, repeat, or add filler. No flattery.

═══════════════════════════════════════════════════════════════════════
FORMATTING (GitHub-flavoured Markdown is rendered — use it well, like a polished
chat assistant; do NOT overuse it)
═══════════════════════════════════════════════════════════════════════
- Open with one direct sentence that answers or frames the question.
- Use **bold** for the key ruling/verdict and important terms (halal, haram, makruh,
  fard, wajib, sunnat, mustahab).
- Use `##`/`###` headings only for longer, multi-part answers — not for a 2-line reply.
- Use `-` bullet lists for related points and `1. 2. 3.` numbered lists for steps or
  ordered rulings. Let consecutive items form ONE list (correct numbering follows).
- Use a Markdown **table** when comparing things (e.g. several views, or fard vs sunnat).
- Put a direct citation or short Arabic/Qur'an text in a `>` blockquote, then translate it.
- Cite sources inline and specifically, e.g. *(Bahar-e-Shariat, Hissa 3)* or
  *(Fatawa Razvia, jild 6)*. Keep Arabic terms, then give the meaning.
- Close with a one-line conclusion (khulasa) and, where rule 4 applies, the mufti note.
- Keep paragraphs short. Never wrap the whole answer in a code block.

═══════════════════════════════════════════════════════════════════════
LANGUAGE
═══════════════════════════════════════════════════════════════════════
- Reply in the SAME language AND script the user used. Decide carefully:
    • Urdu script (اردو) → reply in Urdu script.
    • ROMAN URDU = Urdu/Hindustani written in Latin letters (e.g. "talaq e salasa ka kya
      hukum hai", "namaz parhne ka tareeqa") → reply in ROMAN URDU. This is NOT English —
      NEVER answer Roman-Urdu questions in English.
    • Plain English → reply in English.   • Arabic → reply in Arabic.
  If the user mixes, mirror the mix.
- GREET ONLY IF THE USER GREETS. Add "وعلیکم السلام / Wa Alaikum Assalam" ONLY when the
  user's own message actually contains a salam (e.g. "assalam o alaikum", "السلام علیکم").
  If there is NO salam in their message, do NOT add any salam — just answer directly.
- Use respectful du'a phrases naturally (ﷺ for the Prophet, رضی اللہ عنہ, رحمۃ اللہ علیہ)
  without overdoing it.
- ARABIC STAYS IN ARABIC SCRIPT — ALWAYS. Any Qur'anic ayat, hadith text, du'a, or
  fixed Islamic phrase (e.g. بِسْمِ اللہِ الرَّحْمٰنِ الرَّحِیْمِ، سُبْحَانَ اللہ، مَاشَاءَ اللہ، اَلْحَمْدُ لِلّٰہِ،
  اِنْ شَاءَ اللہ، اللہُ اَکْبَر، اَعُوْذُ بِاللہ) MUST be written in Arabic script, NEVER
  transliterated into Latin/Roman/English letters. This holds even in a Roman-Urdu or
  English reply: write the Arabic phrase in Arabic script inline, then, if helpful, give
  its meaning in the reply's language. Example — WRONG: "…aur BISMILLAHIR RAHMANIR RAHIM
  padhein"; RIGHT: "…aur بِسْمِ اللہِ الرَّحْمٰنِ الرَّحِیْمِ padhein". The surrounding sentence
  follows the user's language; only the sacred Arabic text is kept in Arabic script.

═══════════════════════════════════════════════════════════════════════
SCOPE
═══════════════════════════════════════════════════════════════════════
- Answer Islamic questions: aqaid, fiqh, ibadat, mu'amalat, akhlaq, seerah, tareekh,
  du'a, and sincere personal/spiritual guidance within an Islamic frame — but always
  grounded in the provided sources per ACCURACY rule 1. If no source is provided for a
  substantive mas'ala, give the refusal, not a memory-based answer.
- For clearly non-Islamic requests (coding, general trivia, space, sports, etc.) reply
  that you only cover Islamic matters — this OUT-OF-SCOPE reply takes precedence over the
  "no mustanad reference" refusal, which is only for genuine Islamic masail:
  "معذرت، میں صرف اسلامی مسائل پر علم رکھتا ہوں۔ / Sorry, I only have knowledge about Islamic matters."
- An open-ended request like "koi hadees sunao", "ek dua batao", "نصیحت فرمائیے" IS a valid
  Islamic request: present whatever passage was supplied to you, with its exact reference.
  Do not treat it as too vague to answer.

═══════════════════════════════════════════════════════════════════════
IDENTITY (only when explicitly asked — never volunteer)
═══════════════════════════════════════════════════════════════════════
- Name: "AI MUFTI".
- Creator: "I was created by the world-renowned Naat reciter Sabter Raza Qadri
  (سبطر رضا قادری اختری)."
- Capabilities: answering Hanafi Fiqh questions with references from authentic
  Ahl-e-Sunnat sources and guidance on Islamic practice.
- Do not state your name, creator, maslak label, or capabilities unless asked.
"""

TITLE_PROMPT = (
    "Generate a very short topic title (4-6 words maximum) for an Islamic chat that starts "
    "with this message. Use the same language/script as the message. Reply with ONLY the "
    "title, no quotes, no punctuation at the end.\n\nMessage: "
)

# ================= APP SETUP =================
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _db_configured():
        try:
            db.init_db()
        except Exception as exc:  # pragma: no cover
            print(f"DB init failed: {exc}")
        if _gemini_configured():
            try:
                rag.init_rag()
            except Exception as exc:  # pragma: no cover
                print(f"RAG init failed: {exc}")
    yield


app = FastAPI(title="AI Mufti Backend", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


_default_origins = "https://digitalmufti.vercel.app,http://localhost:3000"
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Chat-Id", "X-Sources"],
)


def _encode_sources(passages) -> str:
    """Base64(JSON) so retrieved citations can ride along in a response header."""
    payload = json.dumps(rag.public_passages(passages), ensure_ascii=False)
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


# Short follow-ups / meta-instructions that mean "act on the previous answer", not a
# new topic — retrieving sources for these pulls in irrelevant context.
_META_FOLLOWUPS = {
    "urdu", "in urdu", "urdu me", "urdu mein", "اردو", "اردو میں",
    "english", "in english", "translate", "translate it",
    "explain", "explain more", "more", "detail", "details", "tafseel",
    "aur batao", "aur btao", "summarize", "short", "continue",
}


def _should_retrieve(user_input: str) -> bool:
    """Only run RAG for substantive questions, not bare follow-up commands.

    Skip ONLY on an exact meta-follow-up match. An earlier word-count guard (skip
    anything under 3 words) silently killed real requests like "Hadees sunao" and
    "Dua batao" — retrieval never ran and the model refused for lack of sources."""
    norm = user_input.strip().lower().rstrip("?.!۔؟ ")
    return norm not in _META_FOLLOWUPS


# Common Roman-Urdu function/topic words — used to detect when a Latin-script
# message is actually Urdu so we can force a Roman-Urdu reply (flash-lite otherwise
# defaults to English for "formal ruling" questions).
_ROMAN_URDU_WORDS = {
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "hai", "hain", "ho", "hota",
    "hoti", "kya", "kyun", "kaise", "kaisay", "kar", "kare", "karna", "karein",
    "kitne", "kitna", "konsa", "kaunsa", "kaun", "hukum", "hukm", "masla", "masala",
    "masail", "batao", "bataye", "bataiye", "jaiz", "najaiz", "gunah", "sahih",
    "ghalat", "namaz", "roza", "rozay", "wuzu", "wudu", "zakat", "talaq", "nikah",
    "farz", "sunnat", "wajib", "makruh", "halal", "haram", "mufti", "shariat",
    "deen", "imaan", "musafir", "sajda", "qaza", "fidya", "qurbani",
}
_ARABIC_RE = __import__("re").compile(r"[؀-ۿݐ-ݿﭐ-﻿]")


def _language_directive(text: str) -> str:
    """If the user wrote in Roman Urdu, return an explicit instruction to reply in
    Roman Urdu. Empty string when it's Urdu/Arabic script (model mirrors fine) or
    clearly plain English."""
    if _ARABIC_RE.search(text):
        return ""  # Urdu/Arabic script — let the model mirror the script
    words = __import__("re").findall(r"[A-Za-z]+", text.lower())
    if not words:
        return ""
    hits = sum(1 for w in words if w in _ROMAN_URDU_WORDS)
    if hits >= 2 or (hits >= 1 and len(words) <= 4):
        return (
            "[LANGUAGE: The user wrote in ROMAN URDU (Urdu in Latin letters). You MUST "
            "reply in Roman Urdu using Latin letters — do NOT reply in English and do NOT "
            "switch to Urdu/Arabic script.]\n\n"
        )
    return ""


# ================= SCHEMAS =================
class Message(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    chat_id: Optional[str] = None


class CreateChatRequest(BaseModel):
    title: Optional[str] = Field(default="New Chat", max_length=500)


class UpdateTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


# ================= HEALTH =================
@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Mufti Backend is running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_configured": _llm_configured(),
        "auth_configured": auth_configured(),
        "db_configured": _db_configured(),
        "provider": LLM_PROVIDER,
        "model": GROQ_MODEL if LLM_PROVIDER == "groq" else MODEL_NAME,
    }


# ================= LIBRARY (public, browsable) =================
# Read-only views over the ingested corpus. Public on purpose: these pages are the
# product's proof of authenticity and its indexable surface.

@app.get("/api/library/books")
async def library_books():
    """Every ingested book with its passage count, newest ingestion first."""
    def _query():
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT tags[1] AS slug,
                       split_part(MIN(reference), ',', 1) AS name,
                       COUNT(*) AS passages
                FROM sources
                WHERE tags IS NOT NULL AND array_length(tags, 1) >= 1
                GROUP BY tags[1]
                ORDER BY passages DESC;
                """
            )
            return [dict(r) for r in cur.fetchall()]

    try:
        return {"books": await run_in_threadpool(_query)}
    except Exception as exc:
        print(f"library_books failed: {exc}")
        raise HTTPException(status_code=503, detail="Library unavailable")


@app.get("/api/library/books/{slug}")
async def library_book(slug: str, page: int = 1, per_page: int = 20):
    """One book's passages, in ingestion (i.e. page) order."""
    page = max(1, page)
    per_page = min(max(1, per_page), 50)

    def _query():
        with db.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM sources WHERE tags[1] = %s;", (slug,))
            total = int(cur.fetchone()["n"])
            cur.execute(
                """
                SELECT title, reference, content
                FROM sources
                WHERE tags[1] = %s
                ORDER BY created_at, id
                LIMIT %s OFFSET %s;
                """,
                (slug, per_page, (page - 1) * per_page),
            )
            return total, [dict(r) for r in cur.fetchall()]

    try:
        total, rows = await run_in_threadpool(_query)
    except Exception as exc:
        print(f"library_book failed: {exc}")
        raise HTTPException(status_code=503, detail="Library unavailable")
    if not total:
        raise HTTPException(status_code=404, detail="Book not found")
    name = (rows[0]["reference"] or slug).split(",")[0] if rows else slug
    return {
        "slug": slug,
        "name": name,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "passages": rows,
    }


# ================= CHAT HISTORY (auth required) =================
@app.get("/api/chats")
async def get_chats(user_id: str = Depends(get_current_user_id)):
    chats = await run_in_threadpool(db.ChatRepository.get_chats, user_id)
    return {"chats": chats}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, user_id: str = Depends(get_current_user_id)):
    chat = await run_in_threadpool(db.ChatRepository.get_chat, chat_id, user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.post("/api/chats")
async def create_chat(request: CreateChatRequest, user_id: str = Depends(get_current_user_id)):
    title = request.title or "New Chat"
    return await run_in_threadpool(db.ChatRepository.create_chat, user_id, title)


@app.put("/api/chats/{chat_id}/title")
async def update_chat_title(
    chat_id: str, request: UpdateTitleRequest, user_id: str = Depends(get_current_user_id)
):
    chat = await run_in_threadpool(
        db.ChatRepository.update_chat_title, chat_id, user_id, request.title
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user_id: str = Depends(get_current_user_id)):
    success = await run_in_threadpool(db.ChatRepository.delete_chat, chat_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}


@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str, user_id: str = Depends(get_current_user_id)):
    messages = await run_in_threadpool(db.MessageRepository.get_messages, chat_id, user_id)
    return {"messages": messages}


# ================= GEMINI HELPERS =================
def _build_model() -> "genai.GenerativeModel":
    """Model with the system prompt as a real system_instruction and low temperature."""
    return genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "8192")),
        ),
    )


def _to_gemini_contents(history, user_input: str):
    """Map stored messages to Gemini's content format (roles: user / model, parts: [...])."""
    contents = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [msg["content"]]})
    contents.append({"role": "user", "parts": [user_input]})
    return contents


def _refine_title_in_background(chat_id: str, user_id: str, user_input: str):
    try:
        model = genai.GenerativeModel(
            MODEL_NAME, generation_config=genai.types.GenerationConfig(temperature=0.4)
        )
        resp = model.generate_content(TITLE_PROMPT + user_input)
        title = (resp.text or "").strip().strip('"').strip()
        if title and len(title.split()) <= 10:
            db.ChatRepository.update_chat_title(chat_id, user_id, title)
    except Exception as exc:  # pragma: no cover
        print(f"Title refinement failed: {exc}")


def _chunk_text(chunk) -> str:
    """Safely read a streamed chunk's text. Gemini's final chunk often has no text
    part (just a finish reason), and accessing `.text` then raises — so guard it."""
    try:
        return chunk.text or ""
    except Exception:
        return ""


def _chunk_finish_reason(chunk):
    """Best-effort read of a chunk's finish_reason (None if not present)."""
    try:
        fr = chunk.candidates[0].finish_reason
        return fr if fr is not None else None
    except Exception:
        return None


def _is_incomplete_finish(finish_reason) -> bool:
    """True when generation stopped for any reason other than a clean stop — e.g.
    MAX_TOKENS/length, SAFETY, RECITATION, content_filter. Handles both Gemini's
    enum and OpenAI/Groq's lowercase string ("stop"/"length"/"content_filter")."""
    if finish_reason is None:
        return False
    name = getattr(finish_reason, "name", str(finish_reason)).upper()
    return name not in ("STOP", "FINISH_REASON_UNSPECIFIED", "0", "1", "END_TURN")


def _is_quota_error(err_detail: str) -> bool:
    """True for 429 / quota / rate-limit errors (Gemini or Groq) vs. a generic failure."""
    d = err_detail.lower()
    return any(s in d for s in ("resourceexhausted", "429", "quota", "rate limit", "rate_limit"))


def _to_openai_messages(history, user_input: str):
    """Map system prompt + stored history to OpenAI/Groq chat message format."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})
    return messages


def _gemini_chunks(grounded_text: str, history):
    """Yield (text, finish_reason) from a Gemini streaming response."""
    model = _build_model()
    response = model.generate_content(_to_gemini_contents(history, grounded_text), stream=True)
    for chunk in response:
        yield _chunk_text(chunk), _chunk_finish_reason(chunk)


def _groq_chunks(grounded_text: str, history):
    """Yield (text, finish_reason) from Groq's OpenAI-compatible SSE stream.

    We read the stream as raw bytes and decode each line as UTF-8 ourselves
    instead of using an HTTP client's text auto-detection: while streaming,
    that detection latches onto the leading ASCII (e.g. "**") and decodes the
    rest as ASCII, mangling all the Urdu/Arabic into '?'. Manual UTF-8 is exact."""
    import requests

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": _to_openai_messages(history, grounded_text),
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "8192")),
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    # iter_lines on raw bytes is UTF-8-safe: 0x0A never appears mid-character.
    for raw_line in resp.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            choices = json.loads(data).get("choices") or []
        except ValueError:
            continue
        if not choices:
            continue
        choice = choices[0]
        text = (choice.get("delta") or {}).get("content") or ""
        yield text, choice.get("finish_reason")


def _stream_text(user_text: str, grounded_text: str, history, persist=None):
    """Stream chunks. `user_text` is persisted; `grounded_text` (with retrieved
    citations) is what the model actually sees. `persist` = (chat_id, user_id)."""
    full = ""
    failed = False
    err_detail = ""
    finish_reason = None
    chunk_source = _groq_chunks if LLM_PROVIDER == "groq" else _gemini_chunks

    try:
        if persist:
            chat_id, user_id = persist
            db.MessageRepository.create_message(chat_id, user_id, "user", user_text)

        for text, fr in chunk_source(grounded_text, history):
            if text:
                full += text
                yield text
            if fr is not None:
                finish_reason = fr
    except Exception as exc:
        failed = True
        err_detail = f"{type(exc).__name__}: {exc}"
        print(f"Generation error: {err_detail}")

    incomplete = _is_incomplete_finish(finish_reason)
    quota = _is_quota_error(err_detail)
    debug = os.getenv("DEBUG_ERRORS") == "1"

    if not full and failed:
        # Nothing was produced at all — show a full fallback message.
        suffix = f"\n\n[debug: {err_detail}]" if debug else ""
        if quota:
            yield (
                "معذرت، اس وقت سروس کی حد (rate limit) عبور ہو گئی ہے۔ "
                "تھوڑی دیر (تقریباً ایک منٹ) بعد دوبارہ کوشش کریں۔ / "
                "Sorry, the AI service hit its rate limit. "
                "Please try again in a minute." + suffix
            )
        else:
            yield (
                "معذرت، اس وقت جواب تیار کرنے میں دشواری ہو رہی ہے۔ / "
                "Sorry, I could not generate a response right now. Please try again." + suffix
            )
    elif full and (failed or incomplete):
        # We streamed PARTIAL text and then got cut off (mid-stream error, quota
        # throttle, or a non-STOP finish like SAFETY/RECITATION/MAX_TOKENS).
        # Tell the user instead of leaving a silently truncated answer.
        suffix = f" [debug: {err_detail or finish_reason}]" if debug else ""
        note = (
            "\n\n⚠️ "
            + (
                "جواب سروس کی حد کی وجہ سے مکمل نہیں ہو سکا۔ / "
                "The answer was cut off because the AI service hit its usage limit. "
                if quota
                else "جواب مکمل نہیں ہو سکا (سرور کی رکاوٹ)۔ / "
                "The answer was cut off before it finished. "
            )
            + "براہِ کرم دوبارہ بھیجیں۔ / Please send the question again."
            + suffix
        )
        full += note
        yield note

    # Persist the assistant reply separately so a DB hiccup can't corrupt the output.
    if persist and full:
        try:
            chat_id, user_id = persist
            db.MessageRepository.create_message(chat_id, user_id, "assistant", full)
        except Exception as exc:
            print(f"Persist assistant message failed: {exc}")


def _stream_response(user_text: str, grounded_text: str, history, persist=None):
    """Encode each chunk to UTF-8 bytes ourselves so the ASGI layer never re-encodes
    the stream with a platform-default charset (which mangled Urdu/Arabic to '?')."""
    for piece in _stream_text(user_text, grounded_text, history, persist):
        yield piece.encode("utf-8")


# ================= CHAT =================
@app.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: Message,
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    if not _llm_configured():
        missing = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "GEMINI_API_KEY"
        raise HTTPException(status_code=503, detail=f"Server missing {missing}")

    user_input = body.content.strip()
    chat_id = body.chat_id

    # RAG: retrieve supporting source passages and ground the prompt — but ONLY for
    # substantive questions. Short / meta follow-ups like "urdu", "explain more",
    # "in english" carry no topic, and embedding them pulls in random sources that
    # derail the answer onto an unrelated subject.
    retrieval_attempted = _should_retrieve(user_input)
    passages = []
    if retrieval_attempted:
        # Topic-less requests ("hadees sunao") have nothing to match on, so they
        # never clear the similarity threshold — draw a random passage instead.
        intent = rag.browse_intent(user_input)
        if intent:
            passages = await run_in_threadpool(rag.browse, intent[0], intent[1])
        if not passages:
            passages = await run_in_threadpool(rag.retrieve, user_input)
    grounded_input = rag.build_grounded_input(user_input, passages, retrieval_attempted)

    # Force a Roman-Urdu reply when the user wrote Roman Urdu (the model sees this
    # directive but it is NOT persisted as the user's message).
    directive = _language_directive(user_input)
    if directive:
        grounded_input = directive + grounded_input

    # Guests (no token) or no DB -> stateless, nothing persisted.
    if not user_id or not _db_configured():
        headers = {"X-Chat-Id": chat_id or str(uuid4())}
        if passages:
            headers["X-Sources"] = _encode_sources(passages)
        return StreamingResponse(
            _stream_response(user_input, grounded_input, history=[]),
            media_type="text/plain; charset=utf-8",
            headers=headers,
        )

    # Signed-in user with DB: load (and verify-owned) history, persist new messages.
    if chat_id:
        owned = await run_in_threadpool(db.ChatRepository.get_chat, chat_id, user_id)
        if not owned:
            raise HTTPException(status_code=404, detail="Chat not found")
        history = await run_in_threadpool(db.MessageRepository.get_messages, chat_id, user_id)
    else:
        # Instant heuristic title now; refine with AI in the background (no first-token delay).
        chat_obj = await run_in_threadpool(
            db.ChatRepository.create_chat, user_id, db.generate_title_from_message(user_input)
        )
        chat_id = str(chat_obj["id"])
        history = []
        # AI title refinement costs a SECOND model call per new chat — off by
        # default to conserve quota; the heuristic title above is already set.
        if os.getenv("AI_TITLES") == "1":
            threading.Thread(
                target=_refine_title_in_background,
                args=(chat_id, user_id, user_input),
                daemon=True,
            ).start()

    headers = {"X-Chat-Id": chat_id}
    if passages:
        headers["X-Sources"] = _encode_sources(passages)
    return StreamingResponse(
        _stream_response(user_input, grounded_input, history, persist=(chat_id, user_id)),
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )
