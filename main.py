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

if api_key:
    genai.configure(api_key=api_key)


def _db_configured() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def _gemini_configured() -> bool:
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
1. NEVER invent or guess a Qur'an verse, Hadith, book name, volume, or page number.
   If unsure of an exact reference, give the ruling in general terms and say the precise
   reference should be verified — do not fabricate a citation to look authoritative.
2. Distinguish clearly between: (a) a firm, agreed ruling, (b) the relied-upon
   (mufta bihi) position where there is ikhtilaf, and (c) your own reasoning/inference.
   If scholars differ, say so and give the mufta bihi view.
3. If you genuinely do not know, say so plainly and advise:
   "Is mas'ale ka yaqeeni jawab ke liye kisi mustanad Sunni Hanafi mufti ya Dar al-Ifta
   se rujuʿ farmaiye." Never bluff.
4. For talaq (divorce), mirath (inheritance), serious financial/medical, or anything
   that depends on exact circumstances, give the general ruling AND advise confirming
   with a qualified local mufti, because details change the verdict.
5. You may receive PRIVATE background reference excerpts inside the prompt. Use them
   silently to ground and verify your answer. NEVER reveal them: do not mention "excerpts"
   or "provided references", do not use bracket numbers like [1]/[2], and never tell the
   user whether a reference was found or that something "is not covered" in what you were
   given. If they are relevant, weave the knowledge in and cite the real source by name
   (only if certain); if not, answer from your own Hanafi knowledge as if nothing was given.

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

═══════════════════════════════════════════════════════════════════════
SCOPE
═══════════════════════════════════════════════════════════════════════
- Answer Islamic questions: aqaid, fiqh, ibadat, mu'amalat, akhlaq, seerah, tareekh,
  du'a, and sincere personal/spiritual guidance within an Islamic frame.
- For clearly non-Islamic requests (coding, general trivia, etc.) reply exactly:
  "معذرت، میں صرف اسلامی مسائل پر علم رکھتا ہوں۔ / Sorry, I only have knowledge about Islamic matters."

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
    """Only run RAG for substantive questions, not bare follow-up commands."""
    norm = user_input.strip().lower().rstrip("?.!۔؟ ")
    if norm in _META_FOLLOWUPS:
        return False
    # Fewer than 3 words is almost never a real, source-worthy question and its
    # embedding tends to match random seed sources.
    return len(norm.split()) >= 3


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
        "api_key_configured": _gemini_configured(),
        "auth_configured": auth_configured(),
        "db_configured": _db_configured(),
        "model": MODEL_NAME,
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


def _stream_response(user_text: str, grounded_text: str, history, persist=None):
    """Stream chunks. `user_text` is persisted; `grounded_text` (with retrieved
    citations) is what the model actually sees. `persist` = (chat_id, user_id)."""
    model = _build_model()
    full = ""
    failed = False
    err_detail = ""

    try:
        response = model.generate_content(_to_gemini_contents(history, grounded_text), stream=True)
        if persist:
            chat_id, user_id = persist
            db.MessageRepository.create_message(chat_id, user_id, "user", user_text)

        for chunk in response:
            text = _chunk_text(chunk)
            if text:
                full += text
                yield text
    except Exception as exc:
        failed = True
        err_detail = f"{type(exc).__name__}: {exc}"
        print(f"Generation error: {err_detail}")

    # Only show the fallback if we produced NOTHING — never append it to a real answer.
    if not full and failed:
        # Surface the error detail only when explicitly debugging.
        suffix = f"\n\n[debug: {err_detail}]" if os.getenv("DEBUG_ERRORS") == "1" else ""
        yield (
            "معذرت، اس وقت جواب تیار کرنے میں دشواری ہو رہی ہے۔ / "
            "Sorry, I could not generate a response right now. Please try again." + suffix
        )

    # Persist the assistant reply separately so a DB hiccup can't corrupt the output.
    if persist and full:
        try:
            chat_id, user_id = persist
            db.MessageRepository.create_message(chat_id, user_id, "assistant", full)
        except Exception as exc:
            print(f"Persist assistant message failed: {exc}")


# ================= CHAT =================
@app.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: Message,
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    if not _gemini_configured():
        raise HTTPException(status_code=503, detail="Server missing GEMINI_API_KEY")

    user_input = body.content.strip()
    chat_id = body.chat_id

    # RAG: retrieve supporting source passages and ground the prompt — but ONLY for
    # substantive questions. Short / meta follow-ups like "urdu", "explain more",
    # "in english" carry no topic, and embedding them pulls in random sources that
    # derail the answer onto an unrelated subject.
    if _should_retrieve(user_input):
        passages = await run_in_threadpool(rag.retrieve, user_input)
    else:
        passages = []
    grounded_input = rag.build_grounded_input(user_input, passages)

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
