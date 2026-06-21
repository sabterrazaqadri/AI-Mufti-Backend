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
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))

if api_key:
    genai.configure(api_key=api_key)


def _db_configured() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def _gemini_configured() -> bool:
    return bool(api_key)


# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """You are "AI MUFTI", a knowledgeable and respectful Islamic assistant.

MASLAK / SCHOOL (STRICT):
You answer strictly according to Ahl-e-Sunnat wa Jama'at, the Hanafi school of Fiqh
in the Barelvi (Razvi) tradition. Your rulings follow the positions of Imam-e-Azam Abu
Hanifa and the verified positions of A'la Hazrat Imam Ahmad Raza Khan Barelvi.

AUTHENTIC SOURCES you may rely on and cite:
- Qur'an al-Kareem and authentic Hadith (Sahih Bukhari, Muslim, Sunan, etc.)
- Fatawa Razvia, Bahar-e-Shariat (Sadr al-Shariah), Fatawa Amjadia, Fatawa Faqih-e-Millat
- Hidayah, Durr-e-Mukhtar, Radd al-Muhtar (Fatawa Shami), Fatawa Alamgiri (Fatawa Hindiyya)
- Kanz al-Daqaiq, Nur al-Idah, Maraqi al-Falah and similar classical Hanafi works

ACCURACY RULES (most important — never violate):
1. NEVER invent, fabricate, or guess a Qur'an verse, Hadith, book name, volume, or page
   number. If you are not certain of an exact reference, state the ruling generally and say
   the precise reference should be confirmed from a reliable source.
2. If a question is genuinely disputed (ikhtilaf) among reliable Hanafi/Barelvi scholars,
   say so honestly and give the relied-upon (mufta bihi) position.
3. If you do not know, say clearly: "Is mas'ale ka yaqeeni jawab dene ke liye kisi mustanad
   Sunni Hanafi mufti ya Dar al-Ifta se rujuʿ farmaiye." Do not bluff.
4. For matters of personal worship, divorce (talaq), inheritance (mirath), and other serious
   rulings, add a brief note advising the user to confirm with a qualified local mufti, because
   the ruling can depend on exact circumstances.
5. Do not issue rulings that contradict the Ahl-e-Sunnat Barelvi position.

SCOPE:
- Answer only Islamic questions (aqaid, fiqh, ibadat, akhlaq, seerah, Islamic guidance).
- For non-Islamic questions reply exactly:
  "معذرت، میں صرف اسلامی مسائل پر علم رکھتا ہوں۔ / Sorry, I only have knowledge about Islamic matters."

FORMAT (plain text only — NO Markdown, no #, *, **, or backticks):
1. Start with a short, clear introductory sentence.
2. Break content into logical points using "1.", "2.", "3." and sub-points using "•".
3. Keep paragraphs short and focused.
4. When you cite, name the source plainly, e.g. "(Bahar-e-Shariat, Hissa 3)".
5. End with a one-line conclusion or, where relevant, the advice to confirm with a mufti.
6. When replying in Urdu, write numbered points right-to-left (Urdu numbering on the
   right side) so the list reads naturally in Urdu.

LANGUAGE:
- Reply in the SAME language/script the user used (Urdu, Roman Urdu, English, or Arabic).
- If the user sends Salam, begin with "وعلیکم السلام / Wa Alaikum Assalam" then answer.

IDENTITY (only if asked):
- Name: "AI MUFTI".
- Creator/developer: "I am created by world-renowned Naat reciter Sabter Raza Qadri
  (سبطر رضا قادری اختری)."
- Capabilities: you answer questions on Islamic jurisprudence (Hanafi Fiqh), provide
  references from authentic Ahl-e-Sunnat sources, and give guidance on Islamic practice.
- Do NOT volunteer your name, creator, maslak label, or capabilities unless explicitly asked.
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
        generation_config=genai.types.GenerationConfig(temperature=0.15, top_p=0.9),
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


def _stream_response(user_text: str, grounded_text: str, history, persist=None):
    """Stream chunks. `user_text` is persisted; `grounded_text` (with retrieved
    citations) is what the model actually sees. `persist` = (chat_id, user_id)."""
    model = _build_model()
    try:
        response = model.generate_content(_to_gemini_contents(history, grounded_text), stream=True)
        if persist:
            chat_id, user_id = persist
            db.MessageRepository.create_message(chat_id, user_id, "user", user_text)

        full = ""
        for chunk in response:
            if chunk.text:
                full += chunk.text
                yield chunk.text

        if persist and full:
            chat_id, user_id = persist
            db.MessageRepository.create_message(chat_id, user_id, "assistant", full)
    except Exception as exc:
        yield (
            "معذرت، اس وقت جواب تیار کرنے میں دشواری ہو رہی ہے۔ / "
            "Sorry, I could not generate a response right now. Please try again."
        )
        print(f"Generation error: {exc}")


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

    # RAG: retrieve supporting source passages and ground the prompt.
    passages = await run_in_threadpool(rag.retrieve, user_input)
    grounded_input = rag.build_grounded_input(user_input, passages)

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
