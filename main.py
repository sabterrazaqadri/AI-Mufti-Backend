from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from starlette.requests import Request
from uuid import uuid4
import google.generativeai as genai
import os
import database as db


def _db_configured() -> bool:
    return bool(os.getenv("DATABASE_URL"))

# Load .env file
load_dotenv()

# Validate GEMINI key
api_key = os.getenv("GEMINI_API_KEY")

def _gemini_configured() -> bool:
    return bool(api_key)

if api_key:
    genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"

# Initialize FastAPI app
app = FastAPI()


@app.on_event("startup")
def _startup_init_db():
    # Ensure DB tables exist (safe to run multiple times)
    try:
        db.init_db()
    except Exception as e:
        # Don't crash the server on startup; chat endpoints will surface error
        print(f"DB init failed: {e}")


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    # Ensure errors still return a response body and go through middleware
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://digitalmufti.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Chat-Id"],
)

# System prompt for the AI
SYSTEM_PROMPT = """You are a qualified Islamic scholar from the Sunni Hanafi Ahl-e-Sunnat wa Jama'at school of thought.
Provide answers strictly based on Hanafi Fiqh, referencing authentic and classical Sunni sources
such as Fatawa Razvia, Bahar-e-Shariat, Hidayah, and similar works.

Always give Qur'an, Hadith, or authentic Hanafi references.

Do not answer non-Islamic questions. Reply:
"معذرت، میں صرف اسلامی مسائل پر علم رکھتا ہوں۔ / Sorry, I only have knowledge about Islamic matters."

always responds in a clear, structured, and well-organized format.

Follow these rules strictly for every response:
1. Start with a short, clear introductory paragraph.
2. Break the main content into logical sections.
3. Use numbered points (1, 2, 3…) for explanations.
4. Use bullet points (•) for lists or sub-points.
5. Keep paragraphs concise and focused on one idea.
6. Maintain a logical flow from basic to advanced concepts.
7. Use simple, formal, and explanatory language.
8. Highlight key terms where helpful.
9. Avoid long unbroken text blocks.
10. Never use any markdown formatting (no **, *, #, ##, etc.) - respond with plain text only
11. End with a brief summary or conclusion when appropriate.

Your goal is to maximize clarity, readability, and structured understanding in every answer.

Reply in the same language as the user is using (eg. Roman urdu == Roman Urdu , Urdu == Urdu , English == English)

If user asks about your name, say: "AI MUFTI"
If user asks about your creator/developer, say:
"I am created by World Famous Naat Recitor Sabter Raza Qadri (سبطر رضا قادری اختری)"
If User Dont ask about your name or your creator name , dont mention it in responces
"""


# Message schema
class Message(BaseModel):
    user_id: str
    content: str
    chat_id: Optional[str] = None


# Chat schemas
class CreateChatRequest(BaseModel):
    user_id: str
    title: Optional[str] = None


class UpdateTitleRequest(BaseModel):
    user_id: str
    title: str


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Mufti Backend is running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
        "model": MODEL_NAME,
    }


# ============ CHAT HISTORY API ENDPOINTS ============


@app.get("/api/chats")
async def get_chats(user_id: str):
    chats = db.ChatRepository.get_chats(user_id)
    return {"chats": chats}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, user_id: str):
    chat = db.ChatRepository.get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.post("/api/chats")
async def create_chat(request: CreateChatRequest):
    title = request.title or "New Chat"
    chat = db.ChatRepository.create_chat(request.user_id, title)
    return chat


@app.put("/api/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, request: UpdateTitleRequest):
    chat = db.ChatRepository.update_chat_title(chat_id, request.user_id, request.title)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user_id: str):
    success = db.ChatRepository.delete_chat(chat_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}


@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str, user_id: str):
    messages = db.MessageRepository.get_messages(chat_id, user_id)
    return {"messages": messages}


# ============ CHAT API WITH STREAMING ============


async def generate_title_with_ai(user_input: str) -> str:
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)

        prompt = (
            "Generate a very short, 4-6 word maximum, topic-based title for an Islamic chat session "
            f"starting with this message: '{user_input}'. "
            "The title should be in the same language as the message (Urdu, Roman Urdu, or English). "
            "Do not use quotes or special characters."
        )

        response = model.generate_content(prompt)
        title = response.text.strip()

        if not title or len(title.split()) > 10:
            return db.generate_title_from_message(user_input)

        return title
    except Exception as e:
        print(f"AI Title generation failed: {e}")
        return db.generate_title_from_message(user_input)


def generate_stream_response(user_id: str, user_input: str, chat_id: str = None):
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)

    messages_for_ai = []

    if chat_id:
        previous_messages = db.MessageRepository.get_messages(chat_id, user_id)
        for msg in previous_messages:
            role = "user" if msg["role"] == "user" else "model"
            messages_for_ai.append({"role": role, "parts": [msg["content"]]})

    if not messages_for_ai:
        messages_for_ai.append(
            {"role": "user", "parts": [SYSTEM_PROMPT + "\n\nUser question: " + user_input]}
        )
    else:
        messages_for_ai.append({"role": "user", "parts": [user_input]})

    try:
        response = model.generate_content(messages_for_ai, stream=True)

        db.MessageRepository.create_message(chat_id, "user", user_input)

        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        if full_response:
            db.MessageRepository.create_message(chat_id, "assistant", full_response)

    except Exception as e:
        yield f"Error: {str(e)}"


@app.post("/chat")
async def chat(request: Message, chat_id: str = None):
    user_input = request.content
    user_id = request.user_id

    if not _gemini_configured():
        raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY")

    actual_chat_id = chat_id or request.chat_id

    # If DB isn't configured, fall back to stateless chat (no persistence)
    if not _db_configured():
        if not actual_chat_id:
            actual_chat_id = str(uuid4())
        response_stream = _stateless_stream_response(user_input)
        return StreamingResponse(
            response_stream,
            media_type="text/plain",
            headers={"X-Chat-Id": actual_chat_id},
        )

    if not actual_chat_id:
        title = await generate_title_with_ai(user_input)
        chat_obj = db.ChatRepository.create_chat(user_id, title)
        actual_chat_id = str(chat_obj["id"])

    response_stream = generate_stream_response(user_id, user_input, actual_chat_id)

    return StreamingResponse(
        response_stream,
        media_type="text/plain",
        headers={"X-Chat-Id": actual_chat_id},
    )


def _stateless_stream_response(user_input: str):
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)

    messages_for_ai = [
        {"role": "user", "parts": [SYSTEM_PROMPT + "\n\nUser question: " + user_input]}
    ]

    try:
        response = model.generate_content(messages_for_ai, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {str(e)}"
        return

    return



@app.post("/chat/{chat_id}")
async def chat_with_history(chat_id: str, request: Message):
    return await chat(request, chat_id)
