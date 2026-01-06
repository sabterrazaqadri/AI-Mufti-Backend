from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import google.generativeai as genai
import os
import database as db

# Load .env file
load_dotenv()

# Validate GEMINI key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY missing in .env")

genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
10. Avoid Rendering Responces in Markdown Format
11. End with a brief summary or conclusion when appropriate.

Your goal is to maximize clarity, readability, and structured understanding in every answer.

Reply in the same language as the user is using (eg. Roman urdu == Roman Urdu , Urdu == Urdu , English == English)

If user asks about your name, say: "AI MUFTI"
If user asks about your creator/developer, say:
"I am created by World Famous Naat Recitor Sabter Raza Qadri (سبطر رضا قادری اختری)"
If User Dont ask about your name or your creator name , dont mention it in responces
"""

# Track per-user chat history (in memory for conversation context)
user_chats_context = {}

# Message schema
class Message(BaseModel):
    user_id: str
    content: str

# Chat schemas
class CreateChatRequest(BaseModel):
    user_id: str
    title: Optional[str] = None

class UpdateTitleRequest(BaseModel):
    user_id: str
    title: str

class DeleteChatRequest(BaseModel):
    user_id: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Mufti Backend is running"}


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
        "model": "gemini-2.5-flash"
    }


# ============ CHAT HISTORY API ENDPOINTS ============

@app.get("/api/chats")
async def get_chats(user_id: str):
    """Get all chats for a user"""
    chats = db.ChatRepository.get_chats(user_id)
    return {"chats": chats}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, user_id: str):
    """Get a specific chat"""
    chat = db.ChatRepository.get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.post("/api/chats")
async def create_chat(request: CreateChatRequest):
    """Create a new chat"""
    title = request.title or "New Chat"
    chat = db.ChatRepository.create_chat(request.user_id, title)
    return chat


@app.put("/api/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, request: UpdateTitleRequest):
    """Update chat title"""
    chat = db.ChatRepository.update_chat_title(chat_id, request.user_id, request.title)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, user_id: str):
    """Delete a chat"""
    success = db.ChatRepository.delete_chat(chat_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}


@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str, user_id: str):
    """Get all messages for a chat"""
    messages = db.MessageRepository.get_messages(chat_id, user_id)
    return {"messages": messages}


# ============ CHAT API WITH STREAMING ============

async def generate_title_with_ai(user_input: str) -> str:
    """Generate a short topic-based title using Gemini"""
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

        prompt = f"Generate a very short, 4-6 word maximum, topic-based title for an Islamic chat session starting with this message: '{user_input}'. The title should be in the same language as the message (Urdu, Roman Urdu, or English). Do not use quotes or special characters."

        response = model.generate_content(prompt)
        title = response.text.strip()

        # Fallback if AI produces something too long or empty
        if not title or len(title.split()) > 10:
            return db.generate_title_from_message(user_input)

        return title
    except Exception as e:
        print(f"AI Title generation failed: {e}")
        return db.generate_title_from_message(user_input)

def generate_stream_response(user_id: str, user_input: str, chat_id: str = None):
    """Generate streaming response for a given user and input"""
    generation_config = genai.types.GenerationConfig(temperature=0.1)
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

    # Build conversation history for AI context
    messages_for_ai = []

    # If chat_id provided, load previous messages
    if chat_id:
        previous_messages = db.MessageRepository.get_messages(chat_id, user_id)
        for msg in previous_messages:
            # Gemini expects 'user' and 'model' roles
            role = "user" if msg["role"] == "user" else "model"
            messages_for_ai.append({"role": role, "parts": [msg["content"]]})

    # Add system prompt for first message
    if not messages_for_ai:
        messages_for_ai.append({"role": "user", "parts": [SYSTEM_PROMPT + "\n\nUser question: " + user_input]})
    else:
        messages_for_ai.append({"role": "user", "parts": [user_input]})

    try:
        # Create chat if doesn't exist (Move this before streaming starts for better title generation)
        if not chat_id:
            # We'll generate a title placeholder and update it later if needed,
            # but for first message we can try generating it now
            from asyncio import run
            # Since this is a generator, we might want to handle title creation synchronously or separately
            # but for now let's keep it simple
            title = db.generate_title_from_message(user_input)
            chat = db.ChatRepository.create_chat(user_id, title)
            chat_id = str(chat["id"])

        # Use streaming with stream=True as keyword argument
        response = model.generate_content(
            messages_for_ai,
            stream=True
        )

        # Save user message
        db.MessageRepository.create_message(chat_id, "user", user_input)

        # Stream the response chunks
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        # Save assistant message
        if full_response:
            db.MessageRepository.create_message(chat_id, "assistant", full_response)

        # Optional: update title after first exchange for better context
        # but the requirement says "first message", so we did that above.

    except Exception as e:
        yield f"Error: {str(e)}"


@app.post("/chat")
async def chat(request: Message, chat_id: str = None):
    """Streaming chat endpoint with optional chat_id for conversation context"""
    user_input = request.content
    user_id = request.user_id

    actual_chat_id = chat_id or request.chat_id if hasattr(request, 'chat_id') else None

    # If first message (no chat_id), generate AI title and create chat before streaming
    if not actual_chat_id:
        title = await generate_title_with_ai(user_input)
        chat_obj = db.ChatRepository.create_chat(user_id, title)
        actual_chat_id = str(chat_obj["id"])

    # Now start the streaming response with the established actual_chat_id
    response_stream = generate_stream_response(user_id, user_input, actual_chat_id)

    # We need to return the chat_id in headers so frontend can pick it up
    return StreamingResponse(
        response_stream,
        media_type="text/plain",
        headers={"X-Chat-Id": actual_chat_id}
    )



@app.post("/chat/{chat_id}")
async def chat_with_history(chat_id: str, request: Message):
    """Chat endpoint with existing chat history"""
    return await chat(request, chat_id)