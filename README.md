---
title: AI Mufti Backend
emoji: 🕌
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# AI Mufti Backend

An Islamic Q&A chatbot backend powered by Google Gemini AI, built with FastAPI.

## Features

- Answers Islamic questions based on Hanafi Fiqh
- Streaming responses for real-time interaction
- Chat history persistence with database support
- RESTful API endpoints for chat management
- CORS enabled for frontend integration

## API Endpoints

- `GET /` - Health check
- `GET /health` - API health status
- `POST /chat` - Send a message and get streaming response
- `GET /api/chats` - Get all chats for a user
- `POST /api/chats` - Create a new chat
- `GET /api/chats/{chat_id}` - Get a specific chat
- `DELETE /api/chats/{chat_id}` - Delete a chat
- `GET /api/chats/{chat_id}/messages` - Get messages for a chat

## Environment Variables

- `GEMINI_API_KEY` - Your Google Gemini API key
- `DATABASE_URL` - Optional database URL for chat persistence

## RAG: ingesting books

Answers are grounded in real book text stored in Postgres (pgvector). Citations
(jild, bab, masla number) are machine-generated at ingestion — the model is
instructed to only quote reference numbers that come from retrieved excerpts.

Pipeline (run from `backend/`):

```bash
# 1. Scrape a book's Unicode text from the Dawat-e-Islami online reader
#    (the PDFs use non-Unicode fonts and are NOT extractable)
python scrape_book.py bahar-e-shariat-jild-1 ../data/bahar_e_shariat/jild_1

# 2. Chunk (masla-wise) + embed + store in Neon. Resume-able: free-tier Gemini
#    embedding quota (requests/day) may stop it mid-run — just re-run the same
#    command later; ingest_checkpoint.json skips completed pages.
python -u ingest_book.py ../data/bahar_e_shariat/jild_1 --jild 1

# 3. Smoke-test retrieval quality (Urdu / Roman Urdu / English questions)
python verify_rag.py
```

Both scripts read `GEMINI_API_KEY` and `DATABASE_URL` from `.env`.

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Docker

This space runs using Docker. Build and run:

```bash
docker build -t ai-mufti-backend .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key ai-mufti-backend
```
