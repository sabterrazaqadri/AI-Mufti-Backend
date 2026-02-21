---
title: AI Mufti Backend
emoji: 🕌
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
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

## License

MIT
