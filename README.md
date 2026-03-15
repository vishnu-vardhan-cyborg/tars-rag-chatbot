# RAG Chatbot using Groq + FastAPI

This project demonstrates a Retrieval Augmented Generation (RAG) chatbot.

Stack:
- FastAPI backend
- FAISS vector database
- Sentence Transformers embeddings
- Groq Llama3 LLM
- HTML CSS JS frontend

Steps to run:

1. Install dependencies

pip install -r requirements.txt

2. Add Groq API key in .env

3. Start server

uvicorn main:app --reload

4. Open frontend/index.html