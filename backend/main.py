from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.evaluation import compute_metrics
from backend.rag import ask_question
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("../frontend/index.html")


@app.post("/chat")
def chat(data: Question):

    answer, context_docs = ask_question(data.question)
    metrics = compute_metrics(data.question, answer, context_docs)

    return {"answer": answer, "metrics": metrics}
