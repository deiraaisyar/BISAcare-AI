from fastapi import FastAPI, Request
from pydantic import BaseModel
from features.bisabot.bisabot import ask_bisabot, get_chat_history, clear_chat_history

app = FastAPI(title="BISAcare")

class Query(BaseModel):
    question: str

@app.post("/bisabot")
async def chat(query: Query):
    response = ask_bisabot(query.question)
    return {"answer": response}

@app.get("/bisabot/history")
async def get_history():
    history = get_chat_history()
    return {"history": history}

@app.delete("/bisabot/history")
async def clear_history():
    clear_chat_history()
    return {"message": "Chat history cleared successfully"}

