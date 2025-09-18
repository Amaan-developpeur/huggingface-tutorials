from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="Text Summarization Using Hugging Face")

# Load Hugging Face Summarization Pipeline
summarizer = pipeline("summarization", model="t5-small")

# Input schema
class PromptInput(BaseModel):
    text: str = Field(..., description="Text to summarize")

# Endpoint
@app.post("/summarize")
async def text_summarization(data: PromptInput):
    summary = summarizer(
        data.text,
        max_length=100,
        min_length=40,
        do_sample=False
    )
    return {"Generated Summary": summary[0]["summary_text"]}
