import os
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None

@app.on_event("startup")
async def startup_event():
    global classifier
    logger.info("Loading sentiment analysis model...")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logger.info("Model loaded successfully!")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/api")
def read_root():
    return {"message": "AI Sentiment Analysis API"}

@app.post("/analyze/")
async def analyze_text(text: str = Form(...)):
    global classifier
    if classifier is None:
        logger.info("Loading model on demand...")
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    result = classifier(text)
    return {"label": result[0]['label'], "score": float(result[0]['score'])}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)