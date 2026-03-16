from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import Optional
import uuid

from .stt import transcribe_audio
from .tts import synthesize_speech
from .agent import run_agent, vector_store
from .pdf_processor import extract_text_from_pdf, chunk_text
from .embeddings import get_embedding
from .config import PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP, logger

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversations = {}

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        text = await transcribe_audio(audio)
        return {"text": text}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(500, "Transcription failed")

@app.post("/api/chat")
async def chat(message: str = Form(...), session_id: Optional[str] = Form(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    history = conversations.get(session_id, [])
    
    response_text = await run_agent(message, history)
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response_text})
    conversations[session_id] = history[-20:]
    
    return {"response": response_text, "session_id": session_id}

@app.post("/api/tts")
async def text_to_speech(text: str = Form(...)):
    try:
        return await synthesize_speech(text)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(500, "TTS failed")

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        chunks = chunk_text(text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
        
        embeddings = []
        valid_chunks = []
        for chunk in chunks:
            try:
                emb = await get_embedding(chunk)
                embeddings.append(emb)
                valid_chunks.append(chunk)
            except:
                continue
        
        vector_store.add_embeddings(embeddings, valid_chunks)
        return {"message": f"Processed {len(valid_chunks)} chunks"}
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise HTTPException(500, "PDF processing failed")

@app.get("/api/health")
async def health():
    return {"status": "ok"}