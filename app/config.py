import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TTS_MODEL = "coqui/XTTS-v2"

FAISS_INDEX_PATH = "faiss_index.bin"
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50

logger.add("app.log", rotation="10 MB")