import httpx
from fastapi import UploadFile
from .config import HF_API_TOKEN, STT_MODEL, logger

async def transcribe_audio(file: UploadFile) -> str:
    audio_bytes = await file.read()
    api_url = f"https://router.huggingface.co/hf-inference/models/{STT_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, data=audio_bytes, timeout=60)
    
    if response.status_code != 200:
        logger.error(f"STT API error: {response.text}")
        raise Exception("STT failed")
    
    result = response.json()
    logger.info(f"Transcription result: {result}")
    return result.get("text", "")