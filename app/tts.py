import httpx
from fastapi.responses import Response
from .config import HF_API_TOKEN, TTS_MODEL, logger

async def synthesize_speech(text: str) -> Response:
    api_url = f"https://router.huggingface.co/hf-inference/models/{TTS_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=payload, timeout=60)
    
    if response.status_code != 200:
        logger.error(f"TTS API error: {response.text}")
        raise Exception("TTS failed")
    
    return Response(content=response.content, media_type="audio/wav")