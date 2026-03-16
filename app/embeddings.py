import httpx
import numpy as np
from .config import HF_API_TOKEN, EMBEDDING_MODEL, logger

async def get_embedding(text: str) -> np.ndarray:
    api_url = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        logger.error(f"Embedding API error: {response.text}")
        raise Exception("Embedding failed")
    
    embedding = np.array(response.json(), dtype=np.float32)
    return embedding