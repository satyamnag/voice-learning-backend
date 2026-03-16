import openai
from .config import OPENAI_API_KEY, logger
from .embeddings import get_embedding
from .vector_store import VectorStore

openai.api_key = OPENAI_API_KEY
openai.log = "debug"

vector_store = VectorStore()

async def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve relevant chunks from vector store."""
    emb = await get_embedding(query)
    results = vector_store.search(emb, k=k)
    if not results:
        return ""
    context = "\n\n".join([chunk for chunk, _ in results])
    return context

async def run_agent(user_message: str, history: list) -> str:
    context = await retrieve_context(user_message)
    system_prompt = "You are a helpful learning assistant. Use the following context if relevant:\n" + context
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )
    logger.info(f"LLM response: {response}")
    return response.choices[0].message.content