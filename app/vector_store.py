import faiss
import numpy as np
import pickle
from typing import List, Tuple
from .config import FAISS_INDEX_PATH, logger

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.load()
    
    def add_embeddings(self, embeddings: List[np.ndarray], chunks: List[str]):
        if not embeddings:
            return
        emb_matrix = np.vstack(embeddings).astype(np.float32)
        self.index.add(emb_matrix)
        self.chunks.extend(chunks)
        self.save()
    
    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query_emb.reshape(1, -1), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(distances[0][i])))
        return results
    
    def save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open("chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("Vector store saved.")
    
    def load(self):
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open("chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            logger.info("Vector store loaded.")
        except:
            logger.info("No existing vector store found, starting fresh.")