import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
import requests

load_dotenv()
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
AIPIPE_EMBEDDING_ENDPOINT = "https://aipipe.org/openai/v1/embeddings"

# Allow model selection via environment variable or parameter
def get_embedding_model(model_name: str = None):
    """
    Load a sentence-transformer embedding model. Default is BAAI/bge-base-en-v1.5 (best for retrieval as of 2025).
    Uses CUDA (GPU) if available for fast embedding.
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

# Cache the model for repeated use
_model_cache = {}
def get_aipipe_embedding(text: str, model_name: str = None):
    """
    Use a local embedding model (default: BGE) or text-embedding-ada-002 via AIPipe API if EMBEDDING_MODEL is 'text-embedding-ada-002'.
    Returns a list (JSON serializable) embedding.
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    if model_name == "text-embedding-ada-002":
        headers = {"Authorization": AIPIPE_API_KEY, "Content-Type": "application/json"}
        payload = {"input": text, "model": "text-embedding-ada-002"}
        response = requests.post(AIPIPE_EMBEDDING_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    if model_name not in _model_cache:
        _model_cache[model_name] = get_embedding_model(model_name)
    model = _model_cache[model_name]
    return model.encode(text).tolist()

def get_aipipe_embedding_batch(texts: list, model_name: str = None):
    """
    Batch embedding for a list of texts. Uses GPU if available, or AIPipe API if EMBEDDING_MODEL is 'text-embedding-ada-002'.
    Returns a list of embedding lists.
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    if model_name == "text-embedding-ada-002":
        # AIPipe does not support batch, so call one by one (watch your budget!)
        return [get_aipipe_embedding(text, model_name="text-embedding-ada-002") for text in texts]
    if model_name not in _model_cache:
        _model_cache[model_name] = get_embedding_model(model_name)
    model = _model_cache[model_name]
    return model.encode(texts, batch_size=64, show_progress_bar=False).tolist()
