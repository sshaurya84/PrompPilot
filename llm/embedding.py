from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.constants import EMBEDDING_MODEL

def load_embedding_model():
    return HuggingFaceEmbedding(EMBEDDING_MODEL)