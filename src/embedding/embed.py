import config
from functools import lru_cache
from llama_index.embeddings import HuggingFaceEmbedding


def embed(text: str) -> list[float]:
    model = initialize_embedding_model(config.EMBEDDING_MODEL)
    return model.get_text_embedding(text)


@lru_cache
def initialize_embedding_model(model: str) -> HuggingFaceEmbedding:
    embedding_model = HuggingFaceEmbedding(model_name=model)
    return embedding_model
