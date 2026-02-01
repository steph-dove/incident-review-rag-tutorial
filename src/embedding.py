"""
Embeddings generation for RAG system.

This module handlers converting text to chunks into vector embeddings
that can be stored and searched in a vector database.
"""

from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-Mini-LM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        def embed_texts(
            self,
            text: Sequence[str],
            batch_size: int = 32,
            show_progress: bool = True
        ) -> np.ndarray:
            
            

