"""
Embedding generation for RAG systems

This module handles converting text chunks into vector embeddings
that can be stored and searched in a vector database.
"""

from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers

    Why sentence-transformers?
    - Open source, no API costs
    - Runs locally (data privacy)
    - High quality embeddings
    - Easy to switch models
    """

    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()


    def embed_texts(
            self,
            texts: Sequence[str],
            batch_size: int = 32,
            show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        :param texts: Liist of text strings to embed
        :param batch_size: Number of texts to process at once
         - Larger = faster but more memory
         - 32 is good for most systems
        :param show_progress: Wether to show progress bar
        :return: numpy array of shape (len(texts), embedding_dim)
        """

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )

        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string
        
        Note: we use the same embedding for queries and documents.
        Some models have separate query/document encoders, but
        Symmatric models like MiniLM work well with the same encoder.
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding
    
    def compute_similarity(
            self,
            query_embedding: np.ndarray,
            document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Since embeddings are normalized, cosine similarity = dot product.
        
        :param query_embedding: Shape (embedding_dim)
        :param document_embeddings: Shape (n_docs, embedding_dim)

        returns similarity scores of shape (n_docs)
        """

        similarities = np.dot(document_embeddings, query_embedding)
        return similarities


# Utility function for quick embedding generation
def create_embedder(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingGenerator:
    """Factory function for creating embedding generators"""
    return EmbeddingGenerator(model_name)
