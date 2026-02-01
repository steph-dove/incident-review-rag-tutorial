"""
Vector database operations using ChromaDB.

This module provides a clean interface for storing and querying
incident review embeddings.
"""

import json
from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings


class IncidentVectorDB:
    """
    ChromaDB wrapper for incident review storage and retrieval

    Design decisions:
    - Persistent storage for production use
    - Metadata filtering for recise queries
    - Configurable similarity threshold
    """

    def __init__(
            self,
            persist_directory: str = './chroma_db',
            collection_name: str = "incident_reviews"
    ):
        """
        Initialize the vector database
        
        :param persist_directory: Where to store the database
        :param collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False, # Disable telemtry
                allow_reset=True # Allow database reset for testing
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_chunks(
            self,
            chunk_ids: list[str],
            texts: list[str],
            embeddings: list[list[float]],
            metadatas: list[dict]
    ) -> None:
        """
        Add document chunks to the vector database.
        
        :param chunk_ids: Unique ID for each chunk
        :param texts: Original text context
        :param embeddings: Vector embeddings
        :param metadatas: Metaddata dicts (must be JSON-serializable)
        """

        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for key, value in meta.items():
                if isinstance(value, (list, dict)):
                    clean_meta[key] = json.dumps(value)
                else:
                    clean_meta[key] = value
            clean_metadatas.append(clean_meta)

        self.collection.add(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas
        )

    def query(
            self,
            query_embedding: list[float],
            n_results: int=5,
            where: dict | None = None,
            where_document: dict | None = None
    ) -> dict:
        """
        Query the vector database for similar cunks
        
        :param query_embedding: Embedding of the query
        :param n_results: Number of results to return
        :param where: Metadata filter (e.g. {"severity": "P1"})
        :param where_document: Full-text filter on document content

        Returns Dict with ids, documents, metadatas, distances
        """

        results = self.collection.query(
            query_embedding=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        return results
    
    def get_all_chunks(self) -> dict:
        """Retrieve all chunks from the collection"""
        return self.collection.get(
            include=["documents", "metadatas"]
        )
    
    def delete_by_source(self, source_file: str) -> None:
        """Delete all chunks from a specific source file"""
        self.collection.delete(
            where={"source_file": source_file}
        )

    def reset(self) -> None:
        """Clear all data from the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return the number of chunks in the collection"""
        return self.collection.count()
    

def format_search_results(results: dict) -> list[dict]:
    """
    Format chromadb results into a cleaner structure
    
    Chromadb returns nested list. This flattens them for easier use.
    """
    formatted = []

    # ChromaDB returns a list of lists: one list per query
    # We only have one query, so take the first element
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadata", [[]])[0]
    distance = results.get("distance", [[]])[0]

    for i in range(len(ids)):
        # Convert cosine distances to similarity
        # ChromaDB returns L2 distance for cosine, so similarity = 1 - distance
        similarity = 1 - distance[i] if distance else None

        formatted.append({
            "id": ids[i],
            "content": documents[i],
            "metadata": metadatas[i],
            "similarity": similarity
        })

    return formatted
