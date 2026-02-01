"""
Metaflow pipelines for indexing icnident review docuemnts

This flow
1. Discovers incidenet review documents
2. Chunks documents intelligently
3. Generates embeddings
4. Stores in vector database
"""

from pathlib import Path
import json
from metaflow import FlowSpec, step, Parameter, current

class IncidentIndexingFlow(FlowSpec):
    """
    Index incident review documents for RAG practices:
    - Parameters for configuration
    - Artifacts for intermediate data
    - Step-by-step processing with checkpoints
    """

    data_dir = Parameter(
        'data_dir',
        help='Directory containing incident review merkdown files',
        default='data/incidents'
    )

    db_path = Parameter(
        'db_path',
        help='Path to ChromaDB database',
        default='./chroma_db'
    )

    embedding_model = Parameter(
        'db_path',
        help='Sentence transformer model to use',
        default='all-MiniLM-L6-v2'
    )

    max_chunk_tokens = Parameter(
        'max_chunk_tokens',
        help='Maximum tokens per chunk',
        default=512,
        type=int
    )

    @step
    def start(self):
        """
        Initialize the pipeline and discover documents.
        
        This step:
        - Validates input directory exists
        - Finds all markdown files
        - Stores document paths as artifacts for next step
        """
        import os

        print(f"Starting indexing flow")
        print(f"Data directory: {self.data_dir}")
        print(f"Database path: {self.db_path}")

        # Find all markdown files
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        self.document_paths = list(data_path.glob('*.md'))
        print(f"Found {len(self.document_paths)} documents to index")

        # store paths as strings (Path objects don't serialize well)
        self.document_paths = [str(p) for p in self.document_paths]

        self.next(self.chunk_documents)


    @step
    def chunk_documents(self):
        """
        Read and chunk all documents

        This step:
        - Reads each document
        - Applies chunking strategy
        - Stores chunks as artifacts

        Why a separate step?
        - Chunking is independent of embedding
        - If embedding fails, we don't re-chunk
        - We can inspect chunks before embedding
        """

        from src.chunking import IncidentChunker

        chunker = IncidentChunker(max_token=self.max_chunk_tokens)

        all_chunks = []
        for doc_path in self.document_paths:
            print(f'chunking: {doc_path}')

            with open(doc_path) as f:
                content = f.read()

            chunks = chunker.chunk_document(content, doc_path)
            all_chunks.extend(chunks)

            print(f'Created {len(chunks)} chunks')

        self.chunks = [{
            'content': c.content,
            'metadata': c.metadata,
            'chunk_index': c.chunk_index,
            'token_count': c.token_count,
            'section': c.section,
        } for c in all_chunks]

        print(f'\nTotal chunks created: {len(self.chunks)}')

        self.next(self.generate_embeddings)

    @step
    def generate_embeddings(self):
        """
        Generate embeddings for all chunks

        This step:
        - Loads the embeddin model
        - Processes chunks in batches
        - Stores embeddings as artifacts

        Metaflow artifact storage means embeddings are versioned
        and can be retrieved for debugging or comparison
        """
        from src.embedding import EmbeddingGenerator

        print(f"Loading embedding model: {self.embedding_model}")
        embedder = EmbeddingGenerator(self.embedding_model)

        # Extract text content for embedding
        texts = [chunk['content'] for chunk in self.chunks]

        print(f'Generating embeddings for {len(texts)} chunks...')
        embeddings = embedder.embed_texts(texts, batch_size=32)

        # Store as list of lists (numpy arrays don't serialize well in some cases)
        self.embeddings = embeddings.tolist()
        self.embedding_dim = embedder.embedding_dim

        print(f'Generated {len(self.embeddings)} embeddings of dimension {self.embedding_dim}')
        self.next(self.store_in_verctordb)

    @step
    def store_in_vectordb(self):
        """
        Store chunks and embeddings in ChromaDB

        This step:
        - Initializes ChromaDB connection
        - Generates unique IDs for chunks
        - Inserts all data into vector database
        """
        from src.vectordb import IncidentVectorDB

        print(f'Initializing vector database at: {self.db_path}')
        db = IncidentVectorDB(persist_directory=self.db_path)

        # Generate unique chunk IDs
        # Format: source_file_chunkN
        chunk_ids = []
        for chunk in self.chunks:
            source = Path(chunk['metadata']['source_file']).stem
            idx = chunk['chunk_index']
            chunk_ids.append(f'{source}_chunk{idx}')

        # Extract text and metadata
        texts = [chunk['content'] for chunk in self.chunks]
        metadatas = [chunk['metadata'] for chunk in self.chunks]

        # Add section info to metadata
        for i, chunk in enumerate(self.chunks):
            metadatas[i]['section'] = chunk['section']
            metadatas[i]['chunk_index'] = chunk['chunk_index']

        print(f'Storing {len(chunk_ids)} chunks in vector database...')
        db.add_chunks(
            chunk_ids=chunk_ids,
            texts=texts,
            embeddings=self.embeddings,
            metadatas=metadatas
        )

        self.total_indexed = db.count()
        print(f'Total chunks in database: {self.total_indexed}')

        self.next(self.end)


    @step
    def end(self):
        """
        Finalize the pipeline and report statistics
        """

        print('\n' + "="*50)
        print('INDEXING COMPLETE')
        print("="*50)
        print(f'Documents processed: {len(self.document_paths)}')
        print(f'Chunks created: {len(self.chunks)}')
        print(f'Chunks indexed: {self.total_indexed}')
        print(f'Embedding dimensions: {self.embedding_dim}')
        print(f'\nRun ID: {current.run_id}')
        print(f'Database location: {self.db_path}')


if __name__ == '__main__':
    IncidentIndexingFlow()