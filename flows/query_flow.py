"""
Metaflow pipeline for querying indexed incident reviews.

This flow:
1. Takes a natural language query
2. Embeds the query
3. Retrieves relevant chunks from vector DB
4. Generates an answer using retrieved context

Run with: python flows/query_flow.py run --query "What caused the payment outage?"
"""

import sys
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metaflow import FlowSpec, step, Parameter, current


class IncidentQueryFlow(FlowSpec):
    """
    Query the incident review RAG system.

    This flow demonstrates:
    - Query embedding
    - Vector similarity search
    - Context-augmented generation
    """

    query = Parameter(
        'query',
        help='Natural language question to answer',
        required=True
    )

    db_path = Parameter(
        'db_path',
        help='Path to ChromaDB database',
        default='./chroma_db'
    )

    embedding_model = Parameter(
        'embedding_model',
        help='Sentence transformer model (must match indexing)',
        default='all-MiniLM-L6-v2'
    )

    top_k = Parameter(
        'top_k',
        help='Number of chunks to retrieve',
        default=5,
        type=int
    )

    severity_filter = Parameter(
        'severity_filter',
        help='Filter by severity (e.g., P1, P2)',
        default=None
    )

    @step
    def start(self):
        """
        Initialize and validate parameters.
        """
        print(f"Query: {self.query}")
        print(f"Database: {self.db_path}")
        print(f"Retrieving top {self.top_k} chunks")

        if self.severity_filter:
            print(f"Filtering by severity: {self.severity_filter}")

        self.next(self.embed_query)

    @step
    def embed_query(self):
        """
        Generate embedding for the query.

        The same embedding model must be used for queries
        and documents for similarity search to work correctly.
        """
        from src.embedding import EmbeddingGenerator

        print(f"Loading embedding model: {self.embedding_model}")
        embedder = EmbeddingGenerator(self.embedding_model)

        print("Embedding query...")
        self.query_embedding = embedder.embed_query(self.query).tolist()

        self.next(self.retrieve_chunks)

    @step
    def retrieve_chunks(self):
        """
        Search vector database for relevant chunks.

        This is where the magic happens:
        - Query embedding is compared to all chunk embeddings
        - Most similar chunks are returned
        - Optional metadata filters narrow results
        """
        from src.vectordb import IncidentVectorDB, format_search_results

        print(f"Searching vector database...")
        db = IncidentVectorDB(persist_directory=self.db_path)

        # Build metadata filter if specified
        where_filter = None
        if self.severity_filter:
            where_filter = {"severity": self.severity_filter}

        # Perform similarity search
        results = db.query(
            query_embedding=self.query_embedding,
            n_results=self.top_k,
            where=where_filter
        )

        # Format results for easier handling
        self.retrieved_chunks = format_search_results(results)

        print(f"Retrieved {len(self.retrieved_chunks)} chunks")
        for i, chunk in enumerate(self.retrieved_chunks):
            print(f"\n  [{i+1}] Similarity: {chunk['similarity']:.3f}")
            print(f"      Source: {chunk['metadata'].get('source_file', 'unknown')}")
            print(f"      Section: {chunk['metadata'].get('section', 'unknown')}")

        self.next(self.generate_answer)

    @step
    def generate_answer(self):
        """
        Generate an answer using retrieved context.

        This step builds a prompt with:
        1. System instruction for the task
        2. Retrieved context from the vector search
        3. The user's original question

        Note: This uses a simple approach without an LLM API.
        In production, you would call OpenAI, Anthropic, or a local model.
        """

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(self.retrieved_chunks):
            source = chunk['metadata'].get('source_file', 'unknown')
            section = chunk['metadata'].get('section', 'unknown')
            context_parts.append(
                f"[Source: {source}, Section: {section}]\n{chunk['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Build the augmented prompt
        self.prompt = f"""You are an expert at analyzing incident reviews.
Use the following context from incident review documents to answer the question.
If the answer is not in the context, say so.

CONTEXT:
{context}

QUESTION: {self.query}

ANSWER:"""

        # For demonstration, we'll just show the prompt
        # In production, you would call an LLM here:
        #
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": self.prompt}]
        # )
        # self.answer = response.choices[0].message.content

        self.answer = "[LLM response would go here - see prompt below]"

        self.next(self.end)

    @step
    def end(self):
        """
        Display the final answer and debugging information.
        """
        print("\n" + "="*60)
        print("QUERY RESULTS")
        print("="*60)

        print(f"\nQuestion: {self.query}")
        print(f"\nRetrieved {len(self.retrieved_chunks)} relevant chunks")

        print("\n" + "-"*60)
        print("RETRIEVED CONTEXT:")
        print("-"*60)
        for i, chunk in enumerate(self.retrieved_chunks):
            print(f"\n[{i+1}] {chunk['metadata'].get('source_file', 'unknown')}")
            print(f"    Section: {chunk['metadata'].get('section', 'unknown')}")
            print(f"    Similarity: {chunk['similarity']:.3f}")
            print(f"    Content preview: {chunk['content'][:200]}...")

        print("\n" + "-"*60)
        print("GENERATED PROMPT (for LLM):")
        print("-"*60)
        print(self.prompt[:1500] + "..." if len(self.prompt) > 1500 else self.prompt)

        print(f"\nRun ID: {current.run_id}")


if __name__ == '__main__':
    IncidentQueryFlow()
