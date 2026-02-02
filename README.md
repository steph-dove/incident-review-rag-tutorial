# Incident Review RAG Tutorial

A RAG (Retrieval-Augmented Generation) system for querying incident review documents, built with Metaflow and ChromaDB.

[Notion Doc Tutorial Walkthrough](https://www.notion.so/Tutorial-Building-a-RAG-System-for-Incident-Reviews-with-Metaflow-2f78c30dc8ad80c29885fbf374e43895#2f78c30dc8ad809f9992cdf2437dab1f)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify Metaflow
metaflow status
```

## Usage

### Index Documents

```bash
python flows/indexing_flow.py run
```

Options:
- `--data_dir`: Directory containing incident markdown files (default: `data/incidents`)
- `--db_path`: Path to ChromaDB database (default: `./chroma_db`)
- `--embedding_model`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--max_chunk_tokens`: Maximum tokens per chunk (default: `512`)

### Query Documents

```bash
python flows/query_flow.py run --query "What caused the outage?"
```
