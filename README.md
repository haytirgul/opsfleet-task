# LangGraph Helper Agent

An AI-powered LangGraph Helper Agent that assists developers working with LangGraph and LangChain by answering practical questions. Supports both offline and online modes for different usage scenarios.

## Features

- **Offline Mode**: Works without internet connectivity using locally stored documentation
- **Online Mode**: Allows real-time information retrieval via web searches and APIs
- **Hybrid RAG System**: Combines semantic search (vector embeddings) with keyword search (BM25)
- **Document Summarization**: Intelligent title and explanation extraction using Fuzzy → LLM fallback pattern
- **Structured Output**: Returns well-formatted answers with source citations

## Architecture

### Core Components

- **Document Ingestion** (`scripts/ingest.py`): Parses llms.txt format documentation and extracts metadata
- **Hybrid Retriever** (`src/rag.py`): Combines vector search (ChromaDB) with BM25 keyword search
- **Document Summarizer** (`src/document_summarizer.py`): Extracts titles/explanations deterministically or via LLM fallback

### Data Flow

1. **Ingestion**: Parse documentation → Extract summaries → Chunk documents → Store in vector DB
2. **Retrieval**: Query → Hybrid search (BM25 + Vector) → Rerank → Return top results
3. **Generation**: Context + Query → LLM → Structured answer

## Setup

### Prerequisites

- Python 3.11+
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/haytirgul/opsfleet-task.git
cd opsfleet-task
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

4. Download documentation data:
```bash
python scripts/download_data.py
```

5. Ingest documents:
```bash
python scripts/ingest.py
```

## Usage

### Offline Mode

```bash
export AGENT_MODE=offline
python main.py "How do I add persistence to a LangGraph agent?"
```

### Online Mode

```bash
export AGENT_MODE=online
python main.py "What are the latest LangGraph features?"
```

Or use command-line flag:
```bash
python main.py --mode offline "How do I use checkpointers?"
```

## Project Structure

```
opsfleet-task/
├── src/                    # Core source code
│   ├── rag.py             # Hybrid retriever implementation
│   └── document_summarizer.py  # Summary extraction
├── scripts/                # Utility scripts
│   ├── ingest.py          # Document ingestion pipeline
│   ├── download_data.py   # Download documentation
│   └── verify_setup.py    # Setup verification
├── data/                   # Data directory
│   ├── input/             # Raw documentation files
│   └── output/            # Processed data and vector store
├── settings.py            # Configuration settings
└── requirements.txt       # Python dependencies
```

## Data Preparation Strategy

### Offline Mode Data Sources

- **LangGraph Python**: Downloaded from https://langchain-ai.github.io/langgraph/llms-full.txt
- **Format**: llms.txt format with multiple virtual files
- **Update Strategy**: 
  - Manual: Re-download llms-full.txt and re-run `scripts/ingest.py`
  - Automated: Can be scheduled via cron or GitHub Actions

### Data Processing

1. **Parsing**: Extracts individual documents from llms.txt format
2. **Summarization**: Extracts titles/explanations (deterministic + LLM fallback)
3. **Chunking**: Splits documents by Markdown headers, then by characters
4. **Embedding**: Generates embeddings using Google Gemini Embeddings
5. **Storage**: Stores in ChromaDB vector database

## Technical Stack

- **Language**: Python 3.11+
- **LLM**: Google Gemini (free tier)
- **Vector DB**: ChromaDB
- **Embeddings**: Google Gemini Embeddings (`models/embedding-001`)
- **Search**: Hybrid (BM25 + Semantic)
- **Framework**: LangChain, LangGraph

## Development

### Code Style

Follows PEP 8, PEP 257, and project-specific rules:
- Type hints required
- Google-style docstrings
- Pydantic models for data schemas
- Pathlib for file operations

### Testing

```bash
pytest tests/
```

### Linting

```bash
ruff check .
black --check .
mypy src/
```

## License

MIT License

## Author

haytirgul

## Acknowledgments

- LangGraph documentation team
- LangChain community

