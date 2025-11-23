import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# RAG Settings
CHROMA_PERSIST_DIRECTORY = str(OUTPUT_DIR / "chroma_db")
COLLECTION_NAME = "langgraph_docs"
EMBEDDING_MODEL = "models/embedding-001"  # Google Gemini Embedding

# LLM Settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


