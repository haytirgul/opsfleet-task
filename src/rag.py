from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))
from settings import (
    CHROMA_PERSIST_DIRECTORY, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL,
    GOOGLE_API_KEY
)

class HybridRetriever:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        
        # Initialize Vector Store (Dense)
        print(f"Loading ChromaDB from {CHROMA_PERSIST_DIRECTORY}...")
        try:
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
        except Exception as e:
            print(f"Error loading ChromaDB: {e}")
            print("Please run scripts/ingest.py first.")
            self.vector_store = None
            self.ensemble_retriever = None
            return

        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Initialize BM25 (Sparse)
        print("Building BM25 index from docstore...")
        # Get all documents from Chroma
        all_docs_data = self.vector_store.get()
        if all_docs_data and 'documents' in all_docs_data:
            self.docs = [
                Document(page_content=c, metadata=m) 
                for c, m in zip(all_docs_data['documents'], all_docs_data['metadatas'])
            ]
        else:
            self.docs = []
            print("Warning: Could not access docs for BM25.")

        if self.docs:
            self.bm25_retriever = BM25Retriever.from_documents(self.docs)
            self.bm25_retriever.k = 5
            
            # Ensemble
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[0.4, 0.6] # Slightly favor semantic search
            )
        else:
            self.ensemble_retriever = self.vector_retriever

    def retrieve(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve documents using the hybrid ensemble.
        """
        if not self.vector_store:
            return []

        # Update k
        self.vector_retriever.search_kwargs["k"] = k
        if hasattr(self, 'bm25_retriever'):
            self.bm25_retriever.k = k
        
        if filter:
            # Use vector search with filter for metadata filtering
            return self.vector_store.similarity_search(query, k=k, filter=filter)
            
        if self.ensemble_retriever:
            return self.ensemble_retriever.invoke(query)
        else:
            return self.vector_store.similarity_search(query, k=k)

    def get_concepts(self, query: str, k: int = 5) -> List[Document]:
        docs = self.retrieve(query, k=k*3)
        filtered = [d for d in docs if d.metadata.get("category") == "concepts"]
        return filtered[:k]

    def get_guides(self, query: str, k: int = 5) -> List[Document]:
        docs = self.retrieve(query, k=k*3)
        filtered = [d for d in docs if d.metadata.get("category") == "how-tos"]
        return filtered[:k]

    def get_api_ref(self, query: str, k: int = 5) -> List[Document]:
        docs = self.retrieve(query, k=k*3)
        filtered = [d for d in docs if d.metadata.get("category") == "reference"]
        return filtered[:k]
