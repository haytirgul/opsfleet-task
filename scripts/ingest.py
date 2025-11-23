import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Load settings
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import (
    DATA_DIR, 
    OUTPUT_DIR,
)
from src.document_summarizer import extract_or_generate_summary, ExtractionStats



# 1. Parser Logic
def _strip_search_metadata(content: str) -> str:
    """
    Strip 'search:' metadata blocks from content.
    
    Removes patterns like:
    search:
      boost: 2
    ---
    """
    lines = content.splitlines()
    cleaned_lines = []
    skip_until_separator = False
    
    for i, line in enumerate(lines):
        # Detect start of search metadata
        if line.strip() == "search:":
            skip_until_separator = True
            continue
        
        # Skip lines until we hit a separator
        if skip_until_separator:
            if line.strip() == "---":
                skip_until_separator = False
                # Don't include the separator itself if it's part of metadata
                continue
            else:
                continue
        
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()


def parse_llms_txt(file_path: Path) -> List[Document]:
    """
    Parses a llms.txt file with multiple virtual files separated by '---'.
    Extracts metadata like file_path, category, topic, title, and explanation.
    
    Handles two formats:
    1. Standard: ---\nfilepath\n---\ncontent
    2. Search metadata: search:\n  boost: 2\n---\ncontent (no filepath)
    
    Uses hybrid approach for title/explanation extraction:
    - Deterministic extraction via regex patterns
    - LLM generation as fallback
    """
    if not file_path.exists():
        print(f"Error: Source file {file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Reset extraction statistics
    ExtractionStats.reset()
    
    # Parse documents
    lines = content.splitlines()
    documents: List[Document] = []
    
    current_file_path = ""
    current_content_lines: List[str] = []
    is_reading_filepath = False
    in_search_metadata = False
    
    print("Parsing documents...")
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.strip() == "---":
            # Check if this is a file separator pattern: --- \n filepath \n ---
            if i + 2 < len(lines) and lines[i+2].strip() == "---":
                # Save previous doc if exists
                if current_content_lines:
                    doc_content = _strip_search_metadata("\n".join(current_content_lines))
                    if doc_content:
                        documents.append(_create_doc(current_file_path, doc_content))
                    current_content_lines = []
                
                # Extract filepath
                current_file_path = lines[i+1].strip()
                is_reading_filepath = True
                i += 3  # Skip: ---, filepath, ---
                continue
            
            # Check if this is a search metadata separator: search:\n  boost: 2\n---
            elif i >= 2 and lines[i-2].strip() == "search:":
                # Save previous doc if exists
                if current_content_lines:
                    doc_content = _strip_search_metadata("\n".join(current_content_lines))
                    if doc_content:
                        documents.append(_create_doc(current_file_path, doc_content))
                    current_content_lines = []
                
                # No filepath for search metadata documents
                current_file_path = ""
                is_reading_filepath = False
                in_search_metadata = False
                i += 1  # Skip the --- separator
                continue
            
            elif is_reading_filepath:
                # End of filepath block
                is_reading_filepath = False
                i += 1
                continue
        
        elif line.strip() == "search:":
            # Start of search metadata block
            in_search_metadata = True
            # Save previous doc if exists
            if current_content_lines:
                doc_content = _strip_search_metadata("\n".join(current_content_lines))
                if doc_content:
                    documents.append(_create_doc(current_file_path, doc_content))
                current_content_lines = []
            current_file_path = ""
            i += 1
            continue
        
        elif in_search_metadata:
            # Skip search metadata lines until we hit ---
            if line.strip() == "---":
                in_search_metadata = False
            i += 1
            continue
        
        elif is_reading_filepath:
            # Skip filepath line (already handled)
            i += 1
            continue
        
        else:
            # Regular content line
            current_content_lines.append(line)
            i += 1
    
    # Add last doc
    if current_content_lines:
        doc_content = _strip_search_metadata("\n".join(current_content_lines))
        if doc_content:
            documents.append(_create_doc(current_file_path, doc_content))
    
    print(f"\nParsed {len(documents)} documents.")
    print(ExtractionStats.report())
    
    return documents

def _extract_title_from_content(content: str) -> str:
    """Extract title from content by finding the first H1 heading."""
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("# ") and len(line) > 2:
            return line[2:].strip()
    return ""


def _create_doc(file_path: str, content: str) -> Document:
    """
    Helper to create a Document with metadata derived from file path and content.
    
    Extracts title and explanation using hybrid approach:
    1. Deterministic extraction (regex patterns)
    2. LLM generation fallback
    
    If file_path is empty, extracts it from the document title.
    """
    # Extract title first to use as fallback for file_path
    title = _extract_title_from_content(content)
    
    # If no filepath, derive it from title
    if not file_path and title:
        # Convert title to filepath-like format
        file_path = title.lower().replace(" ", "-").replace(":", "").replace("'", "")
        # Add .md extension if not present
        if not file_path.endswith((".md", ".ipynb")):
            file_path += ".md"
    
    # If still no filepath, use default
    if not file_path:
        file_path = "untitled.md"
    
    # Extract category and topic from filepath
    parts = file_path.split('/')
    if len(parts) > 1:
        category = parts[0]
        topic = parts[1].replace(".md", "").replace(".ipynb", "")
    else:
        # Try to infer category from content or use general
        category = "general"
        topic = file_path.replace(".md", "").replace(".ipynb", "")
    
    # Create initial document
    doc = Document(
        page_content=content,
        metadata={
            "source": file_path if file_path != "untitled.md" else "",
            "category": category,
            "topic": topic
        }
    )
    
    # Extract or generate title and explanation
    # Temporarily disable LLM fallback due to model name issue - will use deterministic only
    try:
        extracted_title, explanation = extract_or_generate_summary(doc, use_llm_fallback=False)
        # Use extracted title if we got one, otherwise use the one we found
        if extracted_title and extracted_title != "Untitled Document":
            title = extracted_title
        elif not title:
            title = extracted_title if extracted_title else "Untitled"
    except Exception:
        # Fallback to basic extraction if summary extraction fails
        if not title:
            title = file_path.split('/')[-1].replace('.md', '').replace('.ipynb', '').replace('-', ' ').title()
        explanation = content[:200] + "..." if len(content) > 200 else content
    
    # Ensure we have a title
    if not title or title.strip() == "":
        title = "Untitled Document"
    
    # Add to metadata
    doc.metadata["title"] = title
    doc.metadata["explanation"] = explanation
    
    return doc

# 2. Chunking Logic
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents first by Markdown headers, then by characters.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    final_docs = []
    
    print(f"Splitting {len(documents)} source documents...")
    
    for doc in tqdm(documents):
        # 1. Split by Headers
        md_docs = markdown_splitter.split_text(doc.page_content)
        
        for md_doc in md_docs:
            # Merge original metadata
            md_doc.metadata.update(doc.metadata)
            
            # 2. Split by Character
            split_docs = text_splitter.split_documents([md_doc])
            final_docs.extend(split_docs)
            
    print(f"Created {len(final_docs)} total chunks.")
    return final_docs

# 3. JSON Export Functions
def document_to_dict(doc: Document) -> Dict[str, Any]:
    """Convert a Document to a dictionary for JSON serialization."""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }


def save_documents_as_json(
    documents: List[Document], 
    output_dir: Path,
    save_individual: bool = True,
    save_combined: bool = True
) -> None:
    """
    Save documents as JSON files.
    
    Args:
        documents: List of Document objects to save
        output_dir: Directory to save JSON files
        save_individual: If True, save each document as a separate JSON file
        save_combined: If True, save all documents in a single JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert documents to dictionaries
    doc_dicts = [document_to_dict(doc) for doc in documents]
    
    if save_individual:
        print(f"\nSaving {len(documents)} individual JSON files...")
        json_dir = output_dir / "documents_json"
        json_dir.mkdir(parents=True, exist_ok=True)
        
        for i, doc_dict in enumerate(tqdm(doc_dicts, desc="Saving individual files")):
            # Create safe filename from source path
            source = doc_dict["metadata"].get("source", f"document_{i}")
            # Replace path separators and remove extension
            safe_filename = source.replace("/", "_").replace("\\", "_").replace(".md", "")
            if not safe_filename:
                safe_filename = f"document_{i}"
            
            # Add category subdirectory for organization
            category = doc_dict["metadata"].get("category", "general")
            category_dir = json_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual file
            json_file = category_dir / f"{safe_filename}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Saved individual files to: {json_dir}")
    
    if save_combined:
        print(f"\nSaving combined JSON file...")
        combined_file = output_dir / "documents_combined.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(doc_dicts, f, indent=2, ensure_ascii=False)
        print(f"Saved combined file to: {combined_file}")
        
        # Also save a summary index
        summary_data = [
            {
                "index": i,
                "source": doc["metadata"].get("source", "unknown"),
                "title": doc["metadata"].get("title", "Untitled"),
                "category": doc["metadata"].get("category", "general"),
                "topic": doc["metadata"].get("topic", "unknown"),
                "explanation": doc["metadata"].get("explanation", "")[:200] + "..." if len(doc["metadata"].get("explanation", "")) > 200 else doc["metadata"].get("explanation", ""),
                "content_length": len(doc["page_content"])
            }
            for i, doc in enumerate(doc_dicts)
        ]
        
        summary_file = output_dir / "documents_index.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"Saved index file to: {summary_file}")


# 4. Main Ingestion Function
def ingest_data(save_chunks: bool = False):
    """
    Parse documents and save as JSON files.
    
    Args:
        save_chunks: If True, also save chunked versions of documents
    """
    # Path to your data file (now in input subdirectory)
    source_file = DATA_DIR / "input" / "langgraph_llms_full.txt"
    
    print(f"Parsing source file: {source_file}")
    print("="*70)
    raw_docs = parse_llms_txt(source_file)
    
    if not raw_docs:
        print("No documents parsed. Exiting.")
        return

    # Save raw documents
    print("\n" + "="*70)
    print("Saving documents as JSON...")
    print("="*70)
    save_documents_as_json(
        raw_docs,
        output_dir=OUTPUT_DIR,
        save_individual=True,
        save_combined=True
    )
    
    if save_chunks:
        print("\n" + "="*70)
        print("Chunking documents...")
        print("="*70)
        chunks = chunk_documents(raw_docs)
        
        print("\nSaving chunks as JSON...")
        chunks_dir = OUTPUT_DIR / "chunks_json"
        save_documents_as_json(
            chunks,
            output_dir=chunks_dir,
            save_individual=False,  # Too many files for chunks
            save_combined=True
        )
        print(f"Saved {len(chunks)} chunks to: {chunks_dir}")
    
    print("\n" + "="*70)
    print("Ingestion complete!")
    print("="*70)
    print(f"\nDocuments saved to: {OUTPUT_DIR}")
    print(f"  - Individual files: {OUTPUT_DIR / 'documents_json'}")
    print(f"  - Combined file: {OUTPUT_DIR / 'documents_combined.json'}")
    print(f"  - Index file: {OUTPUT_DIR / 'documents_index.json'}")

if __name__ == "__main__":
    ingest_data()
