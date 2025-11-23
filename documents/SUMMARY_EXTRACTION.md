# Document Summary Extraction

## Overview

This system implements a **Fuzzy → LLM Fallback Pattern** for extracting document titles and explanations from the LangGraph documentation corpus. It combines deterministic parsing with LLM-powered generation to ensure all documents get high-quality metadata.

## Architecture

### Two-Stage Approach

1. **Deterministic Extraction (Primary)**
   - Uses regex patterns to identify structured summaries
   - Fast, free, and consistent
   - Works when documents follow standard format

2. **LLM Generation (Fallback)**
   - Activates when deterministic extraction fails
   - Uses Gemini 1.5 Flash for cost-effective generation
   - Provides intelligent summaries for any document

## Document Patterns

### Pattern 1: Well-Structured Documents ✓

These documents are handled deterministically:

```markdown
# How to integrate LangGraph with AutoGen

This guide shows how to integrate AutoGen agents with LangGraph 
to leverage features like persistence, streaming, and memory...

## Prerequisites
...
```

**Characteristics:**
- Clear H1 heading (title)
- Followed by descriptive paragraph (explanation)
- Paragraph contains indicator phrases like "This guide", "This demonstrates", etc.
- Minimum 50 characters in explanation

### Pattern 2: Sparse Documents → LLM

These documents require LLM generation:

```markdown
# Installation

```bash
pip install langgraph
```

## Next Steps
...
```

**Characteristics:**
- H1 heading present
- No explanation paragraph OR very short
- Jumps directly to code/lists/subheadings
- Missing explanatory language

## Implementation

### Core Module: `src/document_summarizer.py`

```python
from src.document_summarizer import extract_or_generate_summary, ExtractionStats

# Process a document
doc = Document(page_content="...", metadata={"source": "..."})
title, explanation = extract_or_generate_summary(doc, use_llm_fallback=True)

# Check statistics
print(ExtractionStats.report())
```

### Integration: `scripts/ingest.py`

The ingestion pipeline automatically enriches all documents:

```python
def _create_doc(file_path: str, content: str) -> Document:
    """Create document with extracted/generated metadata."""
    doc = Document(
        page_content=content,
        metadata={
            "source": file_path,
            "category": category,
            "topic": topic
        }
    )
    
    # Hybrid extraction
    title, explanation = extract_or_generate_summary(doc, use_llm_fallback=True)
    
    doc.metadata["title"] = title
    doc.metadata["explanation"] = explanation
    
    return doc
```

## Metadata Schema

Each document gets enriched with:

| Field | Source | Description |
|-------|--------|-------------|
| `source` | File path | Original file path in corpus |
| `category` | File path | Top-level category (e.g., "how-tos", "concepts") |
| `topic` | File path | Specific topic from filename |
| `title` | Extracted/Generated | Document title (H1 or LLM-generated) |
| `explanation` | Extracted/Generated | 2-3 sentence summary of content |

## Benefits

### 1. **Better Retrieval Quality**
- Titles and explanations provide rich semantic context
- Improves vector search relevance
- Enables filtering by document type

### 2. **Cost Efficiency**
- ~80-90% of documents use deterministic extraction (free)
- Only ~10-20% require LLM calls
- Uses fast, cheap model (Gemini Flash)

### 3. **Consistency**
- All documents get metadata, regardless of structure
- Standardized format across corpus
- Fallback ensures no gaps

### 4. **Transparency**
- Statistics show extraction method distribution
- Easy to audit and improve patterns
- Clear separation of concerns

## Statistics Tracking

The system tracks which method was used:

```
Summary Extraction Stats:
  Deterministic: 45/50 (90.0%)
  LLM Generated: 5/50 (10.0%)
```

This helps:
- Monitor cost (LLM usage)
- Identify pattern improvements
- Validate extraction quality

## Configuration

### Deterministic Extraction Parameters

In `src/document_summarizer.py`:

```python
# Minimum explanation length
MIN_EXPLANATION_LENGTH = 50

# Explanation indicator phrases
explanation_indicators = [
    'this guide', 'this document', 'this tutorial',
    'shows how', 'demonstrates', 'explains', 
    'describes', 'walks through', 'provides',
    'introduces', 'covers', 'learn'
]
```

### LLM Configuration

Model selection in `_generate_summary_with_llm()`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Fast, cheap, good quality
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0  # Deterministic output
)
```

## Testing

Run the test suite:

```bash
cd opsfleet-task
python tests/test_summarizer.py
```

Tests cover:
1. Deterministic extraction for well-structured docs
2. LLM fallback for sparse docs
3. Mixed batch processing with statistics

## Usage Examples

### Basic Usage

```python
from langchain_core.documents import Document
from src.document_summarizer import extract_or_generate_summary

doc = Document(
    page_content="# My Guide\n\nThis guide explains...",
    metadata={"source": "guides/my-guide.md"}
)

title, explanation = extract_or_generate_summary(doc)
print(f"Title: {title}")
print(f"Explanation: {explanation}")
```

### Disable LLM Fallback

For testing or when you only want deterministic extraction:

```python
title, explanation = extract_or_generate_summary(
    doc, 
    use_llm_fallback=False  # Only use regex patterns
)
```

### Track Statistics

```python
from src.document_summarizer import ExtractionStats

ExtractionStats.reset()

# Process documents...
for doc in documents:
    extract_or_generate_summary(doc)

# Report
print(ExtractionStats.report())
```

## Future Improvements

### 1. **Caching**
- Cache LLM-generated summaries
- Avoid re-generating for same content
- Use content hash as cache key

### 2. **Pattern Learning**
- Analyze failed deterministic extractions
- Improve regex patterns over time
- A/B test pattern variations

### 3. **Multi-Model Support**
- Try different LLMs for fallback
- Compare quality vs cost
- Model selection based on document type

### 4. **Validation**
- Quality checks for generated summaries
- Length validation
- Coherence scoring

## Alignment with Project Rules

This implementation follows the project's core principles:

✓ **Agentic Pattern**: Fuzzy → LLM fallback (Rule 03-agentic.md)  
✓ **Structure**: Separate module with clear responsibilities (Rule 01-structure.md)  
✓ **Style**: Type hints, docstrings, `__all__` exports (Rule 02-style.md)  
✓ **Configuration**: Centralized settings, pathlib usage (Rule 04-config.md)  
✓ **Prompting**: Structured LLM prompt with clear format (Rule 05-prompting.md)

## Conclusion

This hybrid approach ensures:
- **100% coverage**: Every document gets metadata
- **Cost efficiency**: Minimize LLM usage
- **Quality**: Intelligent fallback when needed
- **Transparency**: Clear tracking and reporting

The system is production-ready and scales to any documentation corpus size.

