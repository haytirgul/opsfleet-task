"""
Test the document summarization with Fuzzy -> LLM fallback pattern.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from src.document_summarizer import extract_or_generate_summary, ExtractionStats


def test_deterministic_extraction():
    """Test extraction for documents with clear structure."""
    print("\n" + "="*70)
    print("TEST 1: Document WITH clear summary (deterministic extraction)")
    print("="*70)
    
    content = """# How to use the graph API

This guide demonstrates the basics of LangGraph's Graph API. It walks through state management, as well as composing common graph structures such as sequences, branches, and loops.

## Setup

Install langgraph:

```bash
pip install -U langgraph
```
"""
    
    doc = Document(
        page_content=content,
        metadata={"source": "how-tos/graph-api.md"}
    )
    
    ExtractionStats.reset()
    title, explanation = extract_or_generate_summary(doc, use_llm_fallback=False)
    
    print(f"\nTitle: {title}")
    print(f"\nExplanation: {explanation}")
    print(f"\nMethod: {'Deterministic' if ExtractionStats.deterministic_count > 0 else 'Other'}")
    
    assert "Graph API" in title
    assert len(explanation) > 50
    print("\n✓ Test passed!")


def test_llm_fallback():
    """Test LLM fallback for documents without clear structure."""
    print("\n" + "="*70)
    print("TEST 2: Document WITHOUT clear summary (LLM fallback)")
    print("="*70)
    
    # Document with weak/missing explanation
    content = """# Installation

```bash
pip install langgraph
```

## Configuration

Set your API key:

```python
export LANGSMITH_API_KEY=your-key
```
"""
    
    doc = Document(
        page_content=content,
        metadata={"source": "guides/installation.md"}
    )
    
    ExtractionStats.reset()
    
    # First try without LLM (should use basic fallback)
    title_no_llm, explanation_no_llm = extract_or_generate_summary(
        doc, 
        use_llm_fallback=False,
        track_stats=False
    )
    
    print(f"\nWithout LLM fallback:")
    print(f"  Title: {title_no_llm}")
    print(f"  Explanation: {explanation_no_llm}")
    
    # Now with LLM fallback
    title_with_llm, explanation_with_llm = extract_or_generate_summary(
        doc, 
        use_llm_fallback=True
    )
    
    print(f"\nWith LLM fallback:")
    print(f"  Title: {title_with_llm}")
    print(f"  Explanation: {explanation_with_llm}")
    print(f"\nMethod: {'LLM Generated' if ExtractionStats.llm_count > 0 else 'Other'}")
    
    assert len(title_with_llm) > 0
    assert len(explanation_with_llm) > 20  # Should be more descriptive
    print("\n✓ Test passed!")


def test_mixed_batch():
    """Test processing a batch with mixed structure."""
    print("\n" + "="*70)
    print("TEST 3: Mixed batch of documents")
    print("="*70)
    
    docs = [
        Document(
            page_content="""# How to integrate LangGraph with AutoGen

This guide shows how to integrate AutoGen agents with LangGraph to leverage features like persistence, streaming, and memory.""",
            metadata={"source": "how-tos/autogen.md"}
        ),
        Document(
            page_content="""# Quick Reference

- Command 1: Description
- Command 2: Description""",
            metadata={"source": "reference/commands.md"}
        ),
        Document(
            page_content="""# Streaming in LangGraph

LangGraph provides multiple streaming modes to get real-time outputs from your graph execution. This enables responsive applications and better debugging visibility.

## Stream Modes""",
            metadata={"source": "concepts/streaming.md"}
        ),
    ]
    
    ExtractionStats.reset()
    
    results = []
    for doc in docs:
        title, explanation = extract_or_generate_summary(doc, use_llm_fallback=True)
        results.append((doc.metadata['source'], title, explanation[:80] + "..."))
    
    print(f"\nProcessed {len(docs)} documents:\n")
    for source, title, explanation in results:
        print(f"Source: {source}")
        print(f"  Title: {title}")
        print(f"  Explanation: {explanation}\n")
    
    print(ExtractionStats.report())
    
    assert ExtractionStats.deterministic_count + ExtractionStats.llm_count == len(docs)
    print("\n✓ Test passed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DOCUMENT SUMMARIZER TESTS")
    print("Testing Fuzzy -> LLM Fallback Pattern")
    print("="*70)
    
    try:
        test_deterministic_extraction()
        test_llm_fallback()
        test_mixed_batch()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

