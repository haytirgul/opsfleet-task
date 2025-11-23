"""
Document summarization with Fuzzy -> LLM fallback pattern.

This module extracts document summaries deterministically where possible,
and uses LLM generation as a fallback for documents without clear structure.
"""

import re
from typing import Optional, Tuple
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import GOOGLE_API_KEY


__all__ = ["extract_or_generate_summary", "ExtractionStats"]


class ExtractionStats:
    """Track statistics for summary extraction."""
    deterministic_count: int = 0
    llm_count: int = 0
    
    @classmethod
    def reset(cls):
        cls.deterministic_count = 0
        cls.llm_count = 0
    
    @classmethod
    def report(cls) -> str:
        total = cls.deterministic_count + cls.llm_count
        if total == 0:
            return "No documents processed."
        
        det_pct = (cls.deterministic_count / total) * 100
        llm_pct = (cls.llm_count / total) * 100
        
        return (
            f"Summary Extraction Stats:\n"
            f"  Deterministic: {cls.deterministic_count}/{total} ({det_pct:.1f}%)\n"
            f"  LLM Generated: {cls.llm_count}/{total} ({llm_pct:.1f}%)"
        )


# Deterministic pattern for summary extraction
SUMMARY_PATTERN = re.compile(
    r'^#\s+(.+?)\n\n(.+?)(?=\n\n#{1,3}\s|\n\n```|\n\n-|\n\n\*|\n\n\d+\.|\Z)',
    re.DOTALL | re.MULTILINE
)

# Minimum length for a valid explanation (in characters)
MIN_EXPLANATION_LENGTH = 50


def _extract_summary_deterministic(content: str) -> Optional[Tuple[str, str]]:
    """
    Deterministically extract title and explanation from document content.
    
    Args:
        content: Raw document content
        
    Returns:
        Tuple of (title, explanation) if found, None otherwise
    """
    # Try to match the pattern: H1 followed by explanation paragraph
    match = SUMMARY_PATTERN.search(content)
    
    if not match:
        return None
    
    title = match.group(1).strip()
    explanation = match.group(2).strip()
    
    # Validate explanation length and content
    if len(explanation) < MIN_EXPLANATION_LENGTH:
        return None
    
    # Check if explanation looks like actual prose (not just a code snippet or list)
    # Heuristic: should contain common explanation words
    explanation_indicators = [
        'this guide', 'this document', 'this tutorial', 'shows how',
        'demonstrates', 'explains', 'describes', 'walks through',
        'provides', 'introduces', 'covers', 'learn'
    ]
    
    explanation_lower = explanation.lower()
    has_explanation_language = any(
        indicator in explanation_lower 
        for indicator in explanation_indicators
    )
    
    # Also accept if it's a reasonably long paragraph (200+ chars)
    # even without explicit explanation language
    if has_explanation_language or len(explanation) >= 200:
        return (title, explanation)
    
    return None


def _generate_summary_with_llm(content: str, file_path: str) -> Tuple[str, str]:
    """
    Generate title and explanation using LLM.
    
    Args:
        content: Document content to summarize
        file_path: File path for context
        
    Returns:
        Tuple of (title, explanation)
    """
    # Try different model names - common ones: "gemini-pro", "gemini-1.5-pro", "models/gemini-pro"
    # If one fails, we'll catch and use fallback
    model_names = ["gemini-pro", "models/gemini-pro", "gemini-1.5-pro"]
    llm = None
    
    for model_name in model_names:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.0
            )
            # Test if it works by checking the model
            break
        except Exception:
            continue
    
    if llm is None:
        raise ValueError(f"Could not initialize any Gemini model. Tried: {model_names}")
    
    # Truncate content for efficiency (first ~2000 chars should be enough)
    truncated_content = content[:2000] + ("..." if len(content) > 2000 else "")
    
    prompt = f"""Analyze this technical documentation and provide:
1. A concise title (one line, starting with "How to" or describing the concept)
2. A brief explanation (2-3 sentences) of what this document covers

Document path: {file_path}

Document content:
{truncated_content}

Respond in exactly this format:
TITLE: <your title here>
EXPLANATION: <your explanation here>"""
    
    response = llm.invoke(prompt)
    response_text = response.content
    
    # Parse the response
    title_match = re.search(r'TITLE:\s*(.+?)(?=\nEXPLANATION:|\Z)', response_text, re.DOTALL)
    explanation_match = re.search(r'EXPLANATION:\s*(.+)', response_text, re.DOTALL)
    
    title = title_match.group(1).strip() if title_match else "Untitled Document"
    explanation = explanation_match.group(1).strip() if explanation_match else "No description available."
    
    # Clean up any markdown formatting that might have been added
    title = re.sub(r'^#+\s*', '', title).strip()
    
    return (title, explanation)


def extract_or_generate_summary(
    doc: Document,
    use_llm_fallback: bool = True,
    track_stats: bool = True
) -> Tuple[str, str]:
    """
    Extract or generate title and explanation for a document.
    
    Implements the Fuzzy -> LLM fallback pattern:
    1. Try deterministic extraction first
    2. Fall back to LLM generation if needed
    
    Args:
        doc: LangChain Document object with page_content and metadata
        use_llm_fallback: Whether to use LLM when deterministic extraction fails
        track_stats: Whether to track extraction statistics
        
    Returns:
        Tuple of (title, explanation)
    """
    content = doc.page_content
    file_path = doc.metadata.get('source', 'unknown')
    
    # Try deterministic extraction first
    result = _extract_summary_deterministic(content)
    
    if result is not None:
        if track_stats:
            ExtractionStats.deterministic_count += 1
        return result
    
    # Fallback to LLM if enabled
    if use_llm_fallback:
        if track_stats:
            ExtractionStats.llm_count += 1
        return _generate_summary_with_llm(content, file_path)
    
    # Final fallback: extract from file path and first line
    title = file_path.split('/')[-1].replace('.md', '').replace('-', ' ').title()
    
    # Try to get first meaningful line as explanation
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    explanation = lines[0] if lines else "No description available."
    
    return (title, explanation)

