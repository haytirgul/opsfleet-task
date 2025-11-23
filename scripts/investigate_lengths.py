"""Investigate content length mismatches in the index file."""

import json
from pathlib import Path

# Load files
combined_file = Path("data/output/documents_combined.json")
index_file = Path("data/output/documents_index.json")

with open(combined_file, "r", encoding="utf-8") as f:
    combined_data = json.load(f)

with open(index_file, "r", encoding="utf-8") as f:
    index_data = json.load(f)

print("="*70)
print("CONTENT LENGTH INVESTIGATION")
print("="*70)

# Check first few documents
print("\nChecking first 10 documents:")
print("-"*70)
mismatches = []

for i in range(min(10, len(combined_data), len(index_data))):
    actual_length = len(combined_data[i]["page_content"])
    index_length = index_data[i]["content_length"]
    diff = actual_length - index_length
    
    if diff != 0:
        mismatches.append(i)
        status = "[MISMATCH]"
    else:
        status = "[OK]"
    
    print(f"{status} Doc {i}: Index={index_length:6d}, Actual={actual_length:6d}, Diff={diff:+6d}")
    print(f"    Source: {index_data[i].get('source', 'N/A')}")
    print(f"    Title: {index_data[i].get('title', 'N/A')[:50]}")

# Check specific problematic ones from the index
print("\n" + "="*70)
print("Checking specific problematic documents:")
print("-"*70)

problem_indices = [2, 3, 22, 24, 25, 29, 31, 37, 45, 49, 58]

for idx in problem_indices:
    if idx < len(combined_data) and idx < len(index_data):
        actual_length = len(combined_data[idx]["page_content"])
        index_length = index_data[idx]["content_length"]
        diff = actual_length - index_length
        
        print(f"\nDoc {idx}:")
        print(f"  Index length: {index_length}")
        print(f"  Actual length: {actual_length}")
        print(f"  Difference: {diff:+d}")
        print(f"  Source: {index_data[idx].get('source', 'EMPTY')}")
        print(f"  Title: {index_data[idx].get('title', 'EMPTY')}")
        print(f"  Explanation preview: {index_data[idx].get('explanation', '')[:100]}...")
        
        # Show first 200 chars of actual content
        content_preview = combined_data[idx]["page_content"][:200].replace("\n", "\\n")
        print(f"  Content preview: {content_preview}...")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
total_mismatches = sum(1 for i in range(min(len(combined_data), len(index_data))) 
                      if len(combined_data[i]["page_content"]) != index_data[i]["content_length"])
print(f"Total documents: {len(combined_data)}")
print(f"Documents with length mismatches: {total_mismatches}")
print(f"Match rate: {(len(combined_data) - total_mismatches) / len(combined_data) * 100:.1f}%")

