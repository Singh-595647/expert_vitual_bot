import re
from typing import List

def overlapping_chunks(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks. Each chunk has up to max_tokens words,
    and each chunk overlaps the previous by `overlap` words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += max_tokens - overlap
    return chunks
