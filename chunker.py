def chunk_text_per_page(source_name, pages, chunk_size=500, overlap=100):
    """
    Splits PDF pages into overlapping chunks with metadata for source and chunk id.
    """
    chunks = []
    for page_id, text in enumerate(pages):
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "page_id": page_id,
                    "chunk_id": f"{page_id}_{start}"
                }
            })
            start += chunk_size - overlap
    return chunks
