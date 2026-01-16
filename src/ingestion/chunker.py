def chunk_text(pages, chunk_size=800, overlap=150):
    assert chunk_size > overlap, "chunk_size must be > overlap"

    chunks = []

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page", -1)

        # normalize spaces
        text = " ".join(str(text).split())
        if not text:
            continue

        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page_num
                })

            start += chunk_size - overlap

    return chunks

