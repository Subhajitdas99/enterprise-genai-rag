def recall_at_k(retrieved_docs, expected_keywords):
    hits = 0
    for doc in retrieved_docs:
        for kw in expected_keywords:
            if kw.lower() in doc.lower():
                hits += 1
                break
    return int(hits > 0)


