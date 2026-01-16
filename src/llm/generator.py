from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200,
    do_sample=False,
    truncation=True
)

def generate_answer(question, contexts):
    normalized = []

    for i, c in enumerate(contexts):
        if isinstance(c, dict):
            text = c.get("text", "")
            page = c.get("page", i)
        else:
            text = str(c)
            page = i

        text = text.strip()
        if text:
            normalized.append(f"(Page {page}) {text}")

    context_text = "\n".join(normalized)

    if not context_text.strip():
        return "Not found in the provided document."

    prompt = f"""
You are a helpful assistant.

Answer the QUESTION using only the CONTEXT.
Return a clean short answer.
If the answer is missing, reply exactly: Not found in the provided document.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""

    return generator(prompt)[0]["generated_text"].strip()

