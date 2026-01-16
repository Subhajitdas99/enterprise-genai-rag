def truncate_text(text: str, max_tokens: int = 450):
    """
    Hard truncate text to avoid transformer overflow.
    450 keeps buffer for prompt tokens.
    """
    return text[: max_tokens * 4]  # ~4 chars per token
