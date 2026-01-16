def faithfulness(answer, contexts):
    """
    Measures how much of the answer is supported by retrieved contexts
    """

    if not answer or not contexts:
        return 0.0

    answer_sents = [s.strip() for s in answer.split(".") if s.strip()]
    supported = 0

    for sent in answer_sents[:2]:  # check first 2 sentences
        for ctx in contexts:
            text = ctx.get("text", "")
            if sent.lower() in text.lower():
                supported += 1
                break

    return supported / max(len(answer_sents[:2]), 1)

