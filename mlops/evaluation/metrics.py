# mlops/evaluation/metrics.py

def recall_at_k(contexts, expected_keywords):
    """
    Checks if any expected keyword appears in retrieved contexts
    """
    for ctx in contexts:
        text = ctx["text"] if isinstance(ctx, dict) else ctx
        for kw in expected_keywords:
            if kw.lower() in text.lower():
                return 1
    return 0


def faithfulness(answer, contexts):
    """
    Measures whether answer sentences are grounded in retrieved context
    """
    supported = 0
    answer_sents = [s.strip() for s in answer.split(".") if s.strip()]

    for sent in answer_sents[:2]:
        for ctx in contexts:
            text = ctx["text"] if isinstance(ctx, dict) else ctx
            if sent.lower() in text.lower():
                supported += 1
                break

    return supported / max(len(answer_sents), 1)


def hallucination_rate(answer, contexts):
    """
    % of answer sentences not supported by context
    """
    answer_sents = [s.strip() for s in answer.split(".") if s.strip()]
    hallucinated = 0

    for sent in answer_sents:
        supported = False
        for ctx in contexts:
            text = ctx["text"] if isinstance(ctx, dict) else ctx
            if sent.lower() in text.lower():
                supported = True
                break
        if not supported:
            hallucinated += 1

    return hallucinated / max(len(answer_sents), 1)



