import re
from pypdf import PdfReader


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    # 1) Convert newlines to spaces + remove extra spaces
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 2) Fix broken hyphen words: "well - being" -> "well-being"
    text = re.sub(r"\s-\s", "-", text)

    # 3) Fix spaced-out characters issue:
    # "w e l l - b e i n g" -> "well-being"
    text = re.sub(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)

    # 4) Add missing spaces after punctuation
    text = re.sub(r",(?=\S)", ", ", text)
    text = re.sub(r"\.(?=\S)", ". ", text)
    text = re.sub(r":(?=\S)", ": ", text)

    # 5) Split camelCase: "Urbanhealthsurveillance" -> "Urbanhealthsurveillance" (weak)
    # Better split: lower->Upper transitions
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # 6) Fix merged words around numbers: "Jan26" -> "Jan 26"
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

    # 7) Remove repeated spaces again
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_pdf(path: str):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        cleaned = clean_pdf_text(raw_text)

        if cleaned:
            pages.append({
                "page": i + 1,
                "text": cleaned
            })

    return pages





