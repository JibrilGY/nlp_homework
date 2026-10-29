import re

def clean_text(text):
    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        # Dikey çizgilerle oluşturulan tablo satırlarını kaldır.
        if stripped.startswith("|") and stripped.count("|") >= 2:
            continue

        # Yalnızca dekoratif çizgi karakterlerinden oluşan satırları kaldır.
        if re.fullmatch(r"[-+|=_*\s]{3,}", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def tokenize(text):
    # Normalize curly apostrophes
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    # Lowercase before creating the token list
    text = text.lower()
    # Keep apostrophes inside words; return punctuation as separate tokens.
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)

def replace_unknown(tokens, vocab):
    return [
        token if token in vocab else "<unk>"
        for token in tokens
    ]

