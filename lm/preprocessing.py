import re

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

