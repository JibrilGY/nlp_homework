from collections import Counter

def count_ngrams(documents, n):
    if n not in (2, 3):
        raise ValueError("n değeri 2 veya 3 olmalı.")

    ngram_counts = Counter()
    context_counts = Counter()

    for document in documents:
        tokens = (
            ["<bos>"] * (n - 1)
            + document["tokens"]
            + ["<eos>"]
        )

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            context = ngram[:-1]

            ngram_counts[ngram] += 1
            context_counts[context] += 1

    return ngram_counts, context_counts

def next_word_probability(
    context, word, ngram_counts, context_counts
):
    context = tuple(context)
    total = context_counts[context]

    if total == 0:
        return 0.0

    return ngram_counts[context + (word,)] / total

def top_next_words(
    context, ngram_counts, context_counts, k=5
):
    context = tuple(context)
    total = context_counts[context]

    if total == 0:
        return []

    candidates = []

    for sequence, count in ngram_counts.items():
        if sequence[:-1] == context:
            word = sequence[-1]
            probability = count / total
            candidates.append((word, probability))

    candidates.sort(key=lambda item: item[1], reverse=True)

    return candidates[:k]

def smoothed_probability(
    context,
    word,
    ngram_counts,
    context_counts,
    vocab,
    k=0.01,
):
    if k <= 0:
        raise ValueError("k pozitif olmalı.")

    context = tuple(context)

    # Başlangıç belirteci, tahmin edilecek bir token değil.
    if word == "<bos>" or word not in vocab:
        return 0.0

    prediction_vocab_size = len(vocab) - int("<bos>" in vocab)

    count = ngram_counts[context + (word,)]
    total = context_counts[context]

    return (count + k) / (total + k * prediction_vocab_size)