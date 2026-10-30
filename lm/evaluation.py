import math
from lm.ngram import smoothed_probability

def perplexity(
    documents,
    n,
    ngram_counts,
    context_counts,
    vocab,
    k=0.01,
):
    if n not in (2, 3):
        raise ValueError("n değeri 2 veya 3 olmalı.")

    negative_log_sum = 0.0
    prediction_count = 0

    for document in documents:
        tokens = (
            ["<bos>"] * (n - 1)
            + document["tokens"]
            + ["<eos>"]
        )

        for i in range(n - 1, len(tokens)):
            context = tuple(tokens[i - n + 1:i])
            word = tokens[i]

            probability = smoothed_probability(
                context,
                word,
                ngram_counts,
                context_counts,
                vocab,
                k=k,
            )

            if probability <= 0:
                return math.inf

            negative_log_sum -= math.log(probability)
            prediction_count += 1

    if prediction_count == 0:
        raise ValueError("Değerlendirilecek token bulunamadı.")

    return math.exp(negative_log_sum / prediction_count)