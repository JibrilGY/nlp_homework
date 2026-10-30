import random
import re
import torch

from lm.ngram import smoothed_probability
from lm.preprocessing import tokenize, replace_unknown


def generate_text(
    prompt, n, ngram_counts, context_counts, vocab,
    max_new_tokens=30, k=0.01, top_k=10, seed=42,
):
    if n not in (2, 3):
        raise ValueError("n değeri 2 veya 3 olmalı.")

    if top_k < 1:
        raise ValueError("top_k en az 1 olmalı.")

    rng = random.Random(seed)

    prompt_tokens = tokenize(prompt)
    history = (
        ["<bos>"] * (n - 1)
        + replace_unknown(prompt_tokens, vocab)
    )

    candidates = sorted(vocab - {"<bos>", "<unk>"})
    generated = []

    for _ in range(max_new_tokens):
        context = tuple(history[-(n - 1):])

        scores = [
            (
                word,
                smoothed_probability(
                    context, word, ngram_counts,
                    context_counts, vocab, k=k,
                ),
            )
            for word in candidates
        ]

        # Eşit olasılıklarda alfabetik seçim oluşmasını önle.
        rng.shuffle(scores)
        scores.sort(key=lambda item: item[1], reverse=True)
        best = scores[:top_k]

        words = [word for word, probability in best]
        weights = [probability for word, probability in best]

        word = rng.choices(words, weights=weights, k=1)[0]

        if word == "<eos>":
            break

        generated.append(word)
        history.append(word)

    text = " ".join(prompt_tokens + generated)

    # Noktalama işaretlerinden önceki boşlukları kaldır.
    return re.sub(r"\s+([.,!?;:])", r"\1", text)
@torch.no_grad()
def generate_lstm_text(
    model, prompt, token_to_id, id_to_token,
    max_new_tokens=30, temperature=1.0,
    top_k=10, seed=42,
):
    if temperature <= 0 or top_k < 1:
        raise ValueError(
            "temperature pozitif, top_k en az 1 olmalı."
        )

    model.eval()

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)

    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]
    unk_id = token_to_id["<unk>"]

    prompt_tokens = tokenize(prompt)

    prompt_ids = [bos_id] + [
        token_to_id.get(token, unk_id)
        for token in prompt_tokens
    ]

    inputs = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    hidden = None
    generated = []

    for _ in range(max_new_tokens):
        logits, hidden = model(inputs, hidden)

        scores = logits[0, -1].clone() / temperature
        scores[bos_id] = float("-inf")
        scores[unk_id] = float("-inf")

        values, indices = torch.topk(
            scores,
            k=min(top_k, len(token_to_id) - 2),
        )

        probabilities = torch.softmax(values, dim=-1)

        sampled = torch.multinomial(
            probabilities, 1, generator=generator
        ).item()

        next_id = indices[sampled].item()

        if next_id == eos_id:
            break

        generated.append(id_to_token[next_id])

        inputs = torch.tensor(
            [[next_id]],
            dtype=torch.long,
            device=device,
        )

    text = " ".join(prompt_tokens + generated)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)