import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn

def create_sequences(
    documents_ids, bos_id, eos_id, sequence_length=32
):
    if sequence_length < 1:
        raise ValueError("sequence_length pozitif olmalı.")

    inputs = []
    targets = []

    for document_ids in documents_ids:
        sequence = [bos_id] + document_ids + [eos_id]

        for start in range(
            0, len(sequence) - 1, sequence_length
        ):
            end = min(
                start + sequence_length,
                len(sequence) - 1,
            )

            inputs.append(sequence[start:end])
            targets.append(sequence[start + 1:end + 1])

    return inputs, targets

def collate_batch(batch, input_pad_id):
    inputs = [
        torch.tensor(x, dtype=torch.long)
        for x, y in batch
    ]

    targets = [
        torch.tensor(y, dtype=torch.long)
        for x, y in batch
    ]

    inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=input_pad_id,
    )

    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=-100,
    )

    return inputs, targets

class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        bos_id,
        embedding_dim=128,
        hidden_dim=128,
    ):
        super().__init__()

        self.bos_id = bos_id

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_dim, vocab_size
        )

    def forward(self, token_ids, hidden=None):
        embeddings = self.embedding(token_ids)

        states, hidden = self.lstm(
            embeddings, hidden
        )

        logits = self.output(states)

        # <bos> bağlamda kullanılabilir, ancak tahmin edilmemeli.
        logits[..., self.bos_id] = float("-inf")

        return logits, hidden

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum",
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.set_grad_enabled(training):
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            count = (targets != -100).sum().item()

            if count == 0:
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits, _ = model(inputs)

            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            if not torch.isfinite(loss_sum).item():
                raise ValueError("Loss sonlu değil.")

            if training:
                (loss_sum / count).backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )

                optimizer.step()

            total_loss += loss_sum.item()
            total_tokens += count

            if training and step % 100 == 0:
                print(
                    f"Batch {step}/{len(loader)} | "
                    f"loss: {total_loss / total_tokens:.4f}"
                )

    if total_tokens == 0:
        raise ValueError("İşlenecek hedef token bulunamadı.")

    return total_loss / total_tokens