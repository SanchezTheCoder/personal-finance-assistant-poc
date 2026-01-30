#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import List, Tuple

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for training. Install torch and retry.") from exc

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml"
OUT_DIR = ROOT / "backend" / "artifacts" / "model_bundle_v1" / "intent_router"

TOKEN_PATTERN = r"[A-Za-z0-9']+"
TOKEN_RE = re.compile(TOKEN_PATTERN)
SEED = 7
MAX_VOCAB = 6000
NGRAM_MAX = 2
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-2


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[dict[str, str]], vocab: dict[str, int], labels: List[str]):
        self.rows = rows
        self.vocab = vocab
        self.label_to_idx = {label: i for i, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = encode_text(row["text"], self.vocab)
        y = self.label_to_idx[row["intent"]]
        return x, y


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def extract_features(tokens: List[str], ngram_max: int) -> List[str]:
    features = list(tokens)
    if ngram_max >= 2:
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")
    return features


def build_vocab(texts: List[str], max_vocab: int, ngram_max: int) -> dict[str, int]:
    freq: dict[str, int] = {}
    for text in texts:
        tokens = tokenize(text)
        for feature in extract_features(tokens, ngram_max):
            freq[feature] = freq.get(feature, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    vocab_tokens = [t for t, _ in sorted_tokens[:max_vocab]]
    return {token: idx for idx, token in enumerate(vocab_tokens)}


def encode_text(text: str, vocab: dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    tokens = tokenize(text)
    for feature in extract_features(tokens, NGRAM_MAX):
        idx = vocab.get(feature)
        if idx is not None:
            vec[idx] += 1.0
    return vec


def load_jsonl(path: Path) -> List[dict[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def batchify(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_jsonl(DATA_DIR / "intent_train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "intent_val.jsonl")
    test_rows = load_jsonl(DATA_DIR / "intent_test.jsonl")

    labels = sorted({row["intent"] for row in train_rows})
    vocab = build_vocab([row["text"] for row in train_rows], MAX_VOCAB, NGRAM_MAX)

    train_ds = IntentDataset(train_rows, vocab, labels)
    val_ds = IntentDataset(val_rows, vocab, labels)
    test_ds = IntentDataset(test_rows, vocab, labels)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=batchify)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=batchify)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=batchify)

    model = IntentClassifier(len(vocab), len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss {total_loss:.3f} - val_acc {val_acc:.3f}")

    test_acc = evaluate(model, test_loader)
    print(f"Test accuracy: {test_acc:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "intent_router.pt")
    (OUT_DIR / "intent_router_vocab.json").write_text(json.dumps(vocab, indent=2))
    (OUT_DIR / "intent_router_labels.json").write_text(json.dumps(labels, indent=2))
    (OUT_DIR / "intent_router_config.json").write_text(
        json.dumps(
            {
                "max_vocab": MAX_VOCAB,
                "token_pattern": TOKEN_PATTERN,
                "ngram_max": NGRAM_MAX,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
            },
            indent=2,
        )
    )

    print(f"Saved artifacts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
