# ML Training Pipeline

> Generates synthetic intent classification data and trains a lightweight PyTorch bag-of-words classifier for fast, low-cost intent routing.

## Overview

The ML Training Pipeline provides an alternative to LLM-based intent classification. It generates synthetic training data from templates, trains a simple linear classifier on bag-of-words features (including bigrams), and produces model artifacts that the `TorchRouter` loads at runtime.

**Why it exists:**
- LLM reroute calls are slow (~500ms) and expensive
- Rule-based routing has gaps for ambiguous utterances
- A lightweight BoW classifier provides a middle ground: fast inference (~1ms), reasonable accuracy, zero API cost

**Pipeline stages:**
1. **Data Generation** (`generate_intent_dataset.py`) - Expands templates into 25k synthetic utterances
2. **Training** (`train_intent_router.py`) - Trains linear classifier on BoW+bigram features
3. **Inference** (`torch_router.py`) - Loads artifacts and predicts intents at runtime

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/generate_intent_dataset.py` | Synthetic dataset generation |
| `scripts/train_intent_router.py` | Model training script |
| `ml/intent_templates.json` | Utterance templates and slots |
| `ml/intent_train.jsonl` | Training data (80%) |
| `ml/intent_val.jsonl` | Validation data (10%) |
| `ml/intent_test.jsonl` | Test data (10%) |
| `backend/torch_router.py` | Model loading and inference |
| `backend/artifacts/model_bundle_v1/intent_router/` | Trained model artifacts |

---

## Data Generation

### Template Structure

Templates live in `ml/intent_templates.json`. Each intent has a list of utterance templates with placeholders:

```json
{
  "intents": {
    "activity": [
      "What was my most recent trade?",
      "Show my recent activity",
      "Latest trade in my account"
    ],
    "positions": [
      "How many shares of {SYMBOL} do I own?",
      "Do I hold any {SYMBOL}?",
      "What's my {SYMBOL} position?"
    ],
    "symbol_performance": [
      "How is {SYMBOL} performing?",
      "What's my performance on {SYMBOL}?",
      "Is {SYMBOL} up or down?"
    ],
    "quotes": [
      "What's {SYMBOL} trading at?",
      "Give me the quote for {SYMBOL}",
      "{SYMBOL} price today"
    ]
  },
  "slots": {
    "SYMBOL": ["AAPL", "MSFT", "VOO", "TSLA", "NVDA", "AMZN", "GME", "GOOG", "META", "AMD", "SPY", "QQQ"],
    "TIMEFRAME": ["YTD", "1Y", "3M", "QTD", "this year", "this month", "last month"],
    "FACT_TOPIC": ["Roth IRA", "ETF", "rebalancing", "index fund", "mutual fund", "bond", "dividend"],
    "ACCOUNT": ["Brokerage", "IRA", "Roth IRA"]
  }
}
```

### Supported Intents

All 11 classifiable intents (excludes `clarify` which is a fallback):

- `activity` - Recent trades/transactions
- `positions` - Single symbol holdings
- `positions_list` - All holdings
- `portfolio_ranking` - Best/worst performers
- `symbol_performance` - Single symbol P/L
- `performance` - Overall portfolio performance
- `quotes` - Current prices
- `transfers` - Deposits/withdrawals
- `facts` - Educational content
- `account_value` - Total portfolio value
- `cash_balance` - Cash/settled amounts

### Generation Algorithm

```python
# scripts/generate_intent_dataset.py

TARGET_SIZE = int(os.getenv("INTENT_DATASET_SIZE", "25000"))
MAX_COMBOS = int(os.getenv("INTENT_MAX_COMBOS", "200"))

def _render_template(template: str, slots: dict[str, list[str]], max_combos: int) -> list[str]:
    """Expand placeholders with all slot combinations (up to max_combos)."""
    keys = PLACEHOLDER_RE.findall(template)
    if not keys:
        return [template]

    values = [slots.get(k, []) for k in keys]
    if any(not v for v in values):
        return [template]

    combos = list(product(*values))
    if len(combos) > max_combos:
        combos = combos[:max_combos]

    rendered = []
    for combo in combos:
        text = template
        for key, value in zip(keys, combo):
            text = text.replace(f"{{{key}}}", value)
        rendered.append(text)
    return rendered
```

### Data Augmentation

The generator applies multiple augmentation strategies:

**1. Variants** - Case, punctuation, and synonym variations:

```python
def _variants(text: str, typos: dict[str, list[str]]) -> Iterable[str]:
    variants = {text}
    variants.add(text.lower())
    variants.add(re.sub(r"[\?\.!,,]", "", text))

    # Apply typos from config
    for word, typo_list in typos.items():
        for typo in typo_list:
            if word in text.lower():
                variants.add(re.sub(word, typo, text, flags=re.IGNORECASE))

    # Symbol-specific variants
    m = re.search(r"\b([A-Z]{1,5})\b", text)
    if m and "performance" in text.lower():
        variants.add(f"{m.group(1)} perf")
        variants.add(f"{m.group(1)} performance")

    return variants
```

**2. Affixes** - Random prefix/suffix injection:

```python
def _augment_with_affixes(text: str, prefixes: list[str], suffixes: list[str],
                          slots: dict[str, list[str]], rng: random.Random) -> str:
    prefix = rng.choice(prefixes) if prefixes else ""
    suffix = rng.choice(suffixes) if suffixes else ""
    if "{ACCOUNT}" in suffix:
        accounts = slots.get("ACCOUNT", [])
        if accounts:
            suffix = suffix.replace("{ACCOUNT}", rng.choice(accounts))
    parts = [prefix.strip(), text.strip(), suffix.strip()]
    return " ".join([p for p in parts if p]).strip()
```

**Configured prefixes:**
```json
["please", "can you", "could you", "hey", "hi", "quick question",
 "i want to know", "tell me", "show me", "help me with"]
```

**Configured suffixes:**
```json
["please", "right now", "today", "for my account",
 "for my brokerage account", "in my {ACCOUNT}"]
```

**3. Typo injection:**
```json
{
  "typos": {
    "performance": ["perfomance", "performnce"],
    "positions": ["postions", "posiions"],
    "quote": ["qoute"],
    "balance": ["balnce"],
    "portfolio": ["portoflio"]
  }
}
```

### Data Split

```python
def _split_dataset(items: list[dict[str, str]]) -> tuple[...]:
    random.shuffle(items)
    n = len(items)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return items[:train_end], items[train_end:val_end], items[val_end:]
```

**Default output:** 25,000 examples total
- Train: 20,000 (80%)
- Val: 2,500 (10%)
- Test: 2,500 (10%)

### JSONL Format

Each line is a JSON object with `text` and `intent`:

```json
{"text": "can you Portfolio return this month today", "intent": "performance"}
{"text": "can you How is GOOG doing today? please", "intent": "symbol_performance"}
{"text": "META quote in my account", "intent": "quotes"}
{"text": "help me with Account perfomance YTD in my Roth IRA", "intent": "performance"}
```

---

## Training

### Hyperparameters

```python
# scripts/train_intent_router.py

SEED = 7
MAX_VOCAB = 6000
NGRAM_MAX = 2
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-2
```

### Feature Extraction

**Tokenization:**
```python
TOKEN_PATTERN = r"[A-Za-z0-9']+"
TOKEN_RE = re.compile(TOKEN_PATTERN)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]
```

**Bigram features:**
```python
def extract_features(tokens: List[str], ngram_max: int) -> List[str]:
    features = list(tokens)
    if ngram_max >= 2:
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")
    return features
```

**Vocabulary building:**
```python
def build_vocab(texts: List[str], max_vocab: int, ngram_max: int) -> dict[str, int]:
    freq: dict[str, int] = {}
    for text in texts:
        tokens = tokenize(text)
        for feature in extract_features(tokens, ngram_max):
            freq[feature] = freq.get(feature, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    vocab_tokens = [t for t, _ in sorted_tokens[:max_vocab]]
    return {token: idx for idx, token in enumerate(vocab_tokens)}
```

**Bag-of-words encoding:**
```python
def encode_text(text: str, vocab: dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    tokens = tokenize(text)
    for feature in extract_features(tokens, NGRAM_MAX):
        idx = vocab.get(feature)
        if idx is not None:
            vec[idx] += 1.0
    return vec
```

### Model Architecture

A single linear layer (logistic regression equivalent):

```python
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

**Input:** Sparse BoW vector of size `MAX_VOCAB` (6000)
**Output:** Logits for 11 intent classes

### Dataset Class

```python
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
```

### Training Loop

```python
def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_jsonl(DATA_DIR / "intent_train.jsonl")
    val_rows = load_jsonl(DATA_DIR / "intent_val.jsonl")
    test_rows = load_jsonl(DATA_DIR / "intent_test.jsonl")

    labels = sorted({row["intent"] for row in train_rows})
    vocab = build_vocab([row["text"] for row in train_rows], MAX_VOCAB, NGRAM_MAX)

    # Create datasets and loaders
    train_ds = IntentDataset(train_rows, vocab, labels)
    val_ds = IntentDataset(val_rows, vocab, labels)
    test_ds = IntentDataset(test_rows, vocab, labels)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=batchify
    )

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
```

### Evaluation

```python
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
```

### Output Artifacts

Training saves four files to `backend/artifacts/model_bundle_v1/intent_router/`:

| File | Content |
|------|---------|
| `intent_router.pt` | Model weights (`state_dict`) |
| `intent_router_vocab.json` | Token-to-index mapping |
| `intent_router_labels.json` | Sorted list of intent labels |
| `intent_router_config.json` | Training hyperparameters |

**Example config:**
```json
{
  "max_vocab": 6000,
  "token_pattern": "[A-Za-z0-9']+",
  "ngram_max": 2,
  "epochs": 12,
  "batch_size": 32,
  "learning_rate": 0.01
}
```

---

## Runtime Inference

### TorchRouter Class

```python
# backend/torch_router.py

@dataclass
class TorchRouter:
    model: nn.Module
    vocab: dict[str, int]
    labels: list[str]
    token_re: re.Pattern
    ngram_max: int

    def _encode(self, text: str) -> torch.Tensor:
        vec = torch.zeros(len(self.vocab), dtype=torch.float32)
        tokens = [t.lower() for t in self.token_re.findall(text)]
        features = list(tokens)
        if self.ngram_max >= 2:
            for i in range(len(tokens) - 1):
                features.append(f"{tokens[i]}_{tokens[i+1]}")
        for feature in features:
            idx = self.vocab.get(feature)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def predict(self, text: str) -> Tuple[str, float]:
        self.model.eval()
        with torch.no_grad():
            x = self._encode(text).unsqueeze(0)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)
            return self.labels[int(idx)], float(conf.item())
```

### Model Loading (Cached)

```python
@lru_cache
def _load_router(model_path: str, vocab_path: str, labels_path: str) -> Optional[TorchRouter]:
    if torch is None:
        return None

    model_file = Path(model_path)
    vocab_file = Path(vocab_path)
    labels_file = Path(labels_path)

    if not (model_file.exists() and vocab_file.exists() and labels_file.exists()):
        return None

    vocab = json.loads(vocab_file.read_text())
    labels = json.loads(labels_file.read_text())
    config = _load_router_config()
    token_re = re.compile(config.get("token_pattern", TOKEN_PATTERN))
    ngram_max = int(config.get("ngram_max", 1))

    model = IntentClassifier(len(vocab), len(labels))
    state = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return TorchRouter(model=model, vocab=vocab, labels=labels, token_re=token_re, ngram_max=ngram_max)
```

### Primary Entry Point

```python
def torch_attempt(utterance: str, candidates: list[dict[str, float]]) -> tuple[Optional[IntentRoute], dict[str, object]]:
    """Attempt torch classification. Returns (route, metadata)."""

    model_path = os.getenv("TORCH_ROUTER_MODEL_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router.pt")
    vocab_path = os.getenv("TORCH_ROUTER_VOCAB_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router_vocab.json")
    labels_path = os.getenv("TORCH_ROUTER_LABELS_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router_labels.json")
    status = torch_router_status()

    router = _load_router(model_path, vocab_path, labels_path)
    if router is None:
        return None, {**status, "torch_attempted": False, "torch_used": False}

    label, confidence = router.predict(utterance)

    try:
        intent = Intent(label)
    except ValueError:
        return None, {**status, "torch_attempted": True, "torch_used": False,
                      "torch_label": label, "torch_confidence": round(confidence, 4)}

    min_conf = float(status["confidence_threshold"])
    force = bool(status["force_enabled"])

    if confidence < min_conf and not force:
        return None, {**status, "torch_attempted": True, "torch_used": False,
                      "torch_label": intent.value, "torch_confidence": round(confidence, 4)}

    route = IntentRoute(
        intent=intent,
        confidence=confidence,
        missing_params=[],
        extracted={},
        candidates=candidates,
        routing_mode="torch_classifier",
    )

    return route, {**status, "torch_attempted": True, "torch_used": True,
                   "torch_label": intent.value, "torch_confidence": round(confidence, 4)}
```

### Status Check

```python
def torch_router_status() -> dict[str, object]:
    """Return diagnostic info about torch router availability."""
    model_path = os.getenv("TORCH_ROUTER_MODEL_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router.pt")
    vocab_path = os.getenv("TORCH_ROUTER_VOCAB_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router_vocab.json")
    labels_path = os.getenv("TORCH_ROUTER_LABELS_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router_labels.json")
    config = _load_router_config()

    artifacts_present = all(Path(p).exists() for p in [model_path, vocab_path, labels_path])
    torch_available = torch is not None

    return {
        "torch_available": torch_available,
        "artifacts_present": artifacts_present,
        "enabled": torch_available and artifacts_present,
        "confidence_threshold": float(os.getenv("TORCH_ROUTER_CONF_THRESHOLD", "0.6")),
        "force_enabled": os.getenv("TORCH_ROUTER_FORCE", "").lower() in {"1", "true", "yes"},
        "ngram_max": config.get("ngram_max"),
        "token_pattern": config.get("token_pattern"),
    }
```

---

## Integration with Intent Router

The torch router sits in layer 2 of the three-layer routing cascade in `backend/router.py`:

```python
def route_intent(utterance: str, use_llm: bool = True) -> IntentRoute:
    # Layer 1: Rule-based scoring
    rule_result = rule_route(utterance)

    threshold = float(os.getenv("ROUTER_CONF_THRESHOLD", "0.75"))
    candidates = rule_result.candidates
    ambiguous = len(candidates) >= 2 and abs(candidates[0]["score"] - candidates[1]["score"]) < 0.1

    # High-confidence rule match -> return immediately
    if rule_result.confidence >= threshold and not ambiguous:
        return rule_result

    # Layer 2: Torch classifier attempt
    torch_route, torch_meta = torch_attempt(utterance, candidates)
    if torch_route:
        # Apply extraction (symbol, account, etc.) to torch result
        text = _normalize(utterance)
        intent, extracted, missing = _apply_extraction(torch_route.intent, text, utterance)
        torch_route.intent = intent
        torch_route.extracted.update(extracted)
        torch_route.missing_params = missing
        torch_route.routing_meta = torch_meta
        return torch_route

    # Layer 3: LLM fallback (if API key present)
    if os.getenv("OPENAI_API_KEY"):
        reroute = llm_reroute(utterance, ...)
        if reroute:
            return reroute

    return rule_result
```

**Trigger conditions for torch routing:**
1. Rule-based confidence < `ROUTER_CONF_THRESHOLD` (default 0.75)
2. OR top two candidates are ambiguous (score difference < 0.1)

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INTENT_DATASET_SIZE` | `25000` | Target number of generated examples |
| `INTENT_MAX_COMBOS` | `200` | Max slot combinations per template |
| `TORCH_ROUTER_CONF_THRESHOLD` | `0.6` | Min confidence to use torch prediction |
| `TORCH_ROUTER_FORCE` | `false` | Force torch routing regardless of confidence |
| `TORCH_ROUTER_MODEL_PATH` | `artifacts/.../intent_router.pt` | Custom model path |
| `TORCH_ROUTER_VOCAB_PATH` | `artifacts/.../intent_router_vocab.json` | Custom vocab path |
| `TORCH_ROUTER_LABELS_PATH` | `artifacts/.../intent_router_labels.json` | Custom labels path |

---

## Error Handling

### PyTorch Not Installed

```python
try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None
```

If PyTorch is unavailable, `torch_attempt()` returns `None` gracefully and the router falls through to LLM reroute or rule-based result.

### Missing Artifacts

```python
if not (model_file.exists() and vocab_file.exists() and labels_file.exists()):
    return None
```

Missing model files trigger graceful fallback. Status endpoint exposes `artifacts_present: false`.

### Unknown Intent Label

```python
try:
    intent = Intent(label)
except ValueError:
    # Model predicted a label not in Intent enum
    return None, {..., "torch_used": False, "torch_label": label}
```

This can happen if the training data has intents not defined in `schemas.py`. The classifier output is logged but not used.

### Low Confidence

```python
if confidence < min_conf and not force:
    return None, {..., "torch_used": False, "torch_confidence": confidence}
```

Below-threshold predictions are rejected unless `TORCH_ROUTER_FORCE=true`.

---

## Usage

### Generate Dataset

```bash
# Default: 25k examples
uv run python scripts/generate_intent_dataset.py

# Custom size
INTENT_DATASET_SIZE=50000 uv run python scripts/generate_intent_dataset.py
```

### Train Model

```bash
# Requires PyTorch
uv run python scripts/train_intent_router.py
```

**Expected output:**
```
Epoch 1/12 - loss 45.231 - val_acc 0.892
Epoch 2/12 - loss 12.456 - val_acc 0.934
...
Epoch 12/12 - loss 2.341 - val_acc 0.978
Test accuracy: 0.976
Saved artifacts to backend/artifacts/model_bundle_v1/intent_router
```

### Verify Integration

```python
from backend.torch_router import torch_router_status

status = torch_router_status()
print(status)
# {'torch_available': True, 'artifacts_present': True, 'enabled': True,
#  'confidence_threshold': 0.6, 'force_enabled': False, 'ngram_max': 2, ...}
```

---

## Limitations and Gotchas

1. **No parameter extraction** - Torch router only predicts intent. Symbol, account, and other params are extracted by `_apply_extraction()` after classification.

2. **Synthetic data bias** - All training data comes from templates, so novel phrasings may have lower confidence.

3. **Bigram sensitivity** - Word order matters for bigram features. "AAPL price" vs "price AAPL" produce different features.

4. **Vocab truncation** - Only top 6000 tokens are kept. Rare tokens (including rare symbols) may be OOV.

5. **No incremental training** - Must regenerate full dataset and retrain from scratch to add new intents or templates.

6. **CPU-only inference** - Model loads on CPU via `map_location="cpu"`. No GPU acceleration.

7. **LRU cache on model loading** - Model is cached per (model_path, vocab_path, labels_path) tuple. Changing files requires process restart to clear cache.

---

## Potential Improvements

**From code comments and TODOs:**
- None explicitly marked in source

**Architectural considerations:**
- Add character n-grams for typo robustness
- Use TF-IDF weighting instead of raw counts
- Add hidden layer for better feature interaction
- Implement online learning for new examples
- Add confidence calibration (softmax scores are often overconfident)
