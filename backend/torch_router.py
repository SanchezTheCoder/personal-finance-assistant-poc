from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from .schemas import Intent, IntentRoute

TOKEN_PATTERN = r"[A-Za-z0-9']+"
TOKEN_RE = re.compile(TOKEN_PATTERN)
DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "model_bundle_v1" / "intent_router"
DEFAULT_CONFIG_PATH = DEFAULT_ARTIFACT_DIR / "intent_router_config.json"


if nn is not None:
    class IntentClassifier(nn.Module):
        def __init__(self, vocab_size: int, num_classes: int):
            super().__init__()
            self.linear = nn.Linear(vocab_size, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)
else:  # pragma: no cover
    IntentClassifier = None  # type: ignore


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
    if IntentClassifier is None:
        return None
    model = IntentClassifier(len(vocab), len(labels))
    state = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return TorchRouter(model=model, vocab=vocab, labels=labels, token_re=token_re, ngram_max=ngram_max)


def _load_router_config() -> dict[str, object]:
    if DEFAULT_CONFIG_PATH.exists():
        try:
            return json.loads(DEFAULT_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def torch_router_status() -> dict[str, object]:
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


def torch_attempt(utterance: str, candidates: list[dict[str, float]]) -> tuple[Optional[IntentRoute], dict[str, object]]:
    model_path = os.getenv("TORCH_ROUTER_MODEL_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router.pt")
    vocab_path = os.getenv("TORCH_ROUTER_VOCAB_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router_vocab.json")
    labels_path = os.getenv("TORCH_ROUTER_LABELS_PATH") or str(DEFAULT_ARTIFACT_DIR / "intent_router_labels.json")
    status = torch_router_status()

    router = _load_router(model_path, vocab_path, labels_path)
    if router is None:
        meta = {
            **status,
            "torch_attempted": False,
            "torch_used": False,
        }
        return None, meta

    label, confidence = router.predict(utterance)
    try:
        intent = Intent(label)
    except ValueError:
        meta = {
            **status,
            "torch_attempted": True,
            "torch_used": False,
            "torch_label": label,
            "torch_confidence": round(confidence, 4),
        }
        return None, meta

    min_conf = float(status["confidence_threshold"])
    force = bool(status["force_enabled"])
    if confidence < min_conf and not force:
        meta = {
            **status,
            "torch_attempted": True,
            "torch_used": False,
            "torch_label": intent.value,
            "torch_confidence": round(confidence, 4),
        }
        return None, meta

    route = IntentRoute(
        intent=intent,
        confidence=confidence,
        missing_params=[],
        extracted={},
        candidates=candidates,
        routing_mode="torch_classifier",
    )
    meta = {
        **status,
        "torch_attempted": True,
        "torch_used": True,
        "torch_label": intent.value,
        "torch_confidence": round(confidence, 4),
    }
    return route, meta


def torch_reroute(utterance: str, candidates: list[dict[str, float]]) -> Optional[IntentRoute]:
    route, _meta = torch_attempt(utterance, candidates)
    return route
