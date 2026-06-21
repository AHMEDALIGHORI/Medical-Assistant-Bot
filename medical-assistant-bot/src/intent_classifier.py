"""DistilBERT intent classifier for medical queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class IntentClassifier:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        with open(self.model_dir / "label_map.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.label2id = meta["label2id"]

    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())
        return self.id2label[pred_id], confidence

    @classmethod
    def load_if_exists(cls, model_dir: str | Path) -> IntentClassifier | None:
        path = Path(model_dir)
        if (path / "config.json").exists() and (path / "label_map.json").exists():
            return cls(path)
        return None


def keyword_intent_fallback(text: str, labels: list[str]) -> str:
    lowered = text.lower()
    rules = {
        "emergency": ["emergency", "can't breathe", "chest pain", "unconscious", "heart attack"],
        "medication": ["medicine", "drug", "dose", "aspirin", "ibuprofen", "antibiotic", "pill"],
        "symptoms": ["symptom", "fever", "pain", "cough", "nausea", "headache", "feel"],
        "prevention": ["prevent", "avoid", "vaccine", "hygiene", "exercise", "diet"],
    }
    for intent, keywords in rules.items():
        if intent in labels and any(k in lowered for k in keywords):
            return intent
    return "general_info" if "general_info" in labels else labels[0]
