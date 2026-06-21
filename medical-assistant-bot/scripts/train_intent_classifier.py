"""Fine-tune DistilBERT intent classifier on medical FAQ questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from utils import ensure_dirs, load_config, resolve_path, set_seed  # noqa: E402


class IntentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_training_data(csv_path: Path, default_labels: list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        sample = resolve_path("data/sample/medical_faq.csv")
        df = pd.read_csv(sample)
    else:
        df = pd.read_csv(csv_path)
    df = df.dropna(subset=["question"])
    df["intent"] = df["intent"].fillna("unknown")
    df["intent"] = df["intent"].apply(lambda x: x if x in default_labels else "unknown")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    intent_cfg = config["intent"]
    labels = list(intent_cfg["labels"])
    intent_cfg["epochs"] = max(intent_cfg.get("epochs", 3), 5)
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    csv_path = resolve_path(config["paths"]["data_processed"], config) / "intent_train.csv"
    df = load_training_data(csv_path, labels)
    print(f"Training samples: {len(df)}")

    train_df, eval_df = train_test_split(
        df,
        test_size=1 - intent_cfg["train_split"],
        random_state=config.get("seed", 42),
        stratify=df["intent"] if df["intent"].nunique() > 1 else None,
    )

    model_name = intent_cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    max_len = intent_cfg["max_length"]
    train_ds = IntentDataset(
        train_df["question"].tolist(),
        [label2id[i] for i in train_df["intent"]],
        tokenizer,
        max_len,
    )
    eval_ds = IntentDataset(
        eval_df["question"].tolist(),
        [label2id[i] for i in eval_df["intent"]],
        tokenizer,
        max_len,
    )

    batch_size = intent_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=intent_cfg["learning_rate"])
    total_steps = len(train_loader) * intent_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    for epoch in range(intent_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{intent_cfg['epochs']} loss={total_loss / max(1, len(train_loader)):.4f}")

    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pred_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            preds.extend(pred_ids.tolist())
            gold.extend(batch["labels"].cpu().numpy().tolist())

    macro_f1 = f1_score(gold, preds, average="macro", zero_division=0)
    label_ids = list(range(len(labels)))
    report = classification_report(
        gold,
        preds,
        labels=label_ids,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )

    out_dir = resolve_path(config["paths"]["intent_model"], config)
    ensure_dirs(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    metrics = {
        "macro_f1": round(float(macro_f1), 4),
        "num_train": len(train_df),
        "num_eval": len(eval_df),
        "classification_report": report,
    }
    metrics_path = resolve_path(config["paths"]["models"], config) / "intent_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved model -> {out_dir}")
    print(f"Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
