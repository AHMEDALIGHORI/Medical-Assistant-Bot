"""Ingest mixed medical documents into unified chunks.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import (  # noqa: E402
    chunk_text,
    clean_medical_text,
    ensure_dirs,
    load_config,
    project_root,
    resolve_path,
    save_jsonl,
)


def infer_intent(question: str) -> str:
    lowered = question.lower()
    if any(k in lowered for k in ("emergency", "heart attack", "can't breathe", "unconscious", "suicide", "chest pain")):
        return "emergency"
    if any(k in lowered for k in ("medicine", "drug", "dose", "tablet", "prescri", "antibiotic", "mg")):
        return "medication"
    if any(k in lowered for k in ("symptom", "fever", "pain", "cough", "rash", "ache", "vomit", "bleed")):
        return "symptoms"
    if any(k in lowered for k in ("prevent", "avoid", "vaccine", "hygiene", "diet", "exercise")):
        return "prevention"
    return "general_info"


def parse_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    rows: list[dict] = []
    reader = PdfReader(str(path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_medical_text(text)
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "source_file": path.name,
                "page": page_num,
                "doc_type": "pdf",
            }
        )
    return rows


def parse_text_file(path: Path) -> list[dict]:
    text = clean_medical_text(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        return []
    return [
        {
            "text": text,
            "source_file": path.name,
            "page": None,
            "doc_type": path.suffix.lstrip("."),
        }
    ]


def parse_qa_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    q_col = cols.get("question")
    a_col = cols.get("answer")
    intent_col = cols.get("intent")
    if not q_col or not a_col:
        raise ValueError(f"{path.name}: CSV must have question and answer columns")

    rows: list[dict] = []
    for _, row in df.iterrows():
        question = clean_medical_text(str(row[q_col]))
        answer = clean_medical_text(str(row[a_col]))
        if not question or not answer or question == "nan":
            continue
        intent = str(row[intent_col]).strip() if intent_col else infer_intent(question)
        text = f"Question: {question}\nAnswer: {answer}"
        rows.append(
            {
                "text": text,
                "source_file": path.name,
                "page": None,
                "doc_type": "csv",
                "intent": intent if intent and intent != "nan" else infer_intent(question),
                "question": question,
            }
        )
    return rows


def parse_medical_chatbot_csv(path: Path, max_rows: int | None) -> list[dict]:
    df = pd.read_csv(path, nrows=max_rows)
    rows: list[dict] = []
    for _, row in df.iterrows():
        question = clean_medical_text(str(row.get("Description", "")))
        answer = clean_medical_text(str(row.get("Doctor", "")))
        patient = clean_medical_text(str(row.get("Patient", "")))
        if not question or not answer or question == "nan" or answer == "nan":
            continue
        if len(answer) < 20:
            continue
        question = re.sub(r"^Q\.\s*", "", question)
        context = f"\nPatient context: {patient}" if patient and patient != "nan" else ""
        text = f"Question: {question}{context}\nAnswer: {answer}"
        intent = infer_intent(question)
        rows.append(
            {
                "text": text,
                "source_file": path.name,
                "page": None,
                "doc_type": "medical_chatbot",
                "intent": intent,
                "question": question,
            }
        )
    return rows


def parse_symptom_severity_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    lines = []
    for _, row in df.iterrows():
        symptom = str(row.iloc[0]).replace("_", " ")
        weight = row.iloc[1]
        lines.append(f"Symptom '{symptom}' has clinical severity weight {weight} (higher = more severe).")
    text = "Medical symptom severity reference:\n" + "\n".join(lines)
    return [
        {
            "text": text,
            "source_file": path.name,
            "page": None,
            "doc_type": "symptom_reference",
            "intent": "symptoms",
            "question": None,
        }
    ]


def parse_tabular_csv(path: Path, rows_per_chunk: int = 25) -> list[dict]:
    df = pd.read_csv(path)
    dataset_name = path.stem.replace("_", " ")
    rows: list[dict] = []
    for start in range(0, len(df), rows_per_chunk):
        batch = df.iloc[start : start + rows_per_chunk]
        lines = [f"Dataset: {dataset_name}. Columns: {', '.join(df.columns)}."]
        for idx, row in batch.iterrows():
            parts = [f"{col}={row[col]}" for col in df.columns]
            lines.append(f"Record {idx}: " + ", ".join(parts))
        text = "\n".join(lines)
        rows.append(
            {
                "text": text,
                "source_file": path.name,
                "page": start // rows_per_chunk + 1,
                "doc_type": "tabular",
                "intent": "general_info",
                "question": None,
            }
        )
    return rows


def parse_csv(path: Path, config: dict) -> list[dict]:
    name = path.name.lower()
    ingest_cfg = config.get("ingest", {})
    max_chatbot = ingest_cfg.get("max_chatbot_rows")
    tabular_chunk = ingest_cfg.get("tabular_rows_per_chunk", 25)

    if name == "ai-medical-chatbot.csv":
        return parse_medical_chatbot_csv(path, max_chatbot)
    if name == "symptom-severity.csv":
        return parse_symptom_severity_csv(path)
    if name in {
        "diabetes.csv",
        "heart.csv",
        "heart_disease_uci.csv",
        "cardio_train.csv",
        "drug200.csv",
        "thyroid_diff.csv",
        "mental-heath-in-tech-2016_20161114.csv",
    }:
        return parse_tabular_csv(path, tabular_chunk)

    cols = {c.lower() for c in pd.read_csv(path, nrows=0).columns}
    if "question" in cols and "answer" in cols:
        return parse_qa_csv(path)
    return parse_tabular_csv(path, tabular_chunk)


def parse_json_records(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        data = json.loads(content)
        records = data if isinstance(data, list) else data.get("items", [data])

    rows: list[dict] = []
    for rec in records:
        question = clean_medical_text(str(rec.get("question", "")))
        answer = clean_medical_text(str(rec.get("answer", rec.get("text", ""))))
        if question and answer:
            text = f"Question: {question}\nAnswer: {answer}"
        elif answer:
            text = answer
        else:
            continue
        rows.append(
            {
                "text": text,
                "source_file": path.name,
                "page": rec.get("page"),
                "doc_type": path.suffix.lstrip("."),
                "intent": rec.get("intent") or (infer_intent(question) if question else "general_info"),
                "question": question or None,
            }
        )
    return rows


def document_to_chunks(doc: dict, chunk_size: int, overlap: int, doc_index: int = 0) -> list[dict]:
    pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
    chunks: list[dict] = []
    for i, piece in enumerate(pieces):
        chunks.append(
            {
                "chunk_id": f"{doc['source_file']}::{doc.get('page', 'na')}::{doc_index}::{i}",
                "text": piece,
                "source_file": doc["source_file"],
                "page": doc.get("page"),
                "doc_type": doc.get("doc_type"),
                "intent": doc.get("intent"),
                "question": doc.get("question"),
            }
        )
    return chunks


def collect_source_dirs(config: dict) -> list[Path]:
    dirs = [
        resolve_path(config["paths"].get("data_datafile", "Datafile"), config),
        resolve_path(config["paths"]["data_raw"], config),
        resolve_path(config["paths"]["data_sample"], config),
    ]
    return [d for d in dirs if d.exists()]


def ingest(config: dict) -> list[dict]:
    chunk_size = config["rag"]["chunk_size"]
    overlap = config["rag"]["chunk_overlap"]
    all_chunks: list[dict] = []
    doc_counter = 0

    for source_dir in collect_source_dirs(config):
        print(f"Scanning {source_dir}...")
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".pdf":
                    docs = parse_pdf(path)
                elif suffix in {".txt", ".md"}:
                    docs = parse_text_file(path)
                elif suffix == ".csv":
                    docs = parse_csv(path, config)
                elif suffix in {".json", ".jsonl"}:
                    docs = parse_json_records(path)
                else:
                    continue
                for doc in docs:
                    all_chunks.extend(document_to_chunks(doc, chunk_size, overlap, doc_counter))
                    doc_counter += 1
                print(f"  + {path.name}: {len(docs)} doc(s)")
            except Exception as exc:
                print(f"  ! Skipped {path.name}: {exc}")

    return all_chunks


def build_intent_train_csv(chunks: list[dict], out_path: Path) -> None:
    rows: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        question = chunk.get("question")
        intent = chunk.get("intent")
        if not question or not intent:
            continue
        key = (question.lower()[:200], intent)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"question": question, "intent": intent})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("  ! No labeled questions for intent_train.csv")
        return
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "intent"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  + intent_train.csv: {len(rows)} labeled questions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest medical documents")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Limit ai-medical-chatbot.csv rows for quick pipeline runs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fast:
        ingest_cfg = dict(config.get("ingest", {}))
        fast_max = ingest_cfg.get("fast_max_chatbot_rows", 5000)
        ingest_cfg["max_chatbot_rows"] = fast_max
        config = dict(config)
        config["ingest"] = ingest_cfg
        print(f"Fast mode: limiting ai-medical-chatbot.csv to {fast_max} rows")
    processed = resolve_path(config["paths"]["data_processed"], config)
    ensure_dirs(processed)

    print("Ingesting documents...")
    chunks = ingest(config)
    chunks_path = processed / "chunks.jsonl"
    save_jsonl(chunks, chunks_path)
    print(f"Wrote {len(chunks)} chunks -> {chunks_path}")

    build_intent_train_csv(chunks, processed / "intent_train.csv")


if __name__ == "__main__":
    main()
