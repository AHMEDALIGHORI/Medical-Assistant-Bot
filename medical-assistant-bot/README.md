# Medical Assistant Bot (Domain D2)

> See the [animated project README](../README.md) at the repo root for architecture diagrams, badges, and full documentation.

Web-based medical information chatbot using **RAG**, **DistilBERT intent classification**, and **Ollama Hermes3** via Streamlit.

## Features

- Mixed document ingestion (PDF, TXT, MD, CSV, JSON/JSONL)
- ChromaDB vector retrieval with Ollama or sentence-transformer embeddings
- BERT-based intent routing (symptoms, medication, emergency, etc.)
- Streamlit chat UI with source citations and safety disclaimers

## Prerequisites (external Ollama machine)

```bash
ollama pull hermes3
ollama pull nomic-embed-text
ollama serve
```

Edit `config.yaml` and set `ollama.base_url` to your Ollama host (e.g. `http://192.168.1.10:11434`).

## Setup

```bash
cd medical-assistant-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r ../requirements.txt
```

Place your medical files in `data/raw/` (sample data is in `data/sample/`).

## Pipeline

```bash
python scripts/ingest_documents.py
python scripts/build_index.py
python scripts/train_intent_classifier.py
streamlit run app.py
```

### Fast mode

For quicker iteration on CPU (~5–15 min index build vs 1–2 hours for the full corpus), use a smaller chatbot subset and batch `sentence-transformers` embeddings. Ollama is still used for chat at query time.

```bash
python scripts/ingest_documents.py --fast
python scripts/build_index.py --fast
python scripts/train_intent_classifier.py
streamlit run app.py
```

Or run the full fast pipeline in one step (PowerShell):

```powershell
.\run_pipeline_fast.ps1
```

Configure limits in `config.yaml`: `ingest.fast_max_chatbot_rows` (default 5000) and `rag.embed_provider` (default `sentence-transformers`).

## Project structure

```
medical-assistant-bot/
├── app.py                 # Streamlit UI
├── config.yaml
├── data/raw/              # Your documents
├── data/sample/           # Demo FAQ + articles
├── scripts/               # Ingest, index, train
├── src/                   # RAG, Ollama, intent, safety
└── models/                # Saved intent classifier
```

## Safety

This bot provides **general medical information only**. It does **not** diagnose, prescribe, or replace professional healthcare. Emergency keywords trigger an immediate escalation message.

## Assignment alignment

| Requirement | Implementation |
|-------------|----------------|
| Conversational AI | Streamlit multi-turn chat |
| BERT | DistilBERT intent classifier |
| Transformer LLM | Ollama Hermes3 |
| Noisy text | PDF/OCR cleaning in ingest |
| Evaluation | Intent F1 + RAG grounding checks |

## Troubleshooting

### ImportError: cannot import name 'DEFAULT_EXCLUDED_CONTENT_TYPES' from 'starlette.middleware.gzip'

Newer Streamlit expects Starlette 0.41+. An older Starlette (often pulled in with FastAPI) breaks `streamlit run`.

Activate your venv, then upgrade both packages:

```powershell
pip install "streamlit>=1.40.0" "starlette>=0.41.0" --upgrade
```

Confirm:

```powershell
streamlit --version
python -c "import streamlit"
```

Install project deps from the repo root: `pip install -r ../requirements.txt`.
