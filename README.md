<div align="center">

<!-- Typing animation header -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=0E7490&center=true&vCenter=true&width=600&lines=Medical+Assistant+Bot;NLP+Theory+%7C+Domain+D2;RAG+%2B+BERT+Intent+%2B+Ollama" alt="Typing SVG" />
</a>

<br />

<!-- Animated badge -->
<img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge&logo=statuspage&logoColor=white" alt="Status" />
<img src="https://img.shields.io/badge/Domain-D2_Medical-0E7490?style=for-the-badge" alt="Domain D2" />
<img src="https://img.shields.io/badge/Course-NLP_Theory-6366F1?style=for-the-badge" alt="NLP Theory" />
<img src="https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge" alt="License" />

<br /><br />

```
  __  __          _ _            _   _       _           _   
 |  \/  | ___  __| (_) ___ _ __ | |_(_) __ _| |__   ___ | |_ 
 | |\/| |/ _ \/ _` | |/ _ \ '_ \| __| |/ _` | '_ \ / _ \| __|
 | |  | |  __/ (_| | |  __/ | | | |_| | (_| | |_) | (_) | |_ 
 |_|  |_|\___|\__,_|_|\___|_| |_|\__|_|\__,_|_.__/ \___/ \__|
         Assistant Bot  ·  RAG  ·  DistilBERT  ·  Hermes
```

<sub>Web-based medical information chatbot — general health Q&amp;A with retrieval grounding, intent routing, and safety guardrails.</sub>

</div>

---

## Overview

This repository hosts an **academic NLP project** implementing a **Medical Assistant Bot** (Problem **D2**, Domain **D**). The system combines:

| Layer | Technology | Role |
|-------|------------|------|
| **UI** | [Streamlit](https://streamlit.io/) | Multi-turn chat, file upload, pipeline controls |
| **Intent** | [DistilBERT](https://huggingface.co/distilbert-base-uncased) | Route queries by medical intent |
| **Retrieval** | [ChromaDB](https://www.trychroma.com/) + embeddings | Ground answers in ingested documents |
| **Generation** | [Ollama](https://ollama.com/) (`hermes3`) | Transformer LLM for natural-language replies |

> **Disclaimer:** This bot provides **general medical information only**. It does **not** diagnose, prescribe, or replace professional healthcare. Emergency keywords trigger an immediate escalation message.

---

## Architecture

```mermaid
flowchart TB
    subgraph UI["Streamlit Chat UI"]
        U[User Query]
        R[Answer + Source Citations]
    end

    subgraph Safety["Safety Layer"]
        E{Emergency<br/>keywords?}
        EM[Escalation Message]
    end

    subgraph Intent["Intent Routing"]
        BERT[DistilBERT Classifier]
        KW[Keyword Fallback]
        L[Intent Label]
    end

    subgraph RAG["Retrieval-Augmented Generation"]
        CH[(ChromaDB<br/>medical_knowledge)]
        EMB[Embeddings<br/>MiniLM / nomic-embed]
        CTX[Context Assembly]
    end

    subgraph LLM["Ollama LLM"]
        H[Hermes3 Chat Model]
    end

    U --> E
    E -->|yes| EM --> R
    E -->|no| BERT
    BERT -->|conf &lt; 0.4| KW
    BERT --> L
    KW --> L
    L --> CH
    U --> CH
    CH --> EMB
    EMB --> CTX
    CTX --> H
    L --> H
    H --> R
```

<details>
<summary><b>Data pipeline (offline)</b></summary>

```mermaid
flowchart LR
    RAW[PDF · TXT · MD · CSV · JSON] --> ING[ingest_documents.py]
    ING --> CHK[chunks.jsonl]
    ING --> TRN[intent_train.csv]
    CHK --> IDX[build_index.py]
    IDX --> VDB[(ChromaDB Index)]
    TRN --> TRC[train_intent_classifier.py]
    TRC --> MDL[DistilBERT Model]
```

</details>

---

## Features

| | Feature | Description |
|---|---------|-------------|
| 📄 | **Mixed document ingestion** | PDF, TXT, MD, CSV, JSON/JSONL with OCR/noisy-text cleaning |
| 🔍 | **Vector retrieval** | ChromaDB with `sentence-transformers` (default) or Ollama embeddings |
| 🧠 | **Intent classification** | 6 labels: `general_info`, `symptoms`, `medication`, `prevention`, `emergency`, `unknown` |
| 💬 | **Grounded generation** | Ollama Hermes3 with RAG context and configurable temperature |
| 🛡️ | **Safety guardrails** | Medical disclaimer, emergency detection, low-confidence fallback |
| 📎 | **Source citations** | Expandable source chunks with similarity scores in the UI |
| ⚡ | **Fast mode** | Smaller corpus + batch embeddings for quicker CPU iteration |

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Transformers" />
  <img src="https://img.shields.io/badge/ChromaDB-FF6F61?style=flat-square" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat-square" alt="Ollama" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
</p>

---

## Project Structure

```
NLP THEORY CPC/
├── README.md                          ← you are here
├── requirements.txt                   ← shared Python dependencies
└── medical-assistant-bot/
    ├── app.py                         # Streamlit chat UI
    ├── config.yaml                    # Paths, Ollama, RAG, intent settings
    ├── data/
    │   ├── raw/                       # Your uploaded documents
    │   ├── sample/                    # Demo FAQ + articles
    │   ├── Datafile/                  # Tabular medical datasets
    │   └── processed/                 # chunks.jsonl, Chroma index
    ├── scripts/
    │   ├── ingest_documents.py
    │   ├── build_index.py
    │   └── train_intent_classifier.py
    ├── src/                           # RAG, Ollama client, intent, safety
    └── models/intent_classifier/      # Saved DistilBERT weights
```

---

## Quick Start

### 1. Prerequisites — Ollama (external or local)

```bash
ollama pull hermes3
ollama pull nomic-embed-text
ollama serve
```

Edit `medical-assistant-bot/config.yaml` and set `ollama.base_url` to your Ollama host (e.g. `http://192.168.1.10:11434`).

### 2. Python environment

```bash
cd medical-assistant-bot
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r ../requirements.txt
```

Place medical files in `data/raw/` (sample data is in `data/sample/`).

### 3. Run the pipeline

```bash
python scripts/ingest_documents.py
python scripts/build_index.py
python scripts/train_intent_classifier.py
streamlit run app.py
```

### Fast mode (CPU-friendly iteration)

Uses a smaller chatbot subset and batch `sentence-transformers` embeddings (~5–15 min index build vs 1–2 hours for the full corpus). Ollama is still used for chat at query time.

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

Configure limits in `config.yaml`: `ingest.fast_max_chatbot_rows` (default `5000`) and `rag.embed_provider` (default `sentence-transformers`).

---

## Configuration Highlights

| Setting | Default | Purpose |
|---------|---------|---------|
| `ollama.chat_model` | `hermes3` | Primary LLM for chat |
| `ollama.fast_chat_model` | `qwen2.5:3b` | Faster alternative on CPU |
| `ollama.embed_model` | `nomic-embed-text` | Ollama embedding model |
| `rag.embed_provider` | `sentence-transformers` | Index-time embedding backend |
| `rag.embed_model` | `all-MiniLM-L6-v2` | Local embedding model |
| `rag.top_k` | `4` | Retrieved chunks per query |
| `rag.min_similarity` | `0.35` | Similarity threshold |
| `intent.model_name` | `distilbert-base-uncased` | Intent classifier backbone |

---

## Evaluation

| Metric | Value |
|--------|-------|
| Intent macro F1 | **0.78** |
| Intent accuracy | **0.97** |
| Training examples | 2,691 |
| Eval examples | 673 |

Emergency intent also uses **regex-based keyword detection** in the safety layer for immediate escalation regardless of classifier confidence.

---

## Assignment Alignment

| Requirement | Implementation |
|-------------|----------------|
| Conversational AI | Streamlit multi-turn chat with history |
| BERT | DistilBERT intent classifier (`distilbert-base-uncased`) |
| Transformer LLM | Ollama Hermes3 |
| Noisy text handling | PDF/OCR cleaning during ingest |
| Evaluation | Intent F1 metrics + RAG grounding via source citations |

---

## Troubleshooting

<details>
<summary><b>ImportError: <code>DEFAULT_EXCLUDED_CONTENT_TYPES</code> from <code>starlette.middleware.gzip</code></b></summary>

Newer Streamlit expects Starlette ≥ 0.41. Upgrade both packages inside your venv:

```powershell
pip install "streamlit>=1.40.0" "starlette>=0.41.0" --upgrade
```

Verify:

```powershell
streamlit --version
python -c "import streamlit"
```

</details>

<details>
<summary><b>First reply is very slow on CPU</b></summary>

The default `hermes3` model can take 2–5 minutes to load on first query. Switch to `qwen2.5:3b` via the Streamlit sidebar or set `ollama.fast_chat_model` in `config.yaml`.

</details>

---

<div align="center">

**Built for NLP Theory — Domain D2 Medical Assistant Bot**

<br />

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&duration=4000&pause=1500&color=64748B&center=true&vCenter=true&width=500&lines=Always+consult+a+healthcare+professional.;Not+a+substitute+for+medical+advice." alt="Disclaimer typing" />

</div>


## Suggested GitHub Topics

~~~text
python machine-learning ai
~~~

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
