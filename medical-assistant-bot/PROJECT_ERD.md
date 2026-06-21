# Medical Assistant Bot ERD

This project is not backed by a traditional relational database. The diagram below is a **logical ERD** of the main data artifacts used by the pipeline: source files, parsed documents, chunks, intent training examples, vector-store entries, and the saved intent model.

```mermaid
erDiagram
    SOURCE_FILE {
        string file_name PK
        string source_dir
        string file_type
    }

    PARSED_DOCUMENT {
        string document_id PK
        string file_name FK
        string doc_type
        int page_number
        string intent
        string question
        text raw_text
    }

    DOCUMENT_CHUNK {
        string chunk_id PK
        string document_id FK
        string file_name
        int page_number
        string doc_type
        string intent
        string question
        text chunk_text
    }

    QUESTION_INTENT_EXAMPLE {
        string example_id PK
        string question
        string intent FK
        string source_file
    }

    INTENT_LABEL {
        string label PK
        string description
    }

    VECTOR_STORE_ENTRY {
        string vector_id PK
        string chunk_id FK
        string collection_name
        string source_file
        int page_number
        string doc_type
        string intent
        text embedded_text
    }

    INTENT_MODEL_ARTIFACT {
        string model_name PK
        string model_dir
        string tokenizer_dir
        string label_map_path
        string metrics_path
    }

    SOURCE_FILE ||--o{ PARSED_DOCUMENT : yields
    PARSED_DOCUMENT ||--o{ DOCUMENT_CHUNK : splits_into
    DOCUMENT_CHUNK ||--o| QUESTION_INTENT_EXAMPLE : can_generate
    INTENT_LABEL ||--o{ QUESTION_INTENT_EXAMPLE : labels
    DOCUMENT_CHUNK ||--|| VECTOR_STORE_ENTRY : indexed_as
    INTENT_MODEL_ARTIFACT ||--o{ QUESTION_INTENT_EXAMPLE : trained_on
    INTENT_MODEL_ARTIFACT ||--o{ INTENT_LABEL : predicts
```

## Notes

- `SOURCE_FILE` covers the mixed inputs handled by `scripts/ingest_documents.py`, including PDF, TXT, MD, CSV, JSON, and JSONL files.
- `PARSED_DOCUMENT` is the normalized form of a source file or page before chunking.
- `DOCUMENT_CHUNK` is what gets written to `data/processed/chunks.jsonl` and indexed into ChromaDB.
- `QUESTION_INTENT_EXAMPLE` is the deduplicated `question,intent` training data written to `data/processed/intent_train.csv`.
- `INTENT_LABEL` corresponds to the labels configured in `config.yaml`: `general_info`, `symptoms`, `medication`, `prevention`, `emergency`, and `unknown`.
- `VECTOR_STORE_ENTRY` represents the ChromaDB record created from each chunk, with metadata such as `source_file`, `page`, `doc_type`, and `intent`.
- `INTENT_MODEL_ARTIFACT` represents the saved DistilBERT classifier under `models/intent_classifier/` plus the generated metrics file.
- Streamlit chat session state (`messages` and `sources`) is runtime-only, so it is intentionally not shown here.
