"""Build ChromaDB vector index from chunks.jsonl."""



from __future__ import annotations



import argparse

import os

import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))



from utils import load_config, load_jsonl, project_root, resolve_path  # noqa: E402

from src.ollama_client import OllamaClient  # noqa: E402

from src.rag import EmbeddingProvider, MedicalRAG  # noqa: E402





def main() -> None:

    parser = argparse.ArgumentParser(description="Build ChromaDB index")

    parser.add_argument("--config", default=None)

    parser.add_argument(

        "--fast",

        action="store_true",

        help="Fast build: sentence-transformers batch embeddings with progress ETA",

    )

    args = parser.parse_args()



    config = load_config(args.config)

    models_dir = config.get("ollama", {}).get("models_dir")

    if models_dir:

        os.environ["OLLAMA_MODELS"] = str(models_dir).replace("/", os.sep)



    processed = resolve_path(config["paths"]["data_processed"], config)

    chunks_path = processed / "chunks.jsonl"

    chunks = load_jsonl(chunks_path)

    if not chunks:

        print(f"No chunks found at {chunks_path}. Run ingest_documents.py first.")

        return



    rag_cfg = config.get("rag", {})

    embed_provider = EmbeddingProvider.PROVIDER_ST if args.fast else rag_cfg.get(

        "embed_provider", EmbeddingProvider.PROVIDER_ST

    )

    embed_model = rag_cfg.get("embed_model", "all-MiniLM-L6-v2")



    ollama_cfg = config["ollama"]

    ollama = OllamaClient(

        base_url=ollama_cfg["base_url"],

        chat_model=ollama_cfg["chat_model"],

        embed_model=ollama_cfg["embed_model"],

        timeout_seconds=ollama_cfg.get("timeout_seconds", 120),

    )



    if embed_provider == EmbeddingProvider.PROVIDER_OLLAMA:

        ok, msg = ollama.health_check()

        if ok:

            print(f"Ollama embeddings: {msg}")

        else:

            print(f"Ollama unavailable ({msg}). Falling back to sentence-transformers.")

            embed_provider = EmbeddingProvider.PROVIDER_ST

    elif args.fast:

        print(f"Fast mode: sentence-transformers ({embed_model}), batch embeddings")

    else:

        print(f"Embeddings: {embed_provider} ({embed_model})")



    rag = MedicalRAG(

        chroma_path=resolve_path(config["paths"]["chroma"], config),

        collection_name=rag_cfg["collection_name"],

        ollama=ollama,

        embed_provider=embed_provider,

        embed_model=embed_model,

        top_k=rag_cfg["top_k"],

        min_similarity=rag_cfg["min_similarity"],

    )



    batch_size = 128 if args.fast or embed_provider == EmbeddingProvider.PROVIDER_ST else 32

    progress_interval = 5 if args.fast else None

    show_progress = args.fast and embed_provider == EmbeddingProvider.PROVIDER_ST



    print(f"Indexing {len(chunks)} chunks (batch_size={batch_size}, provider={embed_provider})...")

    count = rag.index_chunks(

        chunks,

        batch_size=batch_size,

        show_embed_progress=show_progress,

        progress_interval=progress_interval,

    )

    print(f"Indexed {count} chunks into {config['paths']['chroma']}")





if __name__ == "__main__":

    main()


