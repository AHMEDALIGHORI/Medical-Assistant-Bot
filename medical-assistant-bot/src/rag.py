"""RAG retrieval using ChromaDB."""



from __future__ import annotations



import time

from pathlib import Path

from typing import Any, Callable



import chromadb

from chromadb.config import Settings



from src.ollama_client import OllamaClient





class EmbeddingProvider:

    PROVIDER_ST = "sentence-transformers"

    PROVIDER_OLLAMA = "ollama"



    def __init__(

        self,

        provider: str = PROVIDER_ST,

        ollama: OllamaClient | None = None,

        model_name: str = "all-MiniLM-L6-v2",

    ) -> None:

        self.provider = provider

        self.ollama = ollama

        self.model_name = model_name

        self._st_model = None



    @property

    def uses_sentence_transformers(self) -> bool:

        return self.provider == self.PROVIDER_ST



    def _get_st_model(self):

        if self._st_model is None:

            from sentence_transformers import SentenceTransformer



            self._st_model = SentenceTransformer(self.model_name)

        return self._st_model



    def embed(self, texts: list[str], *, show_progress: bool = False) -> list[list[float]]:

        if not texts:

            return []

        if self.provider == self.PROVIDER_OLLAMA:

            if not self.ollama:

                raise RuntimeError("Ollama embed provider selected but no Ollama client configured.")

            return self.ollama.embed_batch(texts)

        model = self._get_st_model()

        vectors = model.encode(texts, show_progress_bar=show_progress)

        return [v.tolist() for v in vectors]



    def embed_query(self, text: str) -> list[float]:

        return self.embed([text], show_progress=False)[0]





class MedicalRAG:

    def __init__(

        self,

        chroma_path: str | Path,

        collection_name: str = "medical_knowledge",

        ollama: OllamaClient | None = None,

        embed_provider: str = EmbeddingProvider.PROVIDER_ST,

        embed_model: str = "all-MiniLM-L6-v2",

        top_k: int = 4,

        min_similarity: float = 0.35,

    ) -> None:

        self.chroma_path = Path(chroma_path)

        self.collection_name = collection_name

        self.top_k = top_k

        self.min_similarity = min_similarity

        self.embedder = EmbeddingProvider(

            provider=embed_provider,

            ollama=ollama if embed_provider == EmbeddingProvider.PROVIDER_OLLAMA else None,

            model_name=embed_model,

        )

        self._client: chromadb.ClientAPI | None = None

        self._collection = None



    @property

    def client(self) -> chromadb.ClientAPI:

        if self._client is None:

            self.chroma_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(

                path=str(self.chroma_path),

                settings=Settings(anonymized_telemetry=False),

            )

        return self._client



    @property

    def collection(self):

        if self._collection is None:

            self._collection = self.client.get_or_create_collection(

                name=self.collection_name,

                metadata={"hnsw:space": "cosine"},

            )

        return self._collection



    def index_chunks(

        self,

        chunks: list[dict[str, Any]],

        batch_size: int = 32,

        *,

        show_embed_progress: bool = False,

        progress_interval: int | None = None,

        on_progress: Callable[[int, int, float], None] | None = None,

    ) -> int:

        if not chunks:

            return 0



        try:

            self.client.delete_collection(self.collection_name)

        except Exception:

            pass

        self._collection = self.client.get_or_create_collection(

            name=self.collection_name,

            metadata={"hnsw:space": "cosine"},

        )



        total_chunks = len(chunks)

        total = 0

        started = time.perf_counter()



        precomputed: list[list[float]] | None = None

        if show_embed_progress and self.embedder.uses_sentence_transformers:

            all_documents = [c["text"] for c in chunks]

            print(f"Encoding {total_chunks} chunks with sentence-transformers...")

            precomputed = self.embedder.embed(all_documents, show_progress=True)

            print(f"Embedding done in {time.perf_counter() - started:.1f}s. Writing to ChromaDB...")



        for batch_idx, i in enumerate(range(0, total_chunks, batch_size)):

            batch = chunks[i : i + batch_size]

            ids = [c["chunk_id"] for c in batch]

            documents = [c["text"] for c in batch]

            metadatas = [

                {

                    "source_file": c.get("source_file", ""),

                    "page": str(c.get("page") or ""),

                    "doc_type": c.get("doc_type", ""),

                    "intent": c.get("intent") or "",

                }

                for c in batch

            ]

            if precomputed is not None:

                embeddings = precomputed[i : i + batch_size]

            else:

                embeddings = self.embedder.embed(documents, show_progress=False)

            self.collection.add(

                ids=ids,

                documents=documents,

                metadatas=metadatas,

                embeddings=embeddings,

            )

            total += len(batch)



            if on_progress is not None:

                elapsed = time.perf_counter() - started

                on_progress(total, total_chunks, elapsed)

            elif progress_interval and (batch_idx + 1) % progress_interval == 0:

                elapsed = time.perf_counter() - started

                rate = total / elapsed if elapsed > 0 else 0.0

                remaining = (total_chunks - total) / rate if rate > 0 else 0.0

                pct = 100.0 * total / total_chunks

                print(

                    f"  [{total}/{total_chunks}] {pct:.1f}% — "

                    f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining "

                    f"({rate:.1f} chunks/s)"

                )



        if progress_interval and total > 0:

            elapsed = time.perf_counter() - started

            print(f"  [{total}/{total_chunks}] 100.0% — {elapsed:.0f}s total")



        return total



    def retrieve(self, query: str) -> list[dict[str, Any]]:

        if self.collection.count() == 0:

            return []



        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(

            query_embeddings=[query_embedding],

            n_results=self.top_k,

            include=["documents", "metadatas", "distances"],

        )



        hits: list[dict[str, Any]] = []

        docs = results.get("documents", [[]])[0]

        metas = results.get("metadatas", [[]])[0]

        dists = results.get("distances", [[]])[0]



        for doc, meta, dist in zip(docs, metas, dists):

            similarity = 1.0 - float(dist)

            if similarity < self.min_similarity:

                continue

            hits.append(

                {

                    "text": doc,

                    "source_file": meta.get("source_file", ""),

                    "page": meta.get("page") or None,

                    "doc_type": meta.get("doc_type", ""),

                    "intent": meta.get("intent") or None,

                    "similarity": round(similarity, 4),

                }

            )

        return hits



    def format_context(self, hits: list[dict[str, Any]]) -> str:

        if not hits:

            return "No relevant context found."

        parts = []

        for i, hit in enumerate(hits, 1):

            src = hit.get("source_file", "unknown")

            page = hit.get("page")

            loc = f"{src}" + (f" (page {page})" if page else "")

            snippet = hit["text"]
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "..."
            parts.append(f"[{i}] Source: {loc}\n{snippet}")

        return "\n\n".join(parts)



    def has_index(self) -> bool:

        try:

            return self.collection.count() > 0

        except Exception:

            return False


