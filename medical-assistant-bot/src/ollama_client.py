"""Ollama API client for chat and embeddings."""

from __future__ import annotations

from typing import Any

import httpx


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        chat_model: str = "hermes",
        embed_model: str = "nomic-embed-text",
        timeout_seconds: int = 600,
        chat_options: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout_seconds
        self.chat_options = chat_options or {}
        self._timeout = httpx.Timeout(
            connect=15.0,
            read=float(timeout_seconds),
            write=60.0,
            pool=15.0,
        )

    def health_check(self) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if not models:
                    return False, "Ollama is running but no models are installed."
                chat_ok = any(self.chat_model in m for m in models)
                if not chat_ok:
                    return False, f"Model '{self.chat_model}' not found. Installed: {', '.join(models)}"
                return True, f"Connected. Chat model: {self.chat_model}"
        except Exception as exc:
            return False, f"Cannot reach Ollama at {self.base_url}: {exc}"

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        options = dict(self.chat_options)
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + messages

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "input": text}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(f"{self.base_url}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings") or data.get("embedding")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (int, float)):
                return embeddings  # type: ignore[return-value]
            raise ValueError("Unexpected embedding response from Ollama")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
