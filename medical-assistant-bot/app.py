"""Streamlit medical assistant chatbot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from utils import load_config, project_root, resolve_path  # noqa: E402
from src.intent_classifier import IntentClassifier, keyword_intent_fallback  # noqa: E402
from src.ollama_client import OllamaClient  # noqa: E402
from src.rag import MedicalRAG  # noqa: E402
from src.safety import (  # noqa: E402
    EMERGENCY_MESSAGE,
    LOW_CONFIDENCE_MESSAGE,
    MEDICAL_DISCLAIMER,
    build_system_prompt,
    is_emergency,
)


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []


@st.cache_resource
def load_rag(base_url: str, chat_model: str, embed_model: str, config: dict) -> MedicalRAG:
    rag_cfg = config.get("rag", {})
    embed_provider = rag_cfg.get("embed_provider", "sentence-transformers")
    ollama = OllamaClient(
        base_url=base_url,
        chat_model=chat_model,
        embed_model=embed_model,
        timeout_seconds=config["ollama"].get("timeout_seconds", 120),
    )
    return MedicalRAG(
        chroma_path=resolve_path(config["paths"]["chroma"], config),
        collection_name=rag_cfg["collection_name"],
        ollama=ollama,
        embed_provider=embed_provider,
        embed_model=rag_cfg.get("embed_model", "all-MiniLM-L6-v2"),
        top_k=rag_cfg["top_k"],
        min_similarity=rag_cfg["min_similarity"],
    )


@st.cache_resource
def load_intent_model(model_dir: str) -> IntentClassifier | None:
    return IntentClassifier.load_if_exists(model_dir)


def run_script(script_name: str, config: dict | None = None) -> tuple[bool, str]:
    script = ROOT / "scripts" / script_name
    env = os.environ.copy()
    models_dir = (config or {}).get("ollama", {}).get("models_dir")
    if models_dir:
        env["OLLAMA_MODELS"] = str(models_dir).replace("/", os.sep)
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output.strip() or "Done."
    except Exception as exc:
        return False, str(exc)


def generate_response(
    user_text: str,
    rag: MedicalRAG,
    ollama: OllamaClient,
    intent_model: IntentClassifier | None,
    config: dict,
    history: list[dict],
) -> tuple[str, list[dict], str]:
    if is_emergency(user_text):
        return EMERGENCY_MESSAGE, [], "emergency"

    labels = config["intent"]["labels"]
    if intent_model:
        intent, conf = intent_model.predict(user_text)
        if conf < 0.4:
            intent = keyword_intent_fallback(user_text, labels)
    else:
        intent = keyword_intent_fallback(user_text, labels)

    hits = rag.retrieve(user_text)
    if not hits:
        return LOW_CONFIDENCE_MESSAGE, [], intent

    context = rag.format_context(hits)
    max_chars = config.get("ollama", {}).get("max_context_chars", 6000)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[context truncated for faster response]"
    system = build_system_prompt(intent, context)
    chat_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    chat_messages.append({"role": "user", "content": user_text})

    try:
        answer = ollama.chat(chat_messages, system=system)
    except Exception as exc:
        answer = f"Could not reach Ollama: {exc}"

    if not answer:
        answer = LOW_CONFIDENCE_MESSAGE

    return answer, hits, intent


def main() -> None:
    st.set_page_config(page_title="Medical Assistant", page_icon="🩺", layout="wide")
    init_session()

    config = load_config()
    st.title("Medical Assistant Bot")
    st.caption("Domain D2 — RAG + BERT Intent + Ollama Hermes")
    st.info(MEDICAL_DISCLAIMER)

    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("Ollama URL", value=config["ollama"]["base_url"])
        chat_model = st.text_input("Chat model", value=config["ollama"]["chat_model"])
        embed_model = st.text_input("Embed model", value=config["ollama"]["embed_model"])

        ollama_cfg = config["ollama"]
        ollama = OllamaClient(
            base_url=base_url,
            chat_model=chat_model,
            embed_model=embed_model,
            timeout_seconds=ollama_cfg.get("timeout_seconds", 600),
            chat_options=ollama_cfg.get("chat_options"),
        )
        ok, status = ollama.health_check()
        if ok:
            st.success(status)
            fast_model = ollama_cfg.get("fast_chat_model", "qwen2.5:3b")
            st.caption(f"First reply on CPU can take 2–5 min while `{chat_model}` loads. For speed, try `{fast_model}`.")
        else:
            st.error(status)

        st.divider()
        st.subheader("Knowledge base")
        uploaded = st.file_uploader(
            "Upload medical files",
            type=["pdf", "txt", "md", "csv", "json", "jsonl"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Save uploads to data/raw"):
            raw_dir = resolve_path(config["paths"]["data_raw"], config)
            raw_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                (raw_dir / f.name).write_bytes(f.getvalue())
            st.success(f"Saved {len(uploaded)} file(s) to data/raw/")

        if st.button("1. Ingest documents"):
            success, out = run_script("ingest_documents.py", config)
            st.code(out)
            if success:
                st.cache_resource.clear()

        if st.button("2. Build vector index"):
            success, out = run_script("build_index.py", config)
            st.code(out)
            if success:
                st.cache_resource.clear()

        if st.button("3. Train intent model"):
            success, out = run_script("train_intent_classifier.py", config)
            st.code(out)
            if success:
                st.cache_resource.clear()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.sources = []
            st.rerun()

    rag = load_rag(base_url, chat_model, embed_model, config)
    intent_model = load_intent_model(str(resolve_path(config["paths"]["intent_model"], config)))

    if not rag.has_index():
        st.warning(
            "No vector index found. Use the sidebar to run **Ingest documents** then **Build vector index**, "
            "or run the scripts from the terminal."
        )

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i < len(st.session_state.sources):
                srcs = st.session_state.sources[i // 2] if i // 2 < len(st.session_state.sources) else []
                if srcs:
                    with st.expander("Sources"):
                        for s in srcs:
                            loc = s.get("source_file", "unknown")
                            if s.get("page"):
                                loc += f" (page {s['page']})"
                            st.markdown(f"**{loc}** — similarity {s.get('similarity', 'n/a')}")
                            st.caption(s.get("text", "")[:300] + "...")

    if prompt := st.chat_input("Ask a medical information question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer (CPU model may take 2–5 min on first question)..."):
                answer, hits, intent = generate_response(
                    prompt,
                    rag,
                    ollama,
                    intent_model,
                    config,
                    st.session_state.messages[:-1],
                )
            st.markdown(answer)
            st.caption(f"Detected intent: `{intent}`")
            if hits:
                with st.expander("Sources"):
                    for s in hits:
                        loc = s.get("source_file", "unknown")
                        if s.get("page"):
                            loc += f" (page {s['page']})"
                        st.markdown(f"**{loc}** — similarity {s.get('similarity', 'n/a')}")
                        st.caption(s.get("text", "")[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.sources.append(hits)


if __name__ == "__main__":
    main()
