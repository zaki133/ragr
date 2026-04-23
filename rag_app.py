#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET

import gradio as gr
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import pandas as pd
from docx import Document as DocxDocument


ROOT = Path("/opt/nextcloud-rag")
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)


@dataclass
class Config:
    nextcloud_webdav_url: str
    nextcloud_username: str
    nextcloud_app_password: str
    nextcloud_remote_path: str
    local_sync_dir: Path
    state_dir: Path
    index_dir: Path
    embedding_model: str
    openai_base_url: str
    openai_api_key: str
    model_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    allowed_suffixes: Tuple[str, ...]
    system_prompt: str


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config() -> Config:
    suffixes = tuple(s.strip().lower() for s in env("ALLOWED_SUFFIXES", ".pdf,.txt,.md,.docx,.csv").split(",") if s.strip())
    return Config(
        nextcloud_webdav_url=env("NEXTCLOUD_WEBDAV_URL"),
        nextcloud_username=env("NEXTCLOUD_USERNAME"),
        nextcloud_app_password=env("NEXTCLOUD_APP_PASSWORD"),
        nextcloud_remote_path=env("NEXTCLOUD_REMOTE_PATH", "/model_data"),
        local_sync_dir=Path(env("LOCAL_SYNC_DIR", str(ROOT / "data/model_data"))),
        state_dir=Path(env("STATE_DIR", str(ROOT / "state"))),
        index_dir=Path(env("INDEX_DIR", str(ROOT / "index"))),
        embedding_model=env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        openai_base_url=env("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
        openai_api_key=env("OPENAI_API_KEY", "dummy"),
        model_name=env("MODEL_NAME", "Llama-3.2-3B-Instruct-uncensored-Q4_K_M"),
        chunk_size=int(env("CHUNK_SIZE", "1000")),
        chunk_overlap=int(env("CHUNK_OVERLAP", "150")),
        top_k=int(env("TOP_K", "4")),
        allowed_suffixes=suffixes,
        system_prompt=env(
            "SYSTEM_PROMPT",
            "You answer questions using only the provided context from the user's synced documents. If the answer is not clearly supported by the context, say you do not know. Prefer concise answers and cite source filenames when relevant.",
        ),
    )


class NextcloudSync:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.auth = (cfg.nextcloud_username, cfg.nextcloud_app_password)

    def _normalize_base(self) -> str:
        return self.cfg.nextcloud_webdav_url.rstrip("/") + "/"

    def _remote_root(self) -> str:
        base = self._normalize_base()
        remote = self.cfg.nextcloud_remote_path.strip("/")
        if not remote:
            return base
        return urljoin(base, quote(remote) + "/")

    def _propfind(self, url: str, depth: int = 1) -> requests.Response:
        headers = {"Depth": str(depth), "Content-Type": "application/xml"}
        body = """<?xml version='1.0'?>
<d:propfind xmlns:d='DAV:'>
  <d:prop>
    <d:getcontentlength />
    <d:getlastmodified />
    <d:resourcetype />
  </d:prop>
</d:propfind>
"""
        resp = self.session.request("PROPFIND", url, headers=headers, data=body.encode("utf-8"), timeout=120)
        resp.raise_for_status()
        return resp

    def _list_files_recursive(self, root_url: str) -> List[Tuple[str, str]]:
        files: List[Tuple[str, str]] = []
        queue = [root_url]
        ns = {"d": "DAV:"}
        seen = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            resp = self._propfind(current, depth=1)
            tree = ET.fromstring(resp.content)
            responses = tree.findall("d:response", ns)
            for i, item in enumerate(responses):
                href = item.findtext("d:href", default="", namespaces=ns)
                if not href:
                    continue
                full_url = requests.compat.urljoin(current, href)
                prop = item.find("d:propstat/d:prop", ns)
                if prop is None:
                    continue
                is_dir = prop.find("d:resourcetype/d:collection", ns) is not None
                # skip self item
                if i == 0 and full_url.rstrip("/") == current.rstrip("/"):
                    continue
                if is_dir:
                    queue.append(full_url.rstrip("/") + "/")
                else:
                    path = full_url.split(self._remote_root(), 1)[-1]
                    path = requests.utils.unquote(path)
                    files.append((full_url, path))
        return files

    def sync(self) -> str:
        self.cfg.local_sync_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.local_sync_dir.exists():
            shutil.rmtree(self.cfg.local_sync_dir)
        self.cfg.local_sync_dir.mkdir(parents=True, exist_ok=True)
        root_url = self._remote_root()
        files = self._list_files_recursive(root_url)
        kept = 0
        skipped = 0
        for remote_url, rel_path in files:
            suffix = Path(rel_path).suffix.lower()
            if suffix not in self.cfg.allowed_suffixes:
                skipped += 1
                continue
            local_path = self.cfg.local_sync_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with self.session.get(remote_url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    shutil.copyfileobj(resp.raw, f)
            kept += 1
        manifest = {
            "synced_files": kept,
            "skipped_files": skipped,
            "remote_path": self.cfg.nextcloud_remote_path,
            "local_dir": str(self.cfg.local_sync_dir),
        }
        (self.cfg.state_dir / "last_sync.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return f"Sync complete. Downloaded {kept} supported files and skipped {skipped} unsupported files."


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
        return "\n".join(chunks)
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_csv(index=False)
    return ""


def collect_documents(cfg: Config) -> List[dict]:
    docs = []
    for file in cfg.local_sync_dir.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in cfg.allowed_suffixes:
            continue
        try:
            text = extract_text(file)
        except Exception as exc:
            print(f"Skipping {file}: {exc}", file=sys.stderr)
            continue
        if text.strip():
            docs.append({
                "text": text,
                "source": str(file.relative_to(cfg.local_sync_dir)),
            })
    return docs


def build_index(cfg: Config) -> str:
    docs = collect_documents(cfg)
    if not docs:
        raise RuntimeError("No supported documents with extractable text were found in the sync directory.")
    if cfg.index_dir.exists():
        shutil.rmtree(cfg.index_dir)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    texts: List[str] = []
    metadatas: List[dict] = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk": idx})
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    store.save_local(str(cfg.index_dir))
    stats = {
        "documents": len(docs),
        "chunks": len(texts),
        "embedding_model": cfg.embedding_model,
    }
    (cfg.state_dir / "last_index.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return f"Index rebuilt. Documents: {len(docs)}. Chunks: {len(texts)}."


def load_store(cfg: Config) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    return FAISS.load_local(str(cfg.index_dir), embeddings, allow_dangerous_deserialization=True)


def build_context(cfg: Config, question: str) -> Tuple[str, List[dict]]:
    store = load_store(cfg)
    docs = store.similarity_search(question, k=cfg.top_k)
    context_blocks = []
    used = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        used.append({"source": source, "chunk": doc.metadata.get("chunk")})
        context_blocks.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(context_blocks), used


def call_model(cfg: Config, question: str, context: str) -> str:
    url = cfg.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model_name,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def ask(cfg: Config, question: str) -> Tuple[str, str]:
    context, used = build_context(cfg, question)
    answer = call_model(cfg, question, context)
    sources = "\n".join(f"- {item['source']} (chunk {item['chunk']})" for item in used)
    return answer, sources


def cmd_sync(_: argparse.Namespace) -> None:
    cfg = load_config()
    print(NextcloudSync(cfg).sync())


def cmd_build_index(_: argparse.Namespace) -> None:
    cfg = load_config()
    print(build_index(cfg))


def cmd_sync_index(_: argparse.Namespace) -> None:
    cfg = load_config()
    print(NextcloudSync(cfg).sync())
    print(build_index(cfg))


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = load_config()
    answer, sources = ask(cfg, args.question)
    print("Answer:\n")
    print(answer)
    print("\nSources:\n")
    print(sources)


def cmd_serve(args: argparse.Namespace) -> None:
    cfg = load_config()

    def ui_sync() -> str:
        return NextcloudSync(cfg).sync()

    def ui_build() -> str:
        return build_index(cfg)

    def ui_sync_build() -> str:
        return NextcloudSync(cfg).sync() + "\n" + build_index(cfg)

    def ui_ask(question: str) -> Tuple[str, str]:
        if not question.strip():
            return "Please enter a question.", ""
        try:
            return ask(cfg, question)
        except Exception as exc:
            return f"Error: {exc}", ""

    with gr.Blocks(title="Nextcloud RAG with llama.cpp") as demo:
        gr.Markdown("# Nextcloud RAG with llama.cpp")
        with gr.Row():
            sync_btn = gr.Button("Sync Nextcloud")
            build_btn = gr.Button("Rebuild Index")
            both_btn = gr.Button("Sync + Rebuild")
        status = gr.Textbox(label="Status", lines=8)
        sync_btn.click(ui_sync, outputs=status)
        build_btn.click(ui_build, outputs=status)
        both_btn.click(ui_sync_build, outputs=status)
        question = gr.Textbox(label="Question")
        ask_btn = gr.Button("Ask Model")
        answer_box = gr.Textbox(label="Answer", lines=12)
        sources_box = gr.Textbox(label="Sources", lines=8)
        ask_btn.click(ui_ask, inputs=question, outputs=[answer_box, sources_box])
    demo.launch(server_name=args.host, server_port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nextcloud RAG app using llama.cpp OpenAI-compatible server")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("build-index")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("sync-index")
    p.set_defaults(func=cmd_sync_index)

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_serve)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
