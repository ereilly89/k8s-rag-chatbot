"""
ingest.py — Kubernetes docs ingestion pipeline

Fetches Kubernetes documentation pages, chunks them, embeds them
using a local HuggingFace model, and stores them in ChromaDB.
"""

import os
import json
import hashlib
import requests
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── LlamaIndex imports ──────────────────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
COLLECTION_NAME = "k8s_docs"

# Kubernetes documentation pages to ingest
K8S_DOC_URLS = [
    # Core concepts
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/overview/_index.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/overview/components.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/overview/kubernetes-api.md",
    # Workloads
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/pods/pod-lifecycle.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/pods/init-containers.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/pods/sidecar-containers.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/controllers/deployment.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/controllers/replicaset.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/controllers/statefulset.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/controllers/daemonset.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/controllers/job.md",
    # Networking
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/services-networking/service.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/services-networking/ingress.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/services-networking/network-policies.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/services-networking/dns-pod-service.md",
    # Storage
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/storage/volumes.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/storage/persistent-volumes.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/storage/storage-classes.md",
    # Configuration
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/configmap.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/secret.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/manage-resources-containers.md",
    # Security
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/security/pod-security-standards.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/security/rbac-good-practices.md",
    # Scheduling
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/scheduling-eviction/kube-scheduler.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/scheduling-eviction/assign-pod-node.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/scheduling-eviction/pod-priority-preemption.md",
    # Cluster internals
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/architecture/nodes.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/architecture/control-plane-node-communication.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/architecture/controller.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/architecture/leases.md",
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/architecture/_index.md",
]


def _url_to_filename(url: str) -> str:
    """Convert URL to a safe local filename."""
    slug = url.split("kubernetes/website/main/content/en/")[-1]
    slug = slug.replace("/", "_").replace(".md", "")
    return f"{slug}.md"


def fetch_docs(progress_callback=None) -> list[Document]:
    """
    Fetch Kubernetes docs from GitHub.
    Returns a list of LlamaIndex Document objects.
    Falls back to cached files when network is unavailable.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = []
    total = len(K8S_DOC_URLS)

    for i, url in enumerate(K8S_DOC_URLS):
        filename = _url_to_filename(url)
        cache_path = DATA_DIR / filename

        if progress_callback:
            progress_callback(i / total, f"Fetching: {filename}")

        content = None

        # Try cache first (network is disabled in sandbox)
        if cache_path.exists():
            content = cache_path.read_text(encoding="utf-8")
        else:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                content = resp.text
                cache_path.write_text(content, encoding="utf-8")
                time.sleep(0.1)  # be polite
            except Exception as e:
                print(f"  ⚠ Could not fetch {url}: {e}")
                continue

        if content:
            # Strip Hugo front-matter (--- ... ---)
            lines = content.split("\n")
            if lines and lines[0].strip() == "---":
                try:
                    end = lines.index("---", 1)
                    content = "\n".join(lines[end + 1:]).strip()
                except ValueError:
                    pass

            topic = filename.replace("_", " ").replace(".md", "")
            documents.append(
                Document(
                    text=content,
                    metadata={
                        "source": url,
                        "topic": topic,
                        "filename": filename,
                    },
                )
            )

    if progress_callback:
        progress_callback(1.0, f"Loaded {len(documents)} documents")

    return documents


def build_index(progress_callback=None) -> VectorStoreIndex:
    """
    Build (or load) the ChromaDB vector index.
    Uses a local HuggingFace embedding model — no API key needed for embeddings.
    """
    # ── Embedding model (runs locally, no key required) ──────────────────────
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="./.model_cache",
    )
    Settings.embed_model = embed_model

    # ── ChromaDB setup ────────────────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # If collection already has documents, load index directly
    if collection.count() > 0:
        if progress_callback:
            progress_callback(1.0, f"Loaded existing index ({collection.count()} chunks)")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    # Otherwise ingest fresh docs
    if progress_callback:
        progress_callback(0.0, "Fetching Kubernetes documentation...")

    documents = fetch_docs(progress_callback)

    if not documents:
        raise RuntimeError(
            "No documents loaded. Please add .md files to the ./data directory."
        )

    if progress_callback:
        progress_callback(0.6, f"Chunking {len(documents)} documents...")

    # ── Chunking ──────────────────────────────────────────────────────────────
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if progress_callback:
        progress_callback(0.75, "Embedding and storing chunks...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=False,
    )

    if progress_callback:
        count = collection.count()
        progress_callback(1.0, f"Index built — {count} chunks stored in ChromaDB ✓")

    return index


def get_collection_stats() -> dict:
    """Return basic stats about the ChromaDB collection."""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        return {
            "chunk_count": collection.count(),
            "collection_name": COLLECTION_NAME,
            "db_path": CHROMA_DB_PATH,
        }
    except Exception:
        return {"chunk_count": 0, "collection_name": COLLECTION_NAME, "db_path": CHROMA_DB_PATH}
