# ⎈ KubeBot — Kubernetes RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers
questions about Kubernetes internals, powered by **LlamaIndex**, **ChromaDB**,
and **Claude**.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit UI                   │
│     Chat · Source citations · Settings      │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│            RAG Pipeline (LlamaIndex)        │
│  ┌──────────────┐    ┌───────────────────┐  │
│  │  Query Engine│───▶│  Claude (Sonnet)  │  │
│  └──────┬───────┘    └───────────────────┘  │
│         │                                   │
│  ┌──────▼───────┐                           │
│  │  ChromaDB    │  ← Persistent Vector DB   │
│  │  Retriever   │    (BGE-small embeddings) │
│  └──────────────┘                           │
└─────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────┐
│         Knowledge Base (30 docs)            │
│  Architecture · Workloads · Networking      │
│  Storage · Security · Scheduling · Config   │
└─────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ereilly89/k8s-rag-chatbot.git
cd k8s-rag-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, enter your API key, click **Build Index**, then start asking!

---

## Docker Deployment

```bash
# Build and run
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build

# Or set it in .env and just run:
docker compose up --build
```

The ChromaDB index and downloaded docs are persisted in named Docker volumes so
you don't need to re-index on every restart.

---

## Knowledge Base

The chatbot indexes **30 Kubernetes documentation pages** covering:

| Category | Topics |
|----------|--------|
| **Architecture** | Components, Control Plane, etcd, Nodes, Controllers |
| **Workloads** | Pods, Deployments, StatefulSets, DaemonSets, Jobs |
| **Networking** | Services, Ingress, NetworkPolicies, DNS |
| **Storage** | Volumes, PersistentVolumes, StorageClasses |
| **Configuration** | ConfigMaps, Secrets, Resource Management |
| **Security** | Pod Security Standards, RBAC |
| **Scheduling** | kube-scheduler, Node Affinity, Priority |

All content is fetched directly from the official Kubernetes GitHub repository.

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| **LLM** | `anthropic` via LlamaIndex | Answer generation |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (HuggingFace) | Local, free embeddings |
| **Vector DB** | `chromadb` | Persistent semantic search |
| **RAG Framework** | `llama-index` | Retrieval pipeline |
| **UI** | `streamlit` | Chat interface |

---

## Project Structure

```
k8s-rag-chatbot/
├── app.py              # Streamlit chat UI
├── rag/
│   ├── __init__.py
│   ├── ingest.py       # Doc fetching, chunking, embedding
│   └── query.py        # RAG query engine + Claude integration
├── data/               # Cached markdown files (auto-created)
├── chroma_db/          # Persisted vector store (auto-created)
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Customization

### Add more docs
Edit `K8S_DOC_URLS` in `rag/ingest.py` to add more Kubernetes documentation pages (raw GitHub markdown URLs work best).

### Swap the embedding model
Change `model_name` in `build_index()` in `rag/ingest.py`. Any HuggingFace sentence-transformer model works.

### Tune retrieval
Use the **top-k** slider in the sidebar to control how many chunks are retrieved per query.

### System prompt
Edit `K8S_SYSTEM_PROMPT` in `rag/query.py` to change KubeBot's persona and behavior.
