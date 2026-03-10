"""
query.py — RAG query engine using LlamaIndex + Claude

Wraps the vector index with a retrieval-augmented generation pipeline
that uses Claude as the LLM and returns answers with source citations.
"""

import os
from typing import Generator

from dotenv import load_dotenv

load_dotenv()

from llama_index.core import Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.anthropic import Anthropic as AnthropicLLM

# ── System prompt ────────────────────────────────────────────────────────────
K8S_SYSTEM_PROMPT = """\
You are KubeBot, an expert Kubernetes assistant with deep knowledge of \
Kubernetes internals, architecture, and best practices.

Your answers are:
- Accurate and grounded in the provided context documents
- Clearly structured with headers when appropriate
- Illustrated with YAML examples when helpful
- Honest about uncertainty — if the context doesn't cover something, say so

When referencing specific Kubernetes components (like etcd, kube-scheduler, \
kubelet, etc.), be precise about their roles in the control plane or data plane.
"""

# ── RAG prompt template ──────────────────────────────────────────────────────
QA_TEMPLATE = PromptTemplate(
    """\
Context from Kubernetes documentation:
---------------------
{context_str}
---------------------

Using ONLY the context above, answer the following question. \
If the context doesn't contain enough information, say so clearly \
and share what you do know from the context.

Question: {query_str}

Answer:\
"""
)


def build_query_engine(index, similarity_top_k: int = 5):
    """
    Build a RetrieverQueryEngine over the given VectorStoreIndex.

    Args:
        index: A LlamaIndex VectorStoreIndex
        similarity_top_k: Number of document chunks to retrieve per query
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    llm = AnthropicLLM(
        model=model,
        api_key=api_key,
        system_prompt=K8S_SYSTEM_PROMPT,
        max_tokens=2048,
    )
    Settings.llm = llm

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=QA_TEMPLATE,
        streaming=False,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )


def query_with_sources(query_engine, question: str) -> dict:
    """
    Run a RAG query and return the answer + source metadata.

    Returns:
        {
            "answer": str,
            "sources": [{"topic": str, "source": str, "snippet": str}, ...]
        }
    """
    response = query_engine.query(question)

    # Extract source nodes for citation
    sources = []
    seen_sources = set()
    for node in response.source_nodes:
        meta = node.node.metadata
        src = meta.get("source", "unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            snippet = node.node.get_content()[:200].replace("\n", " ").strip()
            sources.append({
                "topic": meta.get("topic", "Unknown"),
                "source": src,
                "snippet": snippet + "…",
                "score": round(node.score or 0.0, 3),
            })

    return {
        "answer": str(response),
        "sources": sources,
    }
