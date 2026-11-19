# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge MCP server (FastMCP over Streamable HTTP).

Exposes retrieval-augmented-generation (RAG) tools over an internal knowledge
base: semantic search across a pgvector store, plus document retrieval and
browsing. Backed by synthetic sample documents (see data.py) plus any files
dropped into the docs folder (see loader.py), embedded via LiteLLM and
indexed in PostgreSQL/pgvector (see vectorstore.py).

Runs as a sidecar container next to the agent; the agent connects as a remote
MCP client at http://localhost:8000/mcp.

Transport: Streamable HTTP on 0.0.0.0:8000, mount path /mcp.
stateless_http=True keeps the server free of per-session state so it needs no
sticky sessions (fine here because the index is read-only demo data).

All tools return JSON-serializable dicts. Errors are returned as structured
{"error": "..."} payloads (not exceptions) so the LLM can reason about them.

This server handles the *internal* half of the assistant's knowledge. Web search
is provided separately by the Brave Search MCP server (a second sidecar).
"""

from __future__ import annotations

import logging
import os

from mcp.server.fastmcp import FastMCP

from . import data
from .vectorstore import KnowledgeIndex

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")  # nosec B104 - containerized service must bind all interfaces to be reachable
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
# get_document returns at most this many characters of body text, so one large
# PDF can't flood the model's context window.
MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "20000"))

# Bind host/port explicitly on the constructor — mcp.run() may otherwise bind
# loopback, which is wrong inside a container.
mcp = FastMCP("knowledge", stateless_http=True, host=MCP_HOST, port=MCP_PORT)

# Connect to pgvector and seed embeddings at import time. This blocks until
# Postgres is reachable and the corpus has been embedded via LiteLLM,
# so the HTTP port only opens once the index is ready to serve.
INDEX = KnowledgeIndex()


# ---------------------------------------------------------------------------
# Semantic search (RAG retrieval)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_knowledge(query: str, top_k: int = 4, collection: str | None = None) -> dict:
    """Search the internal knowledge base for passages relevant to a query.

    This is the primary retrieval tool. It performs semantic (vector similarity)
    search over the internal documents and returns the most relevant passages
    with their source document, so answers can be grounded and cited.

    Args:
        query: the natural-language question or topic to search for.
        top_k: maximum number of passages to return (default 4).
        collection: optional collection to restrict the search to (call
            list_collections for the available names). Omit to search all.

    Returns a list of results, each with doc_id, title, collection, a text
    snippet, and a relevance score in [0, 1]. An empty list means nothing in the
    internal knowledge base matched — consider a web search instead.
    """
    known = set(data.collections().keys())
    if collection is not None and collection not in known:
        return {
            "error": f"Unknown collection '{collection}'.",
            "available_collections": sorted(known),
        }
    results = INDEX.search(query=query, top_k=top_k, collection=collection)
    return {
        "query": query,
        "collection": collection,
        "count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Document retrieval / browsing
# ---------------------------------------------------------------------------

@mcp.tool()
def get_document(doc_id: str) -> dict:
    """Fetch the full text and metadata of a single document by its id.

    Use this after search_knowledge when you need the complete context of a
    document rather than a single matching passage (e.g. "HR-001" or
    "hr/benefits-guide.pdf"). Very long documents are truncated; the result then
    has "truncated": true. Returns {"error": ...} if no such document exists.
    """
    doc = data.get_document(doc_id)
    if not doc:
        return {"error": f"No document found with id '{doc_id}'."}
    if len(doc["text"]) > MAX_DOCUMENT_CHARS:
        doc["text"] = doc["text"][:MAX_DOCUMENT_CHARS]
        doc["truncated"] = True
    return doc


@mcp.tool()
def list_documents(collection: str | None = None) -> dict:
    """List available documents (metadata only), optionally filtered by collection.

    Useful for discovering what the internal knowledge base contains before
    searching. Returns each document's id, title, collection, tags, and last
    updated date — but not the body text (use get_document for that).

    Args:
        collection: optional collection filter (call list_collections for the
            available names). Omit to list everything.
    """
    known = set(data.collections().keys())
    if collection is not None and collection not in known:
        return {
            "error": f"Unknown collection '{collection}'.",
            "available_collections": sorted(known),
        }
    docs = [data.document_meta(d) for d in data.documents_in(collection)]
    return {"collection": collection, "count": len(docs), "documents": docs}


@mcp.tool()
def list_collections() -> dict:
    """List the knowledge-base collections and how many documents each contains.

    Collections are the top-level groupings of internal content (e.g. hr, it,
    security, product, onboarding, or a folder name from the docs folder). Use
    this to orient before searching or to scope a search to the most relevant area.
    """
    counts = data.collections()
    return {
        "count": len(counts),
        "collections": [
            {"name": name, "document_count": n} for name, n in sorted(counts.items())
        ],
    }


if __name__ == "__main__":
    logger.info("Starting Knowledge MCP server on %s:%s (path /mcp)", MCP_HOST, MCP_PORT)
    logger.info("Indexed collections: %s", data.collections())
    # Streamable HTTP transport; serves the MCP endpoint at /mcp.
    mcp.run(transport="streamable-http")
