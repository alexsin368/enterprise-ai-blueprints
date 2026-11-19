# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
pgvector-backed vector store for the Knowledge Assistant.

Documents (see data.py) are split into paragraph-sized chunks, embedded through
LiteLLM's OpenAI-compatible ``/v1/embeddings`` endpoint, and stored in
a PostgreSQL table with a `pgvector` column. Similarity queries embed the query
the same way and rank chunks by cosine distance (`<=>`) in the database.

Why pgvector + gateway embeddings (instead of an in-memory TF-IDF store)?
  • Retrieval quality — dense neural embeddings capture meaning, not just word
    overlap, so paraphrased questions still match the right passages.
  • It reuses the stack's own embedding service (LiteLLM) rather than
    downloading a model into the container, and persists vectors in a real
    database — the same shape a production RAG system uses.

The store syncs itself on startup: it connects to Postgres, enables the `vector`
extension, creates the chunks table (dimension probed from the embedding model),
and reconciles it with the documents in data.py (the synthetic seed plus any files
loaded from DOCS_DIR). Each document's chunks carry a content hash, so only new or
changed documents are embedded and documents that disappeared are deleted — the
corpus is embedded once and reused across restarts.

Configuration (environment variables):
  OPENAI_BASE_URL   LiteLLM OpenAI-compatible base URL (…/v1)
  OPENAI_API_KEY    LiteLLM / LiteLLM API key
  EMBEDDING_MODEL   Embedding model id registered in the gateway
  CHUNK_SIZE        Target chunk length in characters for loaded files (default 1000)
  CHUNK_OVERLAP     Characters repeated between consecutive split chunks (default 150)
  PGVECTOR_URL      Full Postgres connection URL.
                    Matches the Agent Toolkit's `pgvector-credentials` DATABASE_URL.
                    Takes precedence over the discrete PG* variables when set.
  PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD   Discrete connection fallback.

In production, point these at your managed Postgres + embedding model and swap
data.py for a connector to your real document store; nothing else in the MCP
server needs to change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any

import psycopg
from openai import OpenAI

from . import data

logger = logging.getLogger(__name__)

# --- Embedding configuration ------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-base-en")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Postgres / pgvector configuration --------------------------------------
# A full connection URL (e.g. the Agent Toolkit's pgvector DATABASE_URL) wins
# over the discrete PG* variables when provided.
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "").strip()
PG_HOST = os.getenv("PGHOST", "pgvector.pgvector.svc.cluster.local")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DATABASE = os.getenv("PGDATABASE", "agentdb")
PG_USER = os.getenv("PGUSER", "agentuser")
PG_PASSWORD = os.getenv("PGPASSWORD", "")
PG_CONNECT_RETRIES = int(os.getenv("PG_CONNECT_RETRIES", "30"))
PG_CONNECT_DELAY = float(os.getenv("PG_CONNECT_DELAY", "2.0"))

TABLE = "knowledge_chunks"


def _split_long(text: str) -> list[str]:
    """Split text longer than CHUNK_SIZE into overlapping windows on whitespace."""
    pieces = []
    start = 0
    while len(text) - start > CHUNK_SIZE:
        end = text.rfind(" ", start + CHUNK_SIZE // 2, start + CHUNK_SIZE)
        if end == -1:
            end = start + CHUNK_SIZE
        pieces.append(text[start:end].strip())
        start = max(end - CHUNK_OVERLAP, start + 1)
    pieces.append(text[start:].strip())
    return [p for p in pieces if p]


def _pack(paragraphs: list[str]) -> list[str]:
    """Merge consecutive paragraphs into chunks of up to CHUNK_SIZE characters."""
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > CHUNK_SIZE:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document body into chunks, carrying doc metadata.

    Paragraphs are separated by a blank line. Seed documents keep one chunk per
    paragraph. Loaded files (PDF pages, Word paragraphs, …) vary wildly in
    paragraph length, so their paragraphs are packed into ~CHUNK_SIZE chunks. Any
    chunk still over CHUNK_SIZE is split into overlapping windows. Each chunk keeps
    a back-reference to its parent document so search results can cite the source.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if p.strip()]
    if doc.get("source"):
        paragraphs = _pack(paragraphs)
    paragraphs = [piece for para in paragraphs for piece in _split_long(para)]
    chunks = []
    for idx, para in enumerate(paragraphs):
        chunks.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "collection": doc["collection"],
                "tags": doc["tags"],
                "updated": doc["updated"],
                "chunk_index": idx,
                "text": para,
            }
        )
    return chunks


def _content_hash(doc: dict[str, Any]) -> str:
    """Fingerprint of everything that affects a document's stored chunks.

    `updated` is left out on purpose: for loaded files it is the file mtime, which
    a rebuild of the image can change without the content changing.
    """
    payload = json.dumps(
        [doc["title"], doc["collection"], doc["tags"], doc["text"],
         EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_pgvector(vec: list[float]) -> str:
    """Render an embedding as a pgvector text literal, e.g. '[0.1,0.2,0.3]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class KnowledgeIndex:
    """A pgvector cosine-similarity index over document chunks.

    Embeddings are produced by LiteLLM and stored in PostgreSQL. The
    public interface (`search(query, top_k, collection, min_score)`) matches the
    previous in-memory store so the MCP server is unchanged.
    """

    def __init__(self) -> None:
        self._client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
        self.build()

    # -- connection ---------------------------------------------------------
    def _conninfo(self) -> str:
        # psycopg accepts a libpq URL directly as conninfo.
        if PGVECTOR_URL:
            return PGVECTOR_URL
        return (
            f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE} "
            f"user={PG_USER} password={PG_PASSWORD}"
        )

    def _connect(self) -> psycopg.Connection:
        """Open an autocommit connection, retrying until Postgres is ready.

        In the sidecar topology the knowledge MCP container may start before the
        pgvector sidecar is accepting connections, so we poll with a bounded
        retry loop rather than crashing immediately.
        """
        last_err: Exception | None = None
        for attempt in range(1, PG_CONNECT_RETRIES + 1):
            try:
                return psycopg.connect(self._conninfo(), autocommit=True)
            except psycopg.OperationalError as err:  # DB not ready yet
                # Wrong credentials won't fix themselves — fail fast instead of
                # retrying for a minute.
                if "authentication failed" in str(err):
                    raise RuntimeError(
                        f"Postgres rejected the credentials ({err}). Check that "
                        "PGVECTOR_URL / PGPASSWORD match the pgvector user's password."
                    ) from err
                last_err = err
                logger.info(
                    "Waiting for Postgres (%s:%s) — attempt %d/%d",
                    PG_HOST, PG_PORT, attempt, PG_CONNECT_RETRIES,
                )
                time.sleep(PG_CONNECT_DELAY)
        raise RuntimeError(
            f"Could not connect to Postgres at {PG_HOST}:{PG_PORT} after "
            f"{PG_CONNECT_RETRIES} attempts"
        ) from last_err

    # -- embeddings ---------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via LiteLLM embeddings endpoint."""
        resp = self._client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        # The API preserves input order, but sort by index to be safe.
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]

    # -- schema + ingest ----------------------------------------------------
    def build(self) -> None:
        """Ensure the schema exists and sync the stored chunks with data.py."""
        conn = self._connect()
        try:
            # Serialize concurrent starts (e.g. old + new pod during a rollout) so
            # two servers never ingest the same document at once. Released when
            # the connection closes.
            conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (TABLE,))
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            table_exists = conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL", (TABLE,)
            ).fetchone()[0]

            if not table_exists:
                dim = len(self._embed(["dimension probe"])[0])
                logger.info("Embedding model '%s' → %d dimensions", EMBEDDING_MODEL, dim)
                conn.execute(
                    f"""
                    CREATE TABLE {TABLE} (
                        id           SERIAL PRIMARY KEY,
                        doc_id       TEXT NOT NULL,
                        title        TEXT NOT NULL,
                        collection   TEXT NOT NULL,
                        tags         TEXT[] NOT NULL DEFAULT '{{}}',
                        updated      TEXT,
                        chunk_index  INTEGER NOT NULL,
                        text         TEXT NOT NULL,
                        embedding    vector({dim}) NOT NULL,
                        content_hash TEXT
                    )
                    """  # nosec B608 - TABLE is a module constant; values are bound parameters
                )
                conn.execute(
                    f"CREATE INDEX ON {TABLE} (collection)"  # nosec B608 - TABLE is a module constant; values are bound parameters
                )

            # Tables created before content hashes existed get the column added;
            # their rows have a NULL hash, so they are re-embedded once.
            conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS content_hash TEXT")  # nosec B608 - TABLE is a module constant; values are bound parameters
            self._sync(conn)
        finally:
            conn.close()

    def _sync(self, conn: psycopg.Connection) -> None:
        """Embed new/changed documents and delete ones that no longer exist."""
        stored = dict(
            conn.execute(
                f"SELECT doc_id, min(content_hash) FROM {TABLE} GROUP BY doc_id"  # nosec B608 - TABLE is a module constant; values are bound parameters
            ).fetchall()
        )
        wanted = {doc["doc_id"]: (doc, _content_hash(doc)) for doc in data.all_documents()}

        removed = [doc_id for doc_id in stored if doc_id not in wanted]
        if removed:
            conn.execute(f"DELETE FROM {TABLE} WHERE doc_id = ANY(%s)", (removed,))  # nosec B608 - TABLE is a module constant; values are bound parameters

        changed = [(doc, h) for doc_id, (doc, h) in wanted.items() if stored.get(doc_id) != h]
        chunk_count = 0
        for doc, content_hash in changed:
            chunk_count += self._ingest_document(conn, doc, content_hash)

        logger.info(
            "KnowledgeIndex synced: %d document(s) embedded (%d chunks), %d removed, "
            "%d unchanged.",
            len(changed), chunk_count, len(removed), len(wanted) - len(changed),
        )
        if not wanted:
            logger.warning("KnowledgeIndex: no documents to serve — add files to the docs folder.")

    def _ingest_document(
        self, conn: psycopg.Connection, doc: dict[str, Any], content_hash: str
    ) -> int:
        """Embed one document's chunks and replace its rows in a single transaction."""
        chunks = _chunk_document(doc)
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            embeddings.extend(self._embed([c["text"] for c in chunks[start:start + EMBED_BATCH]]))

        # Delete + insert together, so a crash mid-way never leaves a document
        # half-stored with a hash that says it is up to date.
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE doc_id = %s", (doc["doc_id"],))  # nosec B608 - TABLE is a module constant; values are bound parameters
            cur.executemany(
                f"""
                INSERT INTO {TABLE}
                    (doc_id, title, collection, tags, updated, chunk_index, text,
                     embedding, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                """,  # nosec B608 - TABLE is a module constant; values are bound parameters
                [
                    (
                        c["doc_id"], c["title"], c["collection"], c["tags"],
                        c["updated"], c["chunk_index"], c["text"],
                        _to_pgvector(emb), content_hash,
                    )
                    for c, emb in zip(chunks, embeddings)
                ],
            )
        logger.info("Embedded %s (%d chunks).", doc["doc_id"], len(chunks))
        return len(chunks)

    # -- query --------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 4,
        collection: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return the top_k most similar chunks to `query`.

        Args:
            query: the natural-language query.
            top_k: maximum number of chunks to return.
            collection: optional collection filter (e.g. "hr", "it").
            min_score: drop results whose cosine similarity is below this.

        Each result includes the chunk text, its parent document metadata, and a
        cosine-similarity score in [0, 1] rounded to 4 decimals.
        """
        if not query or not query.strip():
            return []
        if top_k <= 0:
            return []

        qvec = _to_pgvector(self._embed([query])[0])

        where = ""
        params: list[Any] = [qvec]
        if collection is not None:
            where = "WHERE collection = %s"
            params.append(collection)
        # `<=>` is cosine distance (1 - cosine similarity); order ascending and
        # convert to a similarity score for the caller.
        params.extend([qvec, top_k])

        sql = f"""
            SELECT doc_id, title, collection, updated, chunk_index, text,
                   1 - (embedding <=> %s::vector) AS score
            FROM {TABLE}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """  # nosec B608 - TABLE is a module constant; values are bound parameters

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        results = []
        for doc_id, title, coll, updated, chunk_index, text, score in rows:
            score = float(score)
            if score < min_score:
                continue
            results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "collection": coll,
                    "updated": updated,
                    "chunk_index": chunk_index,
                    "snippet": text,
                    "score": round(score, 4),
                }
            )
        return results
