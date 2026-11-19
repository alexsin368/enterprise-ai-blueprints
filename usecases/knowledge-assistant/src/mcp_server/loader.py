# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Folder loader for the Knowledge Assistant's internal knowledge base.

Reads documents from a directory tree and turns each file into the same document
shape as the synthetic seed in data.py, so they flow through the same chunking,
embedding, and MCP tools.

Layout — the first-level subfolder is the collection:

    docs/
      hr/benefits-guide.pdf           → collection "hr",  doc_id "hr/benefits-guide.pdf"
      it/runbooks/vpn.md              → collection "it",  doc_id "it/runbooks/vpn.md"
      handbook.docx                   → collection "general"

Supported file types:
  .txt .md .markdown .rst .csv   plain text (UTF-8)
  .pdf                           text layer via pypdf (scanned PDFs have none — no OCR)
  .docx                          paragraphs + tables via python-docx
  .html .htm                     visible text, tags stripped

Hidden files/folders (".*") and unsupported extensions are skipped. A file that
can't be parsed is logged and skipped, so one bad file never blocks startup.

Configuration (environment variables):
  DOCS_DIR   Root folder to load (default: the `docs/` folder next to this
             package, i.e. /app/docs in the container).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(Path(__file__).resolve().parent.parent / "docs")))
DEFAULT_COLLECTION = "general"


# ---------------------------------------------------------------------------
# Per-type readers. Each returns (text, title-or-None).
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> tuple[str, str | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Use a leading Markdown "# Heading" as the title when there is one.
    match = re.match(r"\s*#\s+(.+)", text)
    return text, match.group(1).strip() if match else None


def _read_pdf(path: Path) -> tuple[str, str | None]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # Pages become paragraphs (blank-line separated) for the chunker.
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or None
    return "\n\n".join(p for p in pages if p), title


def _read_docx(path: Path) -> tuple[str, str | None]:
    import docx

    document = docx.Document(str(path))
    blocks = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    title = (document.core_properties.title or "").strip() or None
    return "\n\n".join(blocks), title


class _HTMLText(HTMLParser):
    """Collect visible text, one block per element; skip script/style."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[str] = []
        self.title: str | None = None
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or self._skip:
            return
        if self._in_title:
            self.title = text
        else:
            self.blocks.append(text)


def _read_html(path: Path) -> tuple[str, str | None]:
    parser = _HTMLText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parser.blocks), parser.title


_READERS: dict[str, Callable[[Path], tuple[str, str | None]]] = {
    ".txt": _read_text,
    ".md": _read_text,
    ".markdown": _read_text,
    ".rst": _read_text,
    ".csv": _read_text,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".html": _read_html,
    ".htm": _read_html,
}

SUPPORTED_EXTENSIONS = tuple(sorted(_READERS))


def _title_from_filename(path: Path) -> str:
    return re.sub(r"[_-]+", " ", path.stem).strip().title() or path.name


def load_documents(root: Path = DOCS_DIR) -> list[dict[str, Any]]:
    """Load every supported file under `root` as a knowledge-base document."""
    if not root.is_dir():
        logger.info("Docs folder %s not found — no file documents loaded.", root)
        return []

    docs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in rel.parts):
            continue
        reader = _READERS.get(path.suffix.lower())
        if reader is None:
            logger.warning("Skipping %s: unsupported type (supported: %s).",
                           rel, ", ".join(SUPPORTED_EXTENSIONS))
            continue
        try:
            text, title = reader(path)
        except Exception as err:  # corrupt / encrypted / unreadable file
            logger.warning("Skipping %s: could not read it (%s).", rel, err)
            continue
        if not text.strip():
            logger.warning("Skipping %s: no extractable text (scanned PDF? OCR is not supported).", rel)
            continue

        collection = rel.parts[0].lower() if len(rel.parts) > 1 else DEFAULT_COLLECTION
        updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        docs.append(
            {
                "doc_id": rel.as_posix(),
                "title": title or _title_from_filename(path),
                "collection": collection,
                "tags": [],
                "updated": updated,
                "text": text,
                "source": rel.as_posix(),
            }
        )

    logger.info("Loaded %d document(s) from %s.", len(docs), root)
    return docs
