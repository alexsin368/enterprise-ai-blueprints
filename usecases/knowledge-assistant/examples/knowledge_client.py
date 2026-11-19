#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
examples/knowledge_client.py
Knowledge Assistant — natural-language client (Agent Framework DevUI).

DevUI exposes the OpenAI *Responses* API (POST /v1/responses), not
/v1/chat/completions. You address the agent by its entity id (from /v1/entities)
in `metadata.entity_id`, and all /v1/* routes require a bearer token.

Sending natural-language prompts causes the agent's LLM to call the internal
knowledge tools (search_knowledge, get_document, ...) and, when relevant, the
Brave Search web tools (brave_web_search, brave_news_search, ...), then answer
from the retrieved context.

Usage:
    AGENT_URL=http://localhost:8090 DEVUI_AUTH_TOKEN=<token> python3 knowledge_client.py

Uses only the Python standard library (no extra dependencies).
"""

import json
import os
import urllib.request

AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8090").rstrip("/")
if not AGENT_URL.startswith(("http://", "https://")):
    raise SystemExit(f"AGENT_URL must be an http(s) URL, got: {AGENT_URL}")
TOKEN = os.environ.get("DEVUI_AUTH_TOKEN", "")
if not TOKEN:
    raise SystemExit("Set DEVUI_AUTH_TOKEN (printed by deploy.sh).")

_HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Prompts 1-3 should hit the INTERNAL knowledge base; prompt 4 should trigger a
# WEB search via the Brave Search MCP tools (when web search is enabled).
PROMPTS = [
    "How many PTO days do full-time employees get, and how much can I carry over?",
    "I can't connect to the VPN and I get a certificate error. What should I do?",
    "What are the support tiers and uptime SLAs for the Acme Analytics Platform?",
    "Search the web for the latest news on retrieval-augmented generation.",
]


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{AGENT_URL}{path}", headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=600) as resp:  # nosec B310 - AGENT_URL scheme validated above
        return json.loads(resp.read())


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{AGENT_URL}{path}", data=json.dumps(body).encode(), headers=_HEADERS, method="POST"
    )
    with urllib.request.urlopen(req, timeout=600) as resp:  # nosec B310 - AGENT_URL scheme validated above
        return json.loads(resp.read())


def entity_id() -> str:
    return _get("/v1/entities")["entities"][0]["id"]


def ask(eid: str, prompt: str) -> str:
    resp = _post("/v1/responses", {"input": prompt, "metadata": {"entity_id": eid}, "stream": False})
    if resp.get("status") and resp["status"] != "completed":
        return f"[status:{resp['status']}] {resp.get('error')}"
    out = []
    for it in resp.get("output", []):
        if it.get("type") == "message":
            for c in it.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    out.append(c.get("text", ""))
    return "\n".join(out).strip()


def main() -> None:
    print(f"Target: {AGENT_URL}\n")
    eid = entity_id()
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"=== Query {i} ===")
        print(f"User: {prompt}")
        print(f"Assistant: {ask(eid, prompt)}\n")


if __name__ == "__main__":
    main()
