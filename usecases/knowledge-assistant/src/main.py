#!/usr/bin/env python3

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Launch the Knowledge Assistant with DevUI.

Starts a web interface for interacting with the knowledge agent. The agent
answers questions by searching an internal knowledge base (the "knowledge" MCP
sidecar) and the public web (the Brave Search MCP sidecar), and routes generation
to a LiteLLM-served model.
"""

import logging
import os

from agent_framework.observability import configure_otel_providers
from agent_framework.devui import serve
from dotenv import load_dotenv
from langfuse import get_client

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def setup_observability() -> None:
    """Configure Langfuse + OpenTelemetry if credentials are available."""
    has_langfuse_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(
        os.getenv("LANGFUSE_SECRET_KEY")
    )
    if not has_langfuse_keys:
        logger.info(
            "Langfuse not configured: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing"
        )
        return

    langfuse = get_client()
    if not langfuse.auth_check():
        logger.warning("Langfuse auth check failed. Verify LANGFUSE keys and base URL.")
        return

    enable_sensitive_data = _is_truthy(
        os.getenv("LANGFUSE_ENABLE_SENSITIVE_DATA"), default=True
    )
    configure_otel_providers(enable_sensitive_data=enable_sensitive_data)
    logger.info(
        "Langfuse observability enabled (sensitive_data=%s)", enable_sensitive_data
    )


def main():
    """Launch DevUI with the knowledge agent."""
    setup_observability()
    from agent import agent, KNOWLEDGE_MCP_URL, BRAVE_MCP_URL, ENABLE_WEB_SEARCH

    logger.info("=" * 60)
    logger.info("Knowledge Assistant with DevUI")
    logger.info("=" * 60)
    logger.info("Starting web interface at http://localhost:8090")
    logger.info("")
    logger.info("Features:")
    logger.info("  • Natural-language Q&A grounded in an internal knowledge base")
    logger.info("  • Internal RAG via knowledge MCP server: %s", KNOWLEDGE_MCP_URL)
    if ENABLE_WEB_SEARCH:
        logger.info("  • Web search via Brave Search MCP server: %s", BRAVE_MCP_URL)
    else:
        logger.info("  • Web search DISABLED (ENABLE_WEB_SEARCH=false)")
    logger.info("  • Persistent memory via Redis (text-only)")
    logger.info("  • OpenAI-compatible API (LiteLLM)")
    logger.info("")
    logger.info("Prerequisites:")
    logger.info("  1. Knowledge MCP server reachable at KNOWLEDGE_MCP_URL")
    logger.info("  2. Brave Search MCP server reachable at BRAVE_MCP_URL (if web search enabled)")
    logger.info("  3. Redis: docker run -d -p 6379:6379 redis/redis-stack-server")
    logger.info("  4. Configure .env with your LLM endpoint")
    logger.info("")
    logger.info("=" * 60)

    auto_open = os.getenv("AUTO_OPEN_BROWSER", "true").lower() != "false"
    host = os.getenv("DEVUI_HOST", "0.0.0.0")  # nosec B104 - containerized service must bind all interfaces to be reachable
    serve(entities=[agent], port=8090, host=host, auto_open=auto_open)


if __name__ == "__main__":
    main()
