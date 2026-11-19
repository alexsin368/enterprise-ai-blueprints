# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Assistant agent.

A Microsoft Agent Framework agent that answers questions by combining two
sources of truth, each exposed as a remote MCP server:

  1. An INTERNAL knowledge base (the "knowledge" MCP sidecar at
     KNOWLEDGE_MCP_URL) — retrieval-augmented search over an internal vector
     database and document store.
  2. The public WEB (the Brave Search MCP sidecar at BRAVE_MCP_URL) — live web,
     news, and related searches via the Brave Search API.

The agent connects to both MCP servers as a client and exposes their tools to
the LLM, which decides which to call. It keeps optional per-session memory in
Redis (text-only), reusing the sanitized context-provider pattern from the
coding-agent / banking-assistant usecases.
"""

import logging
import os
import re
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework.redis import RedisContextProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)



def _redis_url_with_password(url: str, password: str) -> str:
    """Inject REDIS_PASSWORD into a credential-less Redis URL.

    Uses the `default` username: Redis 6+ rejects the empty-username
    redis://:<password>@ form with WRONGPASS. URLs that already carry
    credentials are returned unchanged.
    """
    parts = urlsplit(url)
    if not password or "@" in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=f"default:{quote(password, safe='')}@{parts.netloc}"))


# Configuration
REDIS_URL = _redis_url_with_password(
    os.getenv("REDIS_URL", "redis://localhost:6379"), os.getenv("REDIS_PASSWORD", "")
)
KNOWLEDGE_MCP_URL = os.getenv("KNOWLEDGE_MCP_URL", "http://localhost:8000/mcp")
BRAVE_MCP_URL = os.getenv("BRAVE_MCP_URL", "http://localhost:8080/mcp")
ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "true").lower() != "false"
# Web search is on by default but requires the Brave sidecar + a Brave API key.
# Set to false to run internal-knowledge-only (e.g. when no Brave key is available).
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() != "false"


# ============================================================================
# Redis Context Provider (sanitizes special chars that break RediSearch queries)
# Mirrors coding-agent/banking-assistant so search text with '?', '*', etc. does
# not trigger RediSearch syntax errors, and only user messages are re-injected.
# ============================================================================

class SanitizedRedisContextProvider(RedisContextProvider):
    """RedisContextProvider that strips RediSearch special characters from search text."""

    async def before_run(self, *, agent: Any, session: Any, context: Any, state: dict[str, Any]) -> None:
        from agent_framework import Message  # noqa: PLC0415
        self._validate_filters()
        input_text = "\n".join(
            msg.text for msg in context.input_messages if msg and msg.text and msg.text.strip()
        )
        if not input_text.strip():
            return
        memories = await self._redis_search(text=input_text, session_id=context.session_id)
        line_separated_memories = "\n".join(
            str(memory.get("content", "")) for memory in memories if memory.get("content")
        )
        if line_separated_memories:
            context.extend_messages(
                self.source_id,
                [Message(role="user", contents=[f"{self.context_prompt}\n{line_separated_memories}"])],
            )

    async def _redis_search(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        sanitized = re.sub(r'[?!@{}\[\]()|~*^=+<>&,]', ' ', text)
        sanitized = ' '.join(sanitized.split())
        if not sanitized.strip():
            return []
        results = await super()._redis_search(sanitized, **kwargs)
        return [r for r in results if r.get("role") == "user"]


# ============================================================================
# MCP client factory
#
# The remote-MCP client class is provided by the Agent Framework. Its exact
# exported name can vary across framework versions, so we resolve it defensively
# and log loudly on failure — a wrong import must not silently yield a
# no-tools agent.
#
# To confirm the real name/signature in the build image, run:
#   python -c "import agent_framework; print([n for n in dir(agent_framework) if 'MCP' in n.upper()])"
#   python -c "import agent_framework, inspect; print(inspect.signature(agent_framework.MCPStreamableHTTPTool.__init__))"
# ============================================================================

def _resolve_mcp_tool_class():
    """Return the Agent Framework Streamable-HTTP MCP client class, or None."""
    import agent_framework  # noqa: PLC0415

    candidates = [
        "MCPStreamableHTTPTool",
        "MCPStreamableHttpTool",
    ]
    for name in candidates:
        cls = getattr(agent_framework, name, None)
        if cls is not None:
            logger.info("Resolved MCP client class: agent_framework.%s", name)
            return cls

    # Fall back to scanning the namespace for any MCP + HTTP tool export.
    for name in dir(agent_framework):
        upper = name.upper()
        if "MCP" in upper and "HTTP" in upper:
            logger.info("Resolved MCP client class by scan: agent_framework.%s", name)
            return getattr(agent_framework, name)

    logger.error(
        "Could not resolve an MCP Streamable-HTTP tool class in agent_framework. "
        "Available MCP-related exports: %s",
        [n for n in dir(agent_framework) if "MCP" in n.upper()],
    )
    return None


def build_knowledge_tools() -> list:
    """Build the remote MCP toolsets the agent exposes to the LLM.

    Returns a list holding the knowledge MCP tool (internal search) and, when
    ENABLE_WEB_SEARCH is set, the Brave Search MCP tool (web search). Returns an
    empty list if the MCP client class cannot be resolved (the agent then starts
    but reports it cannot access its knowledge sources, rather than crashing).
    """
    mcp_tool_cls = _resolve_mcp_tool_class()
    if mcp_tool_cls is None:
        return []

    tools = []

    # approval_mode="never_require": the framework must not block waiting for a
    # human approval in the headless DevUI context. Both MCP servers here are
    # read-only (search / retrieve), so no confirmation gating is needed.
    try:
        knowledge_mcp = mcp_tool_cls(
            name="knowledge",
            url=KNOWLEDGE_MCP_URL,
            approval_mode="never_require",
        )
        logger.info("Knowledge MCP client configured for %s", KNOWLEDGE_MCP_URL)
        tools.append(knowledge_mcp)
    except Exception as exc:  # pragma: no cover - defensive startup guard
        logger.exception("Failed to construct knowledge MCP client for %s: %s", KNOWLEDGE_MCP_URL, exc)

    if ENABLE_WEB_SEARCH:
        try:
            brave_mcp = mcp_tool_cls(
                name="web_search",
                url=BRAVE_MCP_URL,
                approval_mode="never_require",
            )
            logger.info("Brave Search MCP client configured for %s", BRAVE_MCP_URL)
            tools.append(brave_mcp)
        except Exception as exc:  # pragma: no cover - defensive startup guard
            logger.exception("Failed to construct Brave MCP client for %s: %s", BRAVE_MCP_URL, exc)
    else:
        logger.info("Web search disabled (ENABLE_WEB_SEARCH=false) — internal knowledge only.")

    return tools


# ============================================================================
# Agent Creation
# ============================================================================

INSTRUCTIONS = """You are a helpful, professional knowledge assistant for a company.

You answer questions using two sources, each accessed through tools:
  • An INTERNAL knowledge base (tools prefixed 'knowledge'): search_knowledge,
    get_document, list_documents, list_collections. This holds the company's own
    policies, runbooks, product docs, and onboarding material.
  • WEB SEARCH (Brave Search tools, e.g. brave_web_search, brave_news_search):
    live, public information from the internet.

How to work:
1. For questions about company policies, internal processes, products, IT, HR,
   security, or onboarding, ALWAYS search the internal knowledge base FIRST with
   search_knowledge. Ground your answer in what you retrieve — never invent
   internal policy.
2. Cite your internal sources by document title and id (e.g. "PTO Policy
   (HR-001)") so the user can verify. If you need the full text of a document,
   call get_document.
3. Use WEB SEARCH for current events, external facts, or anything not covered by
   the internal knowledge base. Make clear when information comes from the web
   versus internal docs, and include links/sources returned by the search.
4. If the internal search returns nothing relevant, say so plainly, then decide
   whether a web search can help.
5. Be concise and well-organized. When a tool returns an {"error": ...} payload,
   explain the problem to the user plainly rather than guessing.

Sample documents (ids like HR-001) are synthetic demo content. Documents whose id
is a file path (e.g. hr/benefits-guide.pdf) come from files added by the operator.
"""


def create_knowledge_agent():
    """Create the knowledge agent with MCP tools and optional Redis memory."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")
    model = os.getenv("MODEL_NAME", "gpt-4o")

    context_providers = []
    if ENABLE_MEMORY:
        try:
            context_providers.append(
                SanitizedRedisContextProvider(
                    source_id="knowledge_agent_context",
                    redis_url=REDIS_URL,
                    index_name="knowledge_agent_memory",
                    application_id="knowledge_agent_app",
                    agent_id="knowledge_agent",
                    user_id="default_user",
                )
            )
        except Exception as exc:  # pragma: no cover - memory is optional
            logger.warning("Redis memory disabled (could not initialize): %s", exc)

    tools = build_knowledge_tools()

    agent = OpenAIChatCompletionClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
    ).as_agent(
        name="knowledge-assistant",
        instructions=INSTRUCTIONS,
        tools=tools,
        context_providers=context_providers,
    )
    return agent


# Export agent for DevUI discovery
agent = create_knowledge_agent()
