#!/usr/bin/env bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# examples/knowledge_client.sh
# curl-based examples for the Knowledge Assistant (Agent Framework DevUI).
#
# DevUI exposes the OpenAI *Responses* API (POST /v1/responses), not
# /v1/chat/completions. All /v1/* routes require a bearer token.
#
# Usage:
#   DEVUI_AUTH_TOKEN=<token> ./knowledge_client.sh [AGENT_URL]
# Default AGENT_URL: http://localhost:8090
set -euo pipefail

AGENT="${1:-http://localhost:8090}"
TOKEN="${DEVUI_AUTH_TOKEN:-}"
[[ -n "$TOKEN" ]] || { echo "Set DEVUI_AUTH_TOKEN (printed by deploy.sh)."; exit 1; }
AUTH=(-H "Authorization: Bearer ${TOKEN}")

echo "=== Knowledge Assistant examples ==="
echo "Target: $AGENT"
echo ""

# ── Health check (no auth) ────────────────────────────────────────────────────
echo "--- Health check ---"
curl -s "${AGENT}/health" | python3 -m json.tool || echo "(health endpoint not available)"
echo ""

# ── Resolve the agent entity id ───────────────────────────────────────────────
EID=$(curl -s "${AGENT}/v1/entities" "${AUTH[@]}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['entities'][0]['id'])")
echo "Entity: $EID"
echo ""

ask() {
  local prompt="$1"
  curl -s -X POST "${AGENT}/v1/responses" "${AUTH[@]}" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys;print(json.dumps({'input':sys.argv[1],'metadata':{'entity_id':sys.argv[2]},'stream':False}))" "$prompt" "$EID")" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('status') and d['status'] != 'completed':
    print('[status:%s] %s' % (d['status'], d.get('error')))
for it in d.get('output', []):
    if it.get('type') == 'message':
        for c in it.get('content', []):
            if c.get('type') in ('output_text', 'text'):
                print(c.get('text'))
"
}

# ── Internal HR policy (triggers search_knowledge over the 'hr' collection) ──
echo "--- PTO policy (internal knowledge base) ---"
ask "How many PTO days do full-time employees get, and how much can I carry over?"
echo ""

# ── Internal IT runbook (triggers search_knowledge over the 'it' collection) ─
echo "--- VPN certificate error (internal knowledge base) ---"
ask "I can't connect to the VPN and I get a certificate error. What should I do?"
echo ""

# ── Internal product docs (triggers search_knowledge over 'product') ─────────
echo "--- Acme Analytics support tiers (internal knowledge base) ---"
ask "What are the support tiers and uptime SLAs for the Acme Analytics Platform?"
echo ""

# ── Web search (triggers Brave Search MCP tools) ─────────────────────────────
echo "--- Latest RAG news (web search via Brave) ---"
ask "Search the web for the latest news on retrieval-augmented generation."
echo ""
