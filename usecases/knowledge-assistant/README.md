# Knowledge Assistant

A conversational knowledge assistant built on **Intel® Agent Toolkit**
(installed by Toolkit Manager release-1.2). It answers
natural-language questions ("how much PTO do I get?", "how do I fix a VPN
certificate error?", "what's the latest news on RAG?") by grounding its answers
in two sources, each exposed through a **Model Context Protocol (MCP)** server:

1. An **internal knowledge base** — retrieval-augmented (RAG) search over the
   Agent Toolkit's shared **PostgreSQL + pgvector** vector database of internal
   documents (policies, runbooks, product docs, onboarding material), embedded
   via LiteLLM.
2. The **public web** — live search via the official **Brave Search MCP server**.

Both run as **sidecar containers** in the same pod as the
agent, which connects to each as a remote MCP client over loopback.

Model calls go to LiteLLM (Intel® Inference Toolkit) — any tool-calling model it
serves, or a router name if you configured one there. Redis (memory) and pgvector
(vector store) are the toolkit's shared services; the assistant consumes them
rather than deploying its own.

> ⚠️ **Demo only.** The internal knowledge base defined in [data.py](./src/mcp_server/data.py) is 100% synthetic sample content
> — no real company data, no PII, no confidential material. You can add your own
> files (PDF, Word, Markdown, text, HTML); see
> [Adding your own documents](#4-adding-your-own-documents). See also
> [Security & limitations](#security--limitations).

---

## 1. Overview

```
  https://knowledge-assistant.<base>  (eg-gateway HTTPRoute, TLS at the gateway)
                    ┌──────────────────────── Pod ──────────────────────────┐
  user (NL query)   │   ┌─────────────────────┐        ┌──────────────────┐ │
  ───────────────►  │   │ knowledge-assistant │  MCP   │  knowledge-mcp   │ │
   DevUI / OpenAI   │   │ (Agent Framework    │ ──────►│  (FastMCP, RAG)  │ │
   Responses @ :8090│   │  + DevUI, :8090)    │ loop-  │  :8000 /mcp      │ │
                    │   │                     │ back   └────────┬─────────┘ │
                    │   │                     │  MCP   ┌────────┴─────────┐ │
                    │   │                     │ ──────►│  brave-mcp       │ │
                    │   └───┬──────────┬──────┘ loop-  │  :8080 /mcp      │ │
                    └───────┼──────────┼──────── back ─┼─────────┬────────┘
            model calls     │          │ memory        │ embeds  │ web search
                  ▼         ▼          ▼               ▼         ▼
        ┌────────────────────────┐  ┌──────────┐  ┌──────────────┐ api.search.brave.com
        │ LiteLLM                │  │ Redis    │  │ pgvector     │ (BRAVE_API_KEY)
        │ litellm.litellm:4000   │  │ (memory, │  │ (agentdb,    │
        │ chat (tool calling)    │  │  shared  │  │  shared      │
        │ + /v1/embeddings       │  │  toolkit)│  │  toolkit)    │
        └────────────────────────┘  └──────────┘  └──────────────┘
                 ── Intel Agent Toolkit shared services (Istio ambient, mTLS) ──
```

- **Agent** (`src/agent.py`, `src/main.py`) — Microsoft Agent Framework agent
  served through DevUI on port `8090`. Exposes both MCP servers' tools to the LLM,
  sends model calls to LiteLLM, and keeps optional text-only conversation memory
  in the shared Redis.
- **Knowledge MCP server** (`src/mcp_server/`) — a FastMCP server exposing RAG
  tools over Streamable HTTP on port `8000`, backed by the toolkit's shared
  **PostgreSQL + pgvector** store (`src/mcp_server/vectorstore.py`). On startup it
  chunks the synthetic sample documents (`src/mcp_server/data.py`) plus any files
  in the docs folder (`src/mcp_server/loader.py`), embeds them through LiteLLM's
  `/v1/embeddings` endpoint, and stores the vectors in the shared
  database (table `knowledge_chunks`); queries are embedded the same way and
  ranked by cosine distance. Only new or changed documents are embedded on each
  start; documents that were removed are deleted from the table.
- **pgvector (shared)** — the Agent Toolkit's PostgreSQL 16 + pgvector service
  (`pgvector.pgvector.svc.cluster.local:5432`, database `agentdb`). Consumed as an
  external dependency, exactly like Redis — the chart does **not** deploy a
  database.
- **Brave Search MCP server** — the official
  [`@brave/brave-search-mcp-server`](https://github.com/brave/brave-search-mcp-server)
  run over Streamable HTTP on port `8080` (`src/Dockerfile.brave`). Provides web,
  news, and image search. Requires a Brave Search API key.
- **Topology** — both MCP servers are **sidecars** in the agent's pod; the agent
  reaches them at `http://localhost:8000/mcp` and `http://localhost:8080/mcp`.
  See [Topology](#topology-sidecars) for the rationale and alternatives.

### Internal knowledge tools (knowledge MCP)

| Tool | Purpose |
|---|---|
| `search_knowledge(query, top_k?, collection?)` | Semantic (vector) search over the internal docs; returns ranked passages with sources |
| `get_document(doc_id)` | Full text + metadata for one document (e.g. `HR-001` or `hr/benefits-guide.pdf`); very long documents are truncated |
| `list_documents(collection?)` | List document metadata, optionally by collection |
| `list_collections()` | List collections (the sample ones plus one per docs subfolder) and their doc counts |

### Web search tools (Brave Search MCP)

`brave_web_search`, `brave_news_search`, `brave_image_search`,
`brave_video_search`, `brave_local_search`, `brave_summarizer`, and more — see the
[Brave Search MCP docs](https://github.com/brave/brave-search-mcp-server).

Demo collections: `hr`, `it`, `security`, `product`, `onboarding` (10 sample
documents such as `HR-001` PTO Policy, `IT-001` VPN Runbook, `PROD-001` product
overview). Your own files add to these collections or create new ones; see
[Adding your own documents](#4-adding-your-own-documents).

---

## 2. Prerequisites

- **Intel® Agent Toolkit** installed with Toolkit Manager release-1.2 (see
  [Toolkit Manager setup](../../docs/toolkit-manager-setup.md)), with these
  switches in `env/<env>/config.agent-toolkit.yaml`:
  - `pgvector_enabled: true` — the shared PostgreSQL + pgvector service
    (`pgvector.pgvector.svc.cluster.local:5432`, database `agentdb`), with the
    connection string in the `pgvector-credentials` secret (namespace `pgvector`,
    key `DATABASE_URL`). `deploy.sh` reads it automatically; or pass `PGVECTOR_URL`.
  - `redis_stack_enabled: true` — the shared Redis (conversation memory; set
    `ENABLE_MEMORY=false` to skip). It requires a password, which `deploy.sh`
    reads by reference from `redis/redis-stack-server-credentials`.
  - `knowledge-assistant` in `litellm_allowed_namespaces` — LiteLLM only admits
    mesh callers from listed namespaces (`PATCH_LITELLM_AUTHZ=true` patches the
    policy as a stop-gap).
  - `llm_services_autodeploy_models` includes a **tool-calling chat model** (the
    default `qwen2-5-coder-14b` works) and the **`bge-base-en` embedding model**
    used for RAG. Without tool calling the model describes the tool call in plain
    text instead of making it.
- A **Brave Search API key** for web search — get one at
  [api-dashboard.search.brave.com](https://api-dashboard.search.brave.com/).
  Optional: without one, `deploy.sh` deploys internal-knowledge-only.
- The pod must reach `api.search.brave.com` (directly or via `http(s)_proxy`).
- `kubectl` + `helm` for cluster deploy, or Docker + Docker Compose for local dev
  (the compose stack bundles its own pgvector + Redis for offline use).

---

## 3. Quick Start

`deploy.sh` does everything: auto-discovers the base domain (from `eg-gateway`),
the LiteLLM key, Langfuse keys and the pgvector URL; builds all three images
(agent + knowledge MCP + Brave MCP); prepares the namespace (Pod Security
`restricted` + Istio ambient); copies the Redis password secret; and installs the
Helm chart with an HTTPRoute on `knowledge-assistant.<base_domain_name>`. TLS is
terminated at the gateway, so no certificate is needed. On containerd clusters it
builds with nerdctl into the `k8s.io` namespace so the kubelet can use the images
locally (`pullPolicy: IfNotPresent`).

From the repo root, using the shared `config.local.env`:

```bash
cp config.env config.local.env && $EDITOR config.local.env   # MODEL_NAME, BRAVE_API_KEY, ...
set -a && source config.local.env && set +a
./deploy-usecase.sh knowledge-assistant --mode kubernetes
```

Or directly (it prompts for anything it can't discover):

```bash
cd usecases/knowledge-assistant
./deploy.sh
```

To include your own documents, put them in `src/docs/<collection>/` before running
`deploy.sh`; see [Adding your own documents](#4-adding-your-own-documents).

The proxy variables (`http_proxy`, `https_proxy`, `no_proxy`) are only needed if
the cluster reaches the internet through a proxy. They are used for the image
builds and passed to the Brave sidecar at runtime. `config.local.env` holds
secrets, so don't commit it.

`MODEL_NAME` is shared by every blueprint. To give this one a different model (for
example a router name you configured in LiteLLM), set `KNOWLEDGE_MODEL_NAME`.

To run **internal-knowledge-only** (no Brave sidecar, no API key needed):

```bash
ENABLE_WEB_SEARCH=false ./deploy.sh
```

To skip the build (e.g. images already pushed to a registry):

```bash
SKIP_BUILD=true \
  AGENT_IMAGE=myregistry/knowledge-assistant:1.0 \
  MCP_IMAGE=myregistry/knowledge-mcp:1.0 \
  BRAVE_IMAGE=myregistry/knowledge-brave-mcp:1.0 \
  ./deploy.sh
```

Then open the DevUI (or port-forward) and chat:

```bash
# Through the platform gateway (deploy.sh prints the gateway IP for /etc/hosts;
# the CA is env/<env>/logs/ai-solutions-ca.crt):
#   https://knowledge-assistant.<base_domain_name>

# Or locally:
kubectl port-forward -n knowledge-assistant svc/knowledge-assistant 8090:8090
#   → http://localhost:8090   (enter the DevUI token printed by deploy.sh)
```

> **Gateway auth.** The gateway-wide `SecurityPolicy gateway-auth` requires a
> LiteLLM key on every route. The chart ships an empty route-level SecurityPolicy
> (`httpRoute.bypassGatewayAuth: true`) because DevUI enforces its own
> `DEVUI_AUTH_TOKEN`.

> **Redis auth is automatic.** The Agent Toolkit's shared Redis requires a
> password. `deploy.sh` copies only the `REDIS_PASSWORD` key of
> `redis/redis-stack-server-credentials` into the release namespace and passes
> only the secret's name to Helm, never the password. The agent injects it into
> `REDIS_URL` as `redis://default:<password>@host:port`; `deploy.sh` strips any
> credentials from a `REDIS_URL` you pass in.

---

## 4. Adding your own documents

The knowledge MCP server loads every file in its **docs folder**, extracts the text,
splits it into chunks, embeds them, and serves them through the same tools as the
sample documents (`search_knowledge`, `get_document`, `list_documents`,
`list_collections`).

### Folder layout

Each **first-level subfolder is a collection**. Deeper subfolders are allowed and
stay part of their top-level collection. Files directly in the docs folder go into
the `general` collection.

```
usecases/knowledge-assistant/src/docs/
├── hr/
│   ├── benefits-guide.pdf        → collection "hr", doc_id "hr/benefits-guide.pdf"
│   └── parental-leave.md
├── it/
│   └── runbooks/
│       └── vpn-setup.docx        → collection "it", doc_id "it/runbooks/vpn-setup.docx"
├── engineering/                  → new collections are just new folders
│   └── release-process.txt
└── company-faq.html              → collection "general"
```

| Type | Extensions | Notes |
|---|---|---|
| PDF | `.pdf` | Text layer only. Scanned PDFs (images of text) have none and are skipped, because there is no OCR. |
| Word | `.docx` | Paragraphs and tables. Old binary `.doc` isn't supported; save it as `.docx` first. |
| Text | `.txt`, `.md`, `.markdown`, `.rst`, `.csv` | Read as UTF-8. |
| HTML | `.html`, `.htm` | Visible text only; scripts and styles are dropped. |

- **Title:** the PDF or Word document title if set, else the first `# Heading` of a
  Markdown/text file, else the file name (`benefits-guide.pdf` → "Benefits Guide").
- **Document id:** the file's path inside the docs folder, e.g.
  `hr/benefits-guide.pdf`. The agent cites documents by title and id.
- **Skipped files:** hidden files and folders (starting with `.`), unsupported
  extensions, and files that can't be read. Each is logged with a warning; one bad
  file never stops the server from starting.

> **Keep private documents out of git.** `src/docs/` has a `.gitignore` that
> ignores everything in it, so files you add there aren't committed by accident.

### Kubernetes: bake the files into the image (default)

`deploy.sh` copies `src/docs/` into the knowledge MCP image when it builds it.

```bash
cd usecases/knowledge-assistant
mkdir -p src/docs/hr src/docs/it
cp ~/policies/*.pdf src/docs/hr/
cp ~/runbooks/*.docx src/docs/it/

set -a && source ../../config.local.env && set +a
./deploy.sh            # rebuilds the image with the new files and redeploys
```

Repeat this whenever the documents change. Deploying with `SKIP_BUILD=true` keeps
the old image, so new files are only picked up after a rebuild.

### Kubernetes: keep the files on a volume (no rebuild)

To add documents without rebuilding, store them on a PersistentVolumeClaim
mounted at `/app/docs`. This needs a StorageClass in the cluster
(`kubectl get storageclass`). The volume **replaces** the baked-in `src/docs/`
files, so copy all your documents onto it.

```bash
# 1. Deploy with the docs volume (set these in config.local.env, or inline).
export DOCS_PERSISTENCE=true
export DOCS_STORAGE_CLASS=""          # empty = the cluster's default StorageClass
./deploy.sh

# 2. Copy files in. A whole local folder becomes a new collection; single files
#    go into an existing one.
POD=$(kubectl get pod -n knowledge-assistant \
  -l app.kubernetes.io/instance=knowledge-assistant -o jsonpath='{.items[0].metadata.name}')
kubectl cp ./my-docs/hr "${POD}:/app/docs/hr" -n knowledge-assistant -c knowledge-mcp
kubectl cp ./new-policy.pdf "${POD}:/app/docs/hr/new-policy.pdf" -n knowledge-assistant -c knowledge-mcp
kubectl exec -n knowledge-assistant "${POD}" -c knowledge-mcp -- ls -R /app/docs   # check

# 3. Restart the pod so the server loads and embeds the new files.
kubectl rollout restart deployment/knowledge-assistant -n knowledge-assistant
kubectl rollout status  deployment/knowledge-assistant -n knowledge-assistant
```

To remove a document, delete it from the volume and restart:

```bash
kubectl exec -n knowledge-assistant "${POD}" -c knowledge-mcp -- rm /app/docs/hr/old-policy.pdf
kubectl rollout restart deployment/knowledge-assistant -n knowledge-assistant
```

The PVC (`knowledge-assistant-docs`) is kept on `helm uninstall`, so your files
survive a reinstall. Delete it with `kubectl delete pvc` when you no longer need
it. With the volume enabled, the deployment uses the `Recreate` strategy, so the
pod is briefly unavailable during a restart.

### Docker Compose

Compose mounts `./src/docs` into the container, so no rebuild is needed:

```bash
cp ~/policies/*.pdf src/docs/hr/
docker compose restart knowledge-mcp
```

### What happens on startup

The server compares the files with what is already stored in pgvector, using a
content hash per document:

- **New or changed** documents are embedded and stored.
- **Deleted** documents are removed from the vector store.
- **Unchanged** documents are skipped, so a restart doesn't re-embed everything.

The MCP port only opens once this sync is done, so the first start after adding a
lot of documents takes longer. Watch progress with:

```bash
kubectl logs -n knowledge-assistant deploy/knowledge-assistant -c knowledge-mcp -f
#   Loaded 42 document(s) from /app/docs.
#   Embedded hr/benefits-guide.pdf (18 chunks).
#   KnowledgeIndex synced: 42 document(s) embedded (610 chunks), 0 removed, 10 unchanged.
```

Kubernetes allows the sync up to 15 minutes (`mcp.knowledge.startupProbe` in
`values.yaml`), and `deploy.sh` waits up to 10 minutes for the pod (`HELM_TIMEOUT`).
Raise both for very large document sets.

To serve **only your documents**, drop the 10 synthetic samples with
`INCLUDE_SAMPLE_DOCS=false` (in `config.local.env`, or exported before
`deploy.sh` / `docker compose up`). Their embeddings are deleted on the next start.

---

## 5. Configuration

All configuration is read from environment variables (wired via the Helm
`configmap.yaml` / `secret.yaml`).

### Agent container

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | `http://litellm.litellm.svc.cluster.local:4000/v1` | LiteLLM OpenAI-compatible URL |
| `OPENAI_API_KEY` | — (required) | LiteLLM API key (`deploy.sh` falls back to the master key; prefer a virtual key) |
| `MODEL_NAME` | — (required) | Model the agent sends to LiteLLM (e.g. `qwen2-5-coder-14b`); `deploy.sh` takes it from `KNOWLEDGE_MODEL_NAME`, else `MODEL_NAME`. **Must support tool calling.** |
| `KNOWLEDGE_MCP_URL` | `http://localhost:8000/mcp` | Internal knowledge MCP endpoint (loopback in sidecar mode) |
| `BRAVE_MCP_URL` | `http://localhost:8080/mcp` | Brave Search MCP endpoint (loopback in sidecar mode) |
| `ENABLE_WEB_SEARCH` | `true` | Set `false` to disable the web-search tools (internal knowledge only) |
| `REDIS_URL` | in-cluster redis | Redis URL for conversation memory, without credentials. |
| `REDIS_PASSWORD` | — | Redis password, injected into `REDIS_URL` as the `default` user. In Kubernetes it comes from the `redis-stack-server-credentials` secret. |
| `ENABLE_MEMORY` | `true` | Set `false` to disable Redis memory |
| `DEVUI_HOST` | `0.0.0.0` | Host DevUI binds to |
| `DEVUI_AUTH_TOKEN` | auto-generated | DevUI login token |
| `AUTO_OPEN_BROWSER` | `false` | Whether DevUI opens a browser |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | auto-discovered | Optional tracing; `deploy.sh` reads `langfuse/langfuse-secrets` (see [observability](../../docs/observability-langfuse.md)) |

### Knowledge MCP container

| Variable | Default | Description |
|---|---|---|
| `MCP_HOST` | `0.0.0.0` | Interface the FastMCP server binds to |
| `MCP_PORT` | `8000` | Port for the Streamable HTTP transport |
| `OPENAI_BASE_URL` | in-cluster LiteLLM | LiteLLM base URL used for `/v1/embeddings` |
| `OPENAI_API_KEY` | — (required) | LiteLLM API key for embeddings |
| `EMBEDDING_MODEL` | `bge-base-en` | Embedding model id served by LiteLLM (Inference Toolkit name for `BAAI/bge-base-en-v1.5`) |
| `EMBED_BATCH` | `16` | Chunks embedded per LiteLLM request during the startup sync |
| `DOCS_DIR` | `/app/docs` | Folder your own documents are loaded from (one subfolder per collection) |
| `INCLUDE_SAMPLE_DOCS` | `true` | Set `false` to serve only your own documents and drop the 10 synthetic samples |
| `CHUNK_SIZE` | `1000` | Target chunk length, in characters, for loaded files |
| `CHUNK_OVERLAP` | `150` | Characters repeated between consecutive chunks when a long passage is split |
| `MAX_DOCUMENT_CHARS` | `20000` | Most characters of body text `get_document` returns; longer documents are truncated |
| `PGVECTOR_URL` | — (from Secret) | Full connection URL for the shared pgvector (the toolkit's `DATABASE_URL`). Overrides the discrete `PG*` vars when set. |
| `PGHOST` | `pgvector.pgvector.svc.cluster.local` | Shared pgvector host (used when `PGVECTOR_URL` is unset) |
| `PGPORT` | `5432` | pgvector port |
| `PGDATABASE` | `agentdb` | pgvector database name |
| `PGUSER` | `agentuser` | pgvector user |
| `PGPASSWORD` | — (from Secret) | pgvector password |

The knowledge MCP server reuses the agent's `OPENAI_BASE_URL` / `OPENAI_API_KEY`
for embeddings and connects to the **shared** pgvector — the chart deploys no
database. Point it at your own Postgres (with the `vector` extension) by
overriding `PGVECTOR_URL` or the `pgvector.*` Helm values.

### Brave Search MCP container

| Variable | Default | Description |
|---|---|---|
| `BRAVE_API_KEY` | — (required if web search enabled) | Brave Search API key |
| `BRAVE_MCP_TRANSPORT` | `http` | Transport mode (must be `http` for the sidecar) |
| `BRAVE_MCP_HOST` | `0.0.0.0` | Interface the server binds to (must be `0.0.0.0` in a container) |
| `BRAVE_MCP_PORT` | `8080` | Port for the Streamable HTTP transport |

---

## 6. API Reference

DevUI exposes the OpenAI **Responses** API (`POST /v1/responses`), not
`/v1/chat/completions`. Address the agent by its entity id (from `/v1/entities`)
in `metadata.entity_id`; all `/v1/*` routes require the `DEVUI_AUTH_TOKEN` bearer
token. Send natural language; the LLM decides which MCP tools (internal search vs.
web search) to call.

```bash
AGENT=http://localhost:8090            # or https://knowledge-assistant.<base_domain_name>
AUTH="Authorization: Bearer ${DEVUI_AUTH_TOKEN}"

# Health (no auth; TCP-backed — DevUI may not expose a JSON /health)
curl -s "${AGENT}/health"

# Resolve the agent entity id
EID=$(curl -s "${AGENT}/v1/entities" -H "${AUTH}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['entities'][0]['id'])")

# Ask a question — triggers an internal knowledge-base search
curl -s -X POST "${AGENT}/v1/responses" -H "${AUTH}" -H "Content-Type: application/json" \
  -d "{\"input\": \"How many PTO days do full-time employees get?\",
       \"metadata\": {\"entity_id\": \"${EID}\"}, \"stream\": false}"
```

---

## 7. Running Examples

```bash
# Python (standard library only): runs internal-knowledge and web-search queries
AGENT_URL=http://localhost:8090 DEVUI_AUTH_TOKEN=<token> python3 examples/knowledge_client.py

# curl-based equivalents
DEVUI_AUTH_TOKEN=<token> ./examples/knowledge_client.sh http://localhost:8090
```

---

## 8. Local Development

The compose stack bundles its own pgvector + Redis so it runs offline, without the
Agent Toolkit's shared services; it still needs a reachable LiteLLM for chat and
embeddings. On a Toolkit Manager stack that is `https://litellm.<base_domain_name>/v1`
(resolve it to the eg-gateway IP and trust `env/<env>/logs/ai-solutions-ca.crt`).

Compose reads the **same** `OPENAI_BASE_URL` / `OPENAI_API_KEY` variables as
`deploy.sh` and the Helm chart, so one set of exports covers both local and cluster
deploys. (`LITELLM_URL` / `LITELLM_API_KEY` are still honoured as fallbacks; the
repo-root `./deploy-usecase.sh knowledge-assistant --mode docker` sets them.)

```bash
export OPENAI_BASE_URL=https://litellm.solutions.ai/v1   # same names deploy.sh / the chart use
export OPENAI_API_KEY=sk-your-key
export MODEL_NAME=qwen2-5-coder-14b                      # must support tool calling
export EMBEDDING_MODEL=bge-base-en                       # must be served by LiteLLM
export BRAVE_API_KEY=your-brave-search-api-key           # for web search

docker compose up --build
#   pgvector             → :5432  (bundled Postgres vector store — dev only)
#   knowledge-mcp        → :8000  (internal RAG MCP server; embeds corpus on startup)
#   brave-mcp            → :8080  (Brave Search MCP server)
#   knowledge-assistant  → :8090  (agent + DevUI)
#   redis-stack          → :6379
```

> The `knowledge-mcp` container embeds the whole corpus via LiteLLM on first
> start, so it stays "starting" until Postgres is up and seeding completes. The
> compose Postgres volume is persistent, so later starts only embed new or changed
> documents. `./src/docs` is mounted into the container; after adding files, run
> `docker compose restart knowledge-mcp`.

You can also run the knowledge MCP server on its own for tool debugging (it needs
a reachable Postgres/pgvector and the LiteLLM embeddings endpoint). Point it at any
Postgres with the `vector` extension — the bundled compose one, or the toolkit's
shared pgvector via `PGVECTOR_URL`:

```bash
cd src
export OPENAI_BASE_URL=https://litellm.solutions.ai/v1 OPENAI_API_KEY=sk-your-key
export EMBEDDING_MODEL=bge-base-en
export PGHOST=localhost PGPORT=5432 PGDATABASE=knowledge PGUSER=knowledge PGPASSWORD=knowledge
MCP_PORT=8000 python -m mcp_server.server
```

---

## 9. Uninstall

### Kubernetes

Remove the Helm release, then the namespace. Deleting the namespace also removes
everything `deploy.sh` created there outside the chart: the copy of
`redis-stack-server-credentials`. If you added `knowledge-assistant` to
`litellm_allowed_namespaces`, you can remove it there too.

```bash
helm uninstall knowledge-assistant -n knowledge-assistant
kubectl delete namespace knowledge-assistant
```

> **Documents volume.** With `DOCS_PERSISTENCE=true`, the `knowledge-assistant-docs`
> PVC is annotated `helm.sh/resource-policy: keep`, so `helm uninstall` leaves it
> in place and a reinstall picks your documents back up. Deleting the namespace removes it. To
> uninstall but keep the documents, run only the `helm uninstall` step.

The shared Agent Toolkit services (LiteLLM, Redis, pgvector) are not
touched. To also clear the data this use case wrote to them:

```bash
# Embedded document chunks in the shared pgvector database
PG_POD=$(kubectl get pod -n pgvector -o name | head -1)
kubectl exec -n pgvector "${PG_POD}" -- \
  psql -U agentuser -d agentdb -c 'DROP TABLE IF EXISTS knowledge_chunks;'

# Conversation memory in the shared Redis (index + its documents)
REDISCLI_AUTH=$(kubectl get secret redis-stack-server-credentials -n redis \
  -o jsonpath='{.data.REDIS_PASSWORD}' | base64 -d)
kubectl exec -n redis statefulset/redis-stack-server -- \
  env REDISCLI_AUTH="${REDISCLI_AUTH}" redis-cli FT.DROPINDEX knowledge_agent_memory DD
unset REDISCLI_AUTH
```

Only do this if no other deployment uses the same data: `knowledge_chunks` holds
every document this use case embedded. Finally, remove the locally built
images if you no longer need them:

```bash
sudo nerdctl --namespace k8s.io rmi knowledge-assistant:latest knowledge-mcp:latest knowledge-brave-mcp:latest
# or, with Docker: docker rmi knowledge-assistant:latest knowledge-mcp:latest knowledge-brave-mcp:latest
```

### Docker Compose

From `usecases/knowledge-assistant/`:

```bash
docker compose down      # stop and remove the containers
docker compose down -v   # ...and also delete the bundled pgvector and Redis data volumes
```

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Agent replies but never searches the knowledge base | `MODEL_NAME` doesn't support tool/function calling — pick a tool-capable instruct model. |
| Startup log: "Could not resolve an MCP Streamable-HTTP tool class" | The Agent Framework MCP client class name differs in your installed version. Run `python -c "import agent_framework; print([n for n in dir(agent_framework) if 'MCP' in n.upper()])"` and update `_resolve_mcp_tool_class()` / pin the version in `requirements.txt`. |
| Web search never happens / Brave errors | Confirm `BRAVE_API_KEY` is set and valid, `ENABLE_WEB_SEARCH=true`, and the `brave-mcp` container is Ready. Ensure `no_proxy` includes the MCP hosts. |
| Image build fails at `load metadata for docker.io/...` with `failed to fetch anonymous token ... i/o timeout` | The build can't reach Docker Hub through your proxy. Two processes need it: (1) **buildkitd**, which reads the proxy from its systemd unit. Add `/etc/systemd/system/buildkit.service.d/http-proxy.conf` with `Environment="HTTP_PROXY=..."`, `HTTPS_PROXY` and `NO_PROXY`, then run `systemctl daemon-reload` and `systemctl restart buildkit`. (2) **nerdctl**, which fetches the registry token itself; `deploy.sh` passes your `http_proxy`/`https_proxy` through `sudo`, so export them first (`set -a && source config.local.env && set +a`). Check that only one buildkitd is running (`pgrep -a buildkitd`). |
| Web search tool call hangs or returns "fetch failed" | The `brave-mcp` sidecar can't reach `api.search.brave.com`, usually because outbound traffic must go through a corporate proxy. Export `http_proxy`/`https_proxy` before running `deploy.sh` (or set `mcp.brave.httpProxy`/`httpsProxy`); the chart passes them to the sidecar with `NODE_USE_ENV_PROXY=1`, which Node's built-in fetch needs to honour them. This needs Node >= 22.21 in the Brave image; check with `kubectl exec ... -c brave-mcp -- node -v`. |
| Agent can't reach an MCP server | Confirm the sidecar container is Ready and the URL is `http://localhost:8000/mcp` / `http://localhost:8080/mcp` (sidecar) or the compose DNS name. |
| Internal search returns nothing relevant | The demo corpus is small (10 docs). Ask about PTO, remote work, expenses, VPN, passwords/MFA, data classification, incidents, the Acme Analytics product, or onboarding, or [add your own documents](#4-adding-your-own-documents). |
| A file I added doesn't show up in `list_documents` | Check the `knowledge-mcp` logs for a "Skipping …" line: unsupported type, unreadable file, or no extractable text (scanned PDFs need OCR first). With the default baked-in docs, the image must be rebuilt (`./deploy.sh` without `SKIP_BUILD`); with `DOCS_PERSISTENCE=true` or Compose, restart the pod/container. |
| `deploy.sh` fails with `context deadline exceeded` after adding many documents | Embedding the new files took longer than helm waited. The pod usually keeps going and becomes Ready; check `kubectl logs ... -c knowledge-mcp`. Raise `HELM_TIMEOUT` (and `mcp.knowledge.startupProbe.failureThreshold` if the container gets restarted). |
| `knowledge-mcp` never becomes Ready | It blocks on startup until pgvector is reachable and the corpus is embedded. Check its logs for "Waiting for Postgres" (shared pgvector not reachable — verify `PGVECTOR_URL`/`PG*` and that the toolkit's pgvector is up) or embedding errors. |
| Startup log: embeddings HTTP 400 / model not found | `EMBEDDING_MODEL` isn't served by LiteLLM, or `OPENAI_BASE_URL`/`OPENAI_API_KEY` are wrong. Add `bge-base-en` to `llm_services_autodeploy_models` (or set `EMBEDDING_MODEL` to an embedding model you serve). |
| Every LLM / embedding call fails with `RBAC: access denied` (HTTP 403) | The namespace isn't in LiteLLM's `litellm_allowed_namespaces`. `deploy.sh` warns about this; add it and re-run `./es_auto_installer.sh install agent-toolkit`, or redeploy with `PATCH_LITELLM_AUTHZ=true` as a stop-gap. |
| `deploy.sh` stops with "No pgvector connection" | The shared pgvector secret wasn't found. Set `pgvector_enabled: true` in `config.agent-toolkit.yaml` and reinstall the agent-toolkit, set `PGVECTOR_NAMESPACE`, or pass `PGVECTOR_URL` explicitly. |
| Every chat fails with "Authentication required" (trace shows `redis.exceptions.AuthenticationError`) | The agent isn't sending the Redis password. This is a Redis error, not the DevUI login. Check that `deploy.sh` printed "Copied redis/redis-stack-server-credentials"; if it warned instead, confirm the secret exists (`kubectl get secret redis-stack-server-credentials -n redis`) and redeploy. Or set `ENABLE_MEMORY=false` to run without memory. |
| Chat fails with "invalid username-password pair or user is disabled" (`WRONGPASS`) | Redis rejected the password. Make sure `REDIS_URL` has no credentials of its own and redeploy so `redis-stack-server-credentials` is copied again. After a password rotation, also restart the pod (`kubectl rollout restart deployment/knowledge-assistant -n knowledge-assistant`): it reads the secret only at startup. |
| `/health` returns 404 | DevUI may not expose a JSON health route; liveness/readiness use a TCP check on 8090. |

---

## Topology (sidecars)

Both MCP servers run as additional containers in the agent's pod. The knowledge
server holds a read-only corpus synced on start, so there is no scaling or
availability reason to run it separately, and co-location removes a Service,
cross-pod DNS, and a network hop. The Brave server is a thin stateless proxy to
the Brave Search API, so it co-locates cleanly too.

The vector store is **not** a sidecar: the chart consumes the Agent Toolkit's
shared pgvector as an external dependency, exactly like the shared Redis. Point it
elsewhere by overriding `PGVECTOR_URL` (or the discrete `pgvector.*` Helm values)
to any Postgres with the `vector` extension.

| | Sidecars (this usecase) | Separate deployments |
|---|---|---|
| Networking | loopback, no Service | needs Service + in-cluster DNS |
| Scaling | scales 1:1 with the agent | independent; can front many agents |
| Reuse | this agent only | reusable by other clients |
| Complexity | one chart, one pod | multiple deployments |

To run an MCP server standalone instead, deploy its image as its own Deployment +
Service and point `KNOWLEDGE_MCP_URL` / `BRAVE_MCP_URL` at its Service DNS.

---

## Security & limitations

- **Synthetic sample data by default.** Out of the box the knowledge base is a set
  of fake documents embedded into the shared pgvector (`agentdb`, table
  `knowledge_chunks`) on process start. It is read-only — there are no mutating
  tools; documents are added through the docs folder, not through chat.
- **Your documents are readable by every user.** Files added to the docs folder are
  stored in plain text in the shared pgvector and returned to anyone who has the
  DevUI token. There is no per-document access control, so only add documents that
  everyone with access may read. With the default baked-in docs, the files are also
  inside the knowledge MCP image, so keep that image in a private registry. For
  production, connect a real document store with permissions in place of the docs
  folder, point the chart at your own managed Postgres via `PGVECTOR_URL` if you
  don't want to reuse the toolkit's, and use your chosen embedding model.
- **No authentication / authorization on the MCP servers.** Any caller with pod
  access can query any document or run web searches. This is intentional for a
  demo. A production version would require per-user auth (e.g. OIDC) and would
  scope retrieval to what the authenticated principal is allowed to see.
- **The Brave HTTP endpoint is unauthenticated** and is only ever bound to pod
  loopback / the compose network — never exposed via a Service or Ingress. Keep it
  that way. Your `BRAVE_API_KEY` is stored in a Kubernetes Secret.
- **Web results are external content.** Treat anything returned by web search as
  untrusted input; the agent should attribute and not blindly execute it.
