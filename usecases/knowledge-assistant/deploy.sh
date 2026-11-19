#!/usr/bin/env bash

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# deploy.sh — One-click deployer for the Knowledge Assistant use case
#
# Deploys the Knowledge Assistant (agent + internal knowledge/RAG MCP sidecar +
# Brave Search web MCP sidecar) onto Intel® Agent Toolkit (installed by Toolkit
# Manager release-1.2) using Helm.
#
# Prerequisites:
#   • Intel Agent Toolkit is installed with Toolkit Manager (see ../../docs/toolkit-manager-setup.md)
#     with pgvector_enabled: true and 'knowledge-assistant' in litellm_allowed_namespaces
#   • kubectl is configured and pointing at your cluster (env/<env>/kubeconfig.yaml)
#   • helm is installed
#   • An embedding model is served by LiteLLM (default: bge-base-en)
#
# Usage:
#   ./deploy.sh
#
# Environment variables (all optional — auto-discovered from the cluster or prompted):
#   CLUSTER_URL           Toolkit Manager base_domain_name, e.g. solutions.ai (from eg-gateway)
#   OPENAI_API_KEY        LiteLLM key (default: litellm-master-key; prefer a virtual key)
#   MODEL_NAME            Model ID in LiteLLM (must support tool calling)
#   KNOWLEDGE_MODEL_NAME  Overrides MODEL_NAME for this use case only (e.g. a router name)
#   EMBEDDING_MODEL       Embedding model ID in LiteLLM (default: bge-base-en)
#   OPENAI_BASE_URL       Override LiteLLM URL (default: in-cluster litellm.litellm:4000)
#   PGVECTOR_URL          Shared pgvector DATABASE_URL (default: pgvector/pgvector-credentials)
#   PGVECTOR_NAMESPACE    Namespace of the pgvector-credentials secret (default: pgvector)
#   REDIS_URL             Override Redis URL (credentials are stripped; the password is
#                         read by reference from the copied redis-stack-server-credentials)
#   LANGFUSE_*            Langfuse keys/host (default: langfuse/langfuse-secrets)
#   BRAVE_API_KEY         Brave Search API key (empty = internal knowledge only)
#   ENABLE_WEB_SEARCH     "false" to skip the Brave web-search sidecar (default: true)
#   INCLUDE_SAMPLE_DOCS   "false" to serve only your own files in src/docs/ (default: true)
#   DOCS_PERSISTENCE      "true" to keep documents on a PVC (added with kubectl cp)
#                         instead of baking src/docs/ into the image (default: false)
#   DOCS_STORAGE_CLASS    StorageClass for that PVC (default: cluster default)
#   HELM_TIMEOUT          How long helm waits for Ready (default: 10m; raise it for large docs)
#   NAMESPACE             Kubernetes namespace (default: knowledge-assistant)
#   INGRESS_HOST          HTTPRoute hostname (default: knowledge-assistant.<CLUSTER_URL>)
#   PATCH_LITELLM_AUTHZ   true = add NAMESPACE to LiteLLM's AuthorizationPolicy directly
#   DEVUI_AUTH_TOKEN      Auth token for DevUI web interface (auto-generated if unset)
#   http_proxy/https_proxy/no_proxy
#                         Corporate proxy, if any. Used for the image builds and passed
#                         to the Brave sidecar at runtime (it calls api.search.brave.com)
#   SKIP_BUILD            Set to "true" to skip building images (use pre-built ones)
#   CONTAINER_TOOL        Override image builder: "nerdctl" or "docker" (auto-detected)
#   AGENT_IMAGE           Agent image ref (default: knowledge-assistant:latest)
#   MCP_IMAGE             Knowledge MCP image ref (default: knowledge-mcp:latest)
#   BRAVE_IMAGE           Brave MCP image ref (default: knowledge-brave-mcp:latest)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="${SCRIPT_DIR}/helm-chart"
SRC_DIR="${SCRIPT_DIR}/src"

AGENT_IMAGE="${AGENT_IMAGE:-knowledge-assistant:latest}"
MCP_IMAGE="${MCP_IMAGE:-knowledge-mcp:latest}"
BRAVE_IMAGE="${BRAVE_IMAGE:-knowledge-brave-mcp:latest}"
ENABLE_WEB_SEARCH="${ENABLE_WEB_SEARCH:-true}"
INCLUDE_SAMPLE_DOCS="${INCLUDE_SAMPLE_DOCS:-true}"
DOCS_PERSISTENCE="${DOCS_PERSISTENCE:-false}"
HELM_TIMEOUT="${HELM_TIMEOUT:-10m}"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# shellcheck source=scripts/toolkit-manager-lib.sh
source "${SCRIPT_DIR}/../../scripts/toolkit-manager-lib.sh"

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
for cmd in kubectl helm; do
  command -v "$cmd" &>/dev/null || die "Required tool '${cmd}' not found. Install it and re-run."
done

# ---------------------------------------------------------------------------
# Build the images (agent + knowledge MCP + Brave MCP) into containerd's k8s.io
# namespace (nerdctl) so the kubelet uses them with pullPolicy: IfNotPresent.
# Skip with SKIP_BUILD=true + AGENT_IMAGE/MCP_IMAGE/BRAVE_IMAGE.
# ---------------------------------------------------------------------------
build_images() {
  if [[ "${SKIP_BUILD:-false}" == "true" ]]; then
    info "SKIP_BUILD=true — skipping image build."
    return
  fi

  local tool="${CONTAINER_TOOL:-}"
  if [[ -z "${tool}" ]]; then
    if command -v nerdctl &>/dev/null; then
      tool="nerdctl"
    elif command -v docker &>/dev/null; then
      tool="docker"
      warn "nerdctl not found; using docker. On a containerd-backed cluster the"
      warn "kubelet may not see docker-built images — push to a registry and set"
      warn "AGENT_IMAGE/MCP_IMAGE/BRAVE_IMAGE + SKIP_BUILD=true if pods stay in ImagePullBackOff."
    else
      die "No container builder found. Install nerdctl (or docker), or set SKIP_BUILD=true."
    fi
  fi

  local dockerfiles=("${SRC_DIR}/Dockerfile" "${SRC_DIR}/Dockerfile.mcp")
  [[ "${ENABLE_WEB_SEARCH}" == "true" ]] && dockerfiles+=("${SRC_DIR}/Dockerfile.brave")

  # sudo drops the environment by default, so keep the proxy vars: nerdctl (the
  # BuildKit client) fetches registry auth tokens itself, not buildkitd.
  local build_cmd=("${tool}")
  local ns_args=()
  if [[ "${tool}" == "nerdctl" ]]; then
    ns_args=(--namespace k8s.io)
    if [[ "$(id -u)" -ne 0 ]]; then
      build_cmd=(sudo --preserve-env=http_proxy,https_proxy,no_proxy,HTTP_PROXY,HTTPS_PROXY,NO_PROXY nerdctl)
    fi
    # Install/start buildkitd + pre-stage base images if needed.
    CONTAINER_TOOL="${tool}" "${SCRIPT_DIR}/../../scripts/ensure-image-builder.sh" "${dockerfiles[@]}"
  fi

  # Proxy build args (inherited from the environment; empty when unset).
  local proxy_args=(
    --build-arg "http_proxy=${http_proxy:-}"
    --build-arg "https_proxy=${https_proxy:-}"
    --build-arg "no_proxy=${no_proxy:-}"
    --build-arg "HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-}}"
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}"
    --build-arg "NO_PROXY=${NO_PROXY:-${no_proxy:-}}"
  )

  info "Building knowledge MCP image '${MCP_IMAGE}' with ${build_cmd[*]}..."
  "${build_cmd[@]}" "${ns_args[@]}" build "${proxy_args[@]}" \
    -t "${MCP_IMAGE}" -f "${SRC_DIR}/Dockerfile.mcp" "${SRC_DIR}"

  if [[ "${ENABLE_WEB_SEARCH}" == "true" ]]; then
    info "Building Brave Search MCP image '${BRAVE_IMAGE}' with ${build_cmd[*]}..."
    "${build_cmd[@]}" "${ns_args[@]}" build "${proxy_args[@]}" \
      -t "${BRAVE_IMAGE}" -f "${SRC_DIR}/Dockerfile.brave" "${SRC_DIR}"
  else
    info "ENABLE_WEB_SEARCH=false — skipping Brave Search MCP image build."
  fi

  info "Building agent image '${AGENT_IMAGE}' with ${build_cmd[*]}..."
  "${build_cmd[@]}" "${ns_args[@]}" build "${proxy_args[@]}" \
    -t "${AGENT_IMAGE}" -f "${SRC_DIR}/Dockerfile" "${SRC_DIR}"

  success "Built images: ${AGENT_IMAGE}, ${MCP_IMAGE}$([[ "${ENABLE_WEB_SEARCH}" == "true" ]] && echo ", ${BRAVE_IMAGE}")"
}

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
prompt_if_empty() {
  local var_name="$1" prompt_text="$2" default="${3:-}"
  if [[ -z "${!var_name:-}" ]]; then
    if [[ -n "$default" ]]; then
      read -rp "  ${prompt_text} [${default}]: " input
      input="${input:-$default}"
    else
      read -rp "  ${prompt_text}: " input
    fi
    printf -v "$var_name" '%s' "$input"
  fi
}

prompt_secret_if_empty() {
  local var_name="$1" prompt_text="$2"
  if [[ -z "${!var_name:-}" ]]; then
    read -rsp "  ${prompt_text}: " input; echo ""
    printf -v "$var_name" '%s' "$input"
  fi
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Knowledge Assistant — Deployment"
echo "============================================================"
echo ""

tm_discover_base_domain
tm_discover_litellm_key
tm_discover_langfuse
tm_discover_pgvector

prompt_if_empty CLUSTER_URL \
  "Toolkit Manager base_domain_name (e.g. solutions.ai)"
[[ -n "${CLUSTER_URL}" ]] || die "CLUSTER_URL (Toolkit Manager base_domain_name) is required."

prompt_secret_if_empty OPENAI_API_KEY \
  "LiteLLM API key (OPENAI_API_KEY)"

# KNOWLEDGE_MODEL_NAME lets this use case call a different model (e.g. a router)
# than the MODEL_NAME shared by every blueprint in config.env.
MODEL_NAME="${KNOWLEDGE_MODEL_NAME:-${MODEL_NAME:-}}"
prompt_if_empty MODEL_NAME \
  "Model name registered in the gateway (must support tool calling)"

prompt_if_empty EMBEDDING_MODEL \
  "Embedding model in LiteLLM (for the pgvector knowledge store)" \
  "bge-base-en"

[[ -n "${PGVECTOR_URL:-}" ]] || die "No pgvector connection: set pgvector_enabled: true in config.agent-toolkit.yaml (secret ${PGVECTOR_NAMESPACE:-pgvector}/pgvector-credentials), or pass PGVECTOR_URL."

# Brave Search API key — optional; without one the assistant is internal-knowledge-only.
if [[ "${ENABLE_WEB_SEARCH}" == "true" ]]; then
  if [[ -z "${BRAVE_API_KEY:-}" && -t 0 ]]; then
    prompt_secret_if_empty BRAVE_API_KEY \
      "Brave Search API key (blank = internal knowledge only — https://api-dashboard.search.brave.com/)"
  fi
  if [[ -z "${BRAVE_API_KEY:-}" ]]; then
    warn "No BRAVE_API_KEY — disabling web search (internal knowledge only)."
    ENABLE_WEB_SEARCH="false"
  fi
fi

NAMESPACE="${NAMESPACE:-knowledge-assistant}"
INGRESS_HOST="${INGRESS_HOST:-knowledge-assistant.${CLUSTER_URL}}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${TM_LITELLM_URL}}"
DEVUI_AUTH_TOKEN="${DEVUI_AUTH_TOKEN:-$(openssl rand -hex 24 2>/dev/null || echo "changeme")}"

# The chart keeps REDIS_URL credential-free in its ConfigMap and reads the
# password by reference from a copy of redis/redis-stack-server-credentials.
# Strip any credentials a caller (e.g. the dispatcher's discovery) put in the URL.
REDIS_SECRET_NAME="redis-stack-server-credentials"
REDIS_URL="${REDIS_URL:-redis://redis-stack-server.redis.svc.cluster.local:6379}"
REDIS_URL="$(sed -E 's#://[^@/]*@#://#' <<<"${REDIS_URL}")"

echo ""
info "Deployment configuration:"
echo "  NAMESPACE          = ${NAMESPACE}"
echo "  INGRESS_HOST       = ${INGRESS_HOST}"
echo "  OPENAI_BASE_URL    = ${OPENAI_BASE_URL}"
echo "  MODEL_NAME         = ${MODEL_NAME}"
echo "  EMBEDDING_MODEL    = ${EMBEDDING_MODEL}"
echo "  REDIS_URL          = ${REDIS_URL} (password from ${REDIS_SECRET_NAME})"
echo "  PGVECTOR           = $(sed -E 's#(://[^:]*:)[^@]*(@)#\1****\2#' <<<"${PGVECTOR_URL}")"
echo "  ENABLE_WEB_SEARCH  = ${ENABLE_WEB_SEARCH}"
echo "  LANGFUSE_HOST      = ${LANGFUSE_HOST:-(tracing disabled)}"
echo "  DOCUMENTS          = $([[ "${DOCS_PERSISTENCE}" == "true" ]] && echo "PersistentVolumeClaim (kubectl cp)" || echo "baked in from src/docs/ ($(find "${SRC_DIR}/docs" -type f ! -name '.*' 2>/dev/null | wc -l | tr -d ' ') files)"), samples: ${INCLUDE_SAMPLE_DOCS}"
echo ""

# ---------------------------------------------------------------------------
# Build images
# ---------------------------------------------------------------------------
build_images

# ---------------------------------------------------------------------------
# Namespace (PSS restricted + Istio ambient) and LiteLLM access
# ---------------------------------------------------------------------------
# TLS is terminated at the platform gateway — no per-app certificate needed.
tm_prepare_namespace "${NAMESPACE}"
tm_check_litellm_access "${NAMESPACE}"

# Redis password: copy only the REDIS_PASSWORD key into NAMESPACE.
if tm_copy_secret_key redis "${REDIS_SECRET_NAME}" REDIS_PASSWORD "${NAMESPACE}"; then
  success "Copied redis/${REDIS_SECRET_NAME} (REDIS_PASSWORD) into '${NAMESPACE}'."
else
  warn "redis/${REDIS_SECRET_NAME} not found — connecting to Redis without auth (memory may fail)."
  REDIS_SECRET_NAME=""
fi

# ---------------------------------------------------------------------------
# Helm deploy
# ---------------------------------------------------------------------------
# Split "repo:tag" into repository + tag so the chart images match what we built.
AGENT_IMAGE_REPO="${AGENT_IMAGE%:*}"; AGENT_IMAGE_TAG="${AGENT_IMAGE##*:}"
[[ "${AGENT_IMAGE}" == *:* ]] || AGENT_IMAGE_TAG="latest"
MCP_IMAGE_REPO="${MCP_IMAGE%:*}"; MCP_IMAGE_TAG="${MCP_IMAGE##*:}"
[[ "${MCP_IMAGE}" == *:* ]] || MCP_IMAGE_TAG="latest"
BRAVE_IMAGE_REPO="${BRAVE_IMAGE%:*}"; BRAVE_IMAGE_TAG="${BRAVE_IMAGE##*:}"
[[ "${BRAVE_IMAGE}" == *:* ]] || BRAVE_IMAGE_TAG="latest"

HELM_ARGS=(
  upgrade --install knowledge-assistant "${CHART_DIR}"
  --namespace "${NAMESPACE}"
  --set "agent.image.repository=${AGENT_IMAGE_REPO}"
  --set "agent.image.tag=${AGENT_IMAGE_TAG}"
  --set "mcp.knowledge.image.repository=${MCP_IMAGE_REPO}"
  --set "mcp.knowledge.image.tag=${MCP_IMAGE_TAG}"
  --set "mcp.brave.image.repository=${BRAVE_IMAGE_REPO}"
  --set "mcp.brave.image.tag=${BRAVE_IMAGE_TAG}"
  --set "agent.openaiBaseUrl=${OPENAI_BASE_URL}"
  --set-string "agent.openaiApiKey=${OPENAI_API_KEY}"
  --set "agent.modelName=${MODEL_NAME}"
  --set-string "agent.devuiAuthToken=${DEVUI_AUTH_TOKEN}"
  --set-string "agent.enableWebSearch=${ENABLE_WEB_SEARCH}"
  --set "mcp.knowledge.embeddingModel=${EMBEDDING_MODEL}"
  --set-string "pgvector.url=${PGVECTOR_URL}"
  --set "redisUrl=${REDIS_URL}"
  --set-string "redisPasswordSecret.name=${REDIS_SECRET_NAME}"
  --set "mcp.knowledge.docs.includeSamples=${INCLUDE_SAMPLE_DOCS}"
  --set "mcp.knowledge.docs.persistence.enabled=${DOCS_PERSISTENCE}"
  --set-string "mcp.knowledge.docs.persistence.storageClass=${DOCS_STORAGE_CLASS:-}"
  --set "httpRoute.enabled=true"
  --set "httpRoute.host=${INGRESS_HOST}"
  --wait
  --timeout "${HELM_TIMEOUT}"
)

if [[ "${ENABLE_WEB_SEARCH}" == "true" ]]; then
  HELM_ARGS+=(
    --set "mcp.brave.enabled=true"
    --set-string "mcp.brave.apiKey=${BRAVE_API_KEY}"
    # The Brave sidecar calls the public api.search.brave.com at runtime.
    --set-string "mcp.brave.httpProxy=${HTTP_PROXY:-${http_proxy:-}}"
    --set-string "mcp.brave.httpsProxy=${HTTPS_PROXY:-${https_proxy:-}}"
  )
else
  HELM_ARGS+=(--set "mcp.brave.enabled=false")
fi

# Langfuse tracing (optional) — passed only when keys are provided.
if [[ -n "${LANGFUSE_PUBLIC_KEY:-}" && -n "${LANGFUSE_SECRET_KEY:-}" ]]; then
  HELM_ARGS+=(
    --set "agent.langfusePublicKey=${LANGFUSE_PUBLIC_KEY}"
    --set "agent.langfuseSecretKey=${LANGFUSE_SECRET_KEY}"
    --set "agent.langfuseHost=${LANGFUSE_HOST:-}"
    --set "agent.langfuseEnableSensitiveData=${LANGFUSE_ENABLE_SENSITIVE_DATA:-true}"
  )
  info "Langfuse tracing enabled (host: ${LANGFUSE_HOST:-<default>})."
fi

info "Deploying Knowledge Assistant via Helm..."
helm "${HELM_ARGS[@]}"

# ---------------------------------------------------------------------------
# Post-deploy summary
# ---------------------------------------------------------------------------
success "Knowledge Assistant deployed — namespace: ${NAMESPACE}"
echo ""
info "Access the Knowledge Assistant:"
info "  https://${INGRESS_HOST}"
tm_print_gateway_hint "${INGRESS_HOST}"
info "  (or locally: kubectl port-forward -n ${NAMESPACE} svc/knowledge-assistant 8090:8090 → http://localhost:8090)"
info ""
info "  DevUI login token (enter in the browser login prompt):"
info "  ${DEVUI_AUTH_TOKEN}"
if [[ "${DOCS_PERSISTENCE}" == "true" ]]; then
  info ""
  info "  Add documents to the docs volume, then restart to embed them:"
  info "  POD=\$(kubectl get pod -n ${NAMESPACE} -l app.kubernetes.io/instance=knowledge-assistant -o name | head -1)"
  info "  kubectl cp ./my-docs/hr \"\${POD#pod/}:/app/docs/hr\" -n ${NAMESPACE} -c knowledge-mcp"
  info "  kubectl rollout restart deployment/knowledge-assistant -n ${NAMESPACE}"
fi
echo ""
