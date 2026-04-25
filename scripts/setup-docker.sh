#!/bin/bash
set -euo pipefail

# Set up Docker containers for LiteLLM (PostgreSQL), Open WebUI, and mcpo

CONFIG_DIR="$HOME/.config/wsl-llm"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$CONFIG_DIR/install.env" ] && source "$CONFIG_DIR/install.env" 2>/dev/null || true

DB_PASSWORD="${DB_PASSWORD:-litellm123}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-}"
WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY:-$(openssl rand -hex 24)}"

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not installed."
    echo "Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon not running. Start it with:"
    echo "  sudo service docker start"
    exit 1
fi

echo "=== PostgreSQL for LiteLLM ==="
if docker ps -a --format '{{.Names}}' | grep -q '^litellm-pg$'; then
    echo "  Container exists. Starting..."
    docker start litellm-pg 2>/dev/null || true
else
    echo "  Creating container..."
    docker run -d --name litellm-pg --restart unless-stopped \
        -e POSTGRES_USER=litellm \
        -e POSTGRES_PASSWORD="$DB_PASSWORD" \
        -e POSTGRES_DB=litellm \
        -p 5434:5432 \
        postgres:15-alpine
fi
echo "  PostgreSQL ready on port 5434"
echo ""

echo "=== Open WebUI ==="
if docker ps -a --format '{{.Names}}' | grep -q '^open-webui$'; then
    echo "  Container exists. Starting..."
    docker start open-webui 2>/dev/null || true
else
    echo "  Creating container..."
    docker run -d --name open-webui --network host --restart always \
        -e PORT=3000 \
        -e OPENAI_API_BASE_URL=http://localhost:4000/v1 \
        -e OPENAI_API_KEY="${LITELLM_MASTER_KEY:-sk-placeholder}" \
        -e WEBUI_SECRET_KEY="$WEBUI_SECRET_KEY" \
        -e ENABLE_WEB_SEARCH=true \
        -e WEB_SEARCH_ENGINE=duckduckgo \
        -e ANONYMIZED_TELEMETRY=false \
        -e DO_NOT_TRACK=true \
        -v open-webui:/app/backend/data \
        ghcr.io/open-webui/open-webui:main
fi
echo "  Open WebUI ready on port 3000"
echo ""

echo "=== mcpo (MCP-to-OpenAPI Bridge) ==="
MCPO_CONFIG="$HOME/mcpo-config.json"
if [ ! -f "$MCPO_CONFIG" ]; then
    echo "  Generating config from template..."
    sed "s|{{LITELLM_MASTER_KEY}}|${LITELLM_MASTER_KEY:-sk-placeholder}|g" \
        "$REPO_DIR/config/mcpo/config.json.template" > "$MCPO_CONFIG"
fi
if docker ps -a --format '{{.Names}}' | grep -q '^mcpo$'; then
    echo "  Container exists. Starting..."
    docker start mcpo 2>/dev/null || true
else
    echo "  Creating container..."
    docker run -d --name mcpo --network host --restart always \
        -v "$MCPO_CONFIG":/app/config.json:ro \
        ghcr.io/open-webui/mcpo:main \
        --config /app/config.json --port 8000 --host 0.0.0.0
fi
echo "  mcpo ready on port 8000"
echo ""

echo "=== Docker Status ==="
docker ps --format "  {{.Names}}\t{{.Status}}\t{{.Ports}}"
