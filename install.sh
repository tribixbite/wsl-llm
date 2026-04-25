#!/bin/bash
set -euo pipefail

# WSL-LLM installer — sets up local LLM inference stack on WSL2
# Usage: ./install.sh [options]
#
# Options:
#   --admin-user USER       LiteLLM/WebUI admin username (default: admin)
#   --admin-password PASS   Admin password (default: auto-generated)
#   --admin-email EMAIL     Admin email (optional)
#   --llama-api-key KEY     llama-server API key (default: auto-generated)
#   --domain DOMAIN         Cloudflare tunnel domain (optional, skips tunnel if not set)
#   --skip-build            Skip building inference engines
#   --skip-model            Skip model download
#   --skip-venv             Skip Python venv setup
#   --skip-docker           Skip Docker container setup
#   --yes                   Non-interactive, accept all defaults

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.config/wsl-llm"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[!]${NC} $*" >&2; }
step() { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}"; }

# ── Argument Parsing ─────────────────────────────────────────

ADMIN_USER="admin"
ADMIN_PASSWORD=""
ADMIN_EMAIL=""
LLAMA_API_KEY=""
DOMAIN=""
SKIP_BUILD=false
SKIP_MODEL=false
SKIP_VENV=false
SKIP_DOCKER=false
YES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --admin-user)     ADMIN_USER="$2"; shift 2 ;;
        --admin-password) ADMIN_PASSWORD="$2"; shift 2 ;;
        --admin-email)    ADMIN_EMAIL="$2"; shift 2 ;;
        --llama-api-key)  LLAMA_API_KEY="$2"; shift 2 ;;
        --domain)         DOMAIN="$2"; shift 2 ;;
        --skip-build)     SKIP_BUILD=true; shift ;;
        --skip-model)     SKIP_MODEL=true; shift ;;
        --skip-venv)      SKIP_VENV=true; shift ;;
        --skip-docker)    SKIP_DOCKER=true; shift ;;
        --yes|-y)         YES=true; shift ;;
        -h|--help)
            head -15 "$0" | tail -13 | sed 's/^# //' | sed 's/^#//'
            exit 0 ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Generate Secrets ─────────────────────────────────────────

generate_secrets() {
    if [ -f "$CONFIG_DIR/install.env" ]; then
        log "Loading existing secrets from $CONFIG_DIR/install.env"
        source "$CONFIG_DIR/install.env"
    fi

    [ -z "$LLAMA_API_KEY" ]  && LLAMA_API_KEY=$(openssl rand -base64 36 | tr -d '/+=' | head -c 48)
    [ -z "$ADMIN_PASSWORD" ] && ADMIN_PASSWORD=$(openssl rand -base64 18 | tr -d '/+=' | head -c 16)
    LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-$(openssl rand -hex 12)}"
    LITELLM_SALT_KEY="${LITELLM_SALT_KEY:-sk-salt-$(openssl rand -hex 12)}"
    DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -base64 18 | tr -d '/+=' | head -c 16)}"
}

# ── Template Substitution ────────────────────────────────────

substitute_template() {
    local input="$1" output="$2"
    sed \
        -e "s|{{HOME}}|$HOME|g" \
        -e "s|{{USER}}|$(whoami)|g" \
        -e "s|{{REPO_DIR}}|$REPO_DIR|g" \
        -e "s|{{LLAMA_API_KEY}}|$LLAMA_API_KEY|g" \
        -e "s|{{LITELLM_MASTER_KEY}}|$LITELLM_MASTER_KEY|g" \
        -e "s|{{LITELLM_SALT_KEY}}|$LITELLM_SALT_KEY|g" \
        -e "s|{{DB_PASSWORD}}|$DB_PASSWORD|g" \
        -e "s|{{ADMIN_USER}}|$ADMIN_USER|g" \
        -e "s|{{ADMIN_PASSWORD}}|$ADMIN_PASSWORD|g" \
        -e "s|{{ADMIN_EMAIL}}|$ADMIN_EMAIL|g" \
        -e "s|{{MODEL_PATH}}|$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf|g" \
        -e "s|{{DOMAIN}}|$DOMAIN|g" \
        -e "s|{{TUNNEL_UUID}}|${TUNNEL_UUID:-CONFIGURE_ME}|g" \
        "$input" > "$output"
}

# ── Preflight Checks ────────────────────────────────────────

preflight() {
    step "Preflight Checks"

    # WSL2
    if grep -qi microsoft /proc/version 2>/dev/null; then
        log "WSL2 detected"
    else
        warn "Not running in WSL2 — some features may not work"
    fi

    # NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        log "NVIDIA GPU detected ($gpu_count GPU(s))"
    else
        err "nvidia-smi not found. NVIDIA GPU required."
        exit 1
    fi

    # CUDA toolkit
    if [ -d /usr/local/cuda-12.6 ]; then
        log "CUDA 12.6 toolkit found"
    elif [ -d /usr/local/cuda ]; then
        warn "CUDA found at /usr/local/cuda (not 12.6). May need adjustment."
    else
        err "CUDA toolkit not found. Install from: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi

    # Docker
    if command -v docker &>/dev/null; then
        log "Docker installed"
    else
        warn "Docker not found. Install for LiteLLM DB and Open WebUI."
    fi

    # Disk space
    local avail_gb
    avail_gb=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$avail_gb" -lt 30 ]; then
        warn "Low disk space: ${avail_gb}GB available (need ~30GB for model + builds)"
    else
        log "Disk space: ${avail_gb}GB available"
    fi
}

# ── Install System Dependencies ──────────────────────────────

install_deps() {
    step "System Dependencies"

    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential git curl wget python3 python3-pip python3-venv jq bc

    # cmake 3.25+ (system cmake is usually too old for ik_llama.cpp)
    if ! "$HOME/.local/bin/cmake" --version 2>/dev/null | grep -q "3\.\(2[5-9]\|[3-9][0-9]\)\|[4-9]\."; then
        log "Installing cmake via pip..."
        pip3 install --user cmake
    else
        log "cmake $(${HOME}/.local/bin/cmake --version | head -1 | awk '{print $3}') already installed"
    fi

    # huggingface-cli for model downloads
    if ! command -v huggingface-cli &>/dev/null; then
        log "Installing huggingface-cli..."
        pip3 install --user "huggingface_hub[cli,hf_transfer]"
    else
        log "huggingface-cli already installed"
    fi
}

# ── Setup CUDA Environment ───────────────────────────────────

setup_cuda() {
    step "CUDA Environment"

    if ! grep -q "cuda-12.6" "$HOME/.bash_profile" 2>/dev/null; then
        cp "$REPO_DIR/config/bash_profile" "$HOME/.bash_profile"
        log "Installed CUDA PATH config to ~/.bash_profile"
    else
        log "CUDA PATH already configured"
    fi

    export PATH="/usr/local/cuda-12.6/bin:$HOME/.local/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
}

# ── Generate Configs from Templates ──────────────────────────

generate_configs() {
    step "Generating Configs"
    mkdir -p "$CONFIG_DIR"

    # llama-server config
    if [ -f "$HOME/llama-server.conf" ]; then
        log "llama-server.conf exists, backing up"
        cp "$HOME/llama-server.conf" "$HOME/llama-server.conf.bak"
    fi
    substitute_template "$REPO_DIR/config/llama-server.conf.template" "$HOME/llama-server.conf"
    log "Generated ~/llama-server.conf"

    # LiteLLM config
    if [ -f "$HOME/litellm_config.yaml" ]; then
        cp "$HOME/litellm_config.yaml" "$HOME/litellm_config.yaml.bak"
    fi
    substitute_template "$REPO_DIR/config/litellm_config.yaml.template" "$HOME/litellm_config.yaml"
    log "Generated ~/litellm_config.yaml"

    # Cloudflare config (only if domain provided)
    if [ -n "$DOMAIN" ]; then
        mkdir -p "$HOME/.cloudflared"
        if [ -f "$HOME/.cloudflared/config.yml" ]; then
            # Preserve existing tunnel UUID
            TUNNEL_UUID=$(grep "^tunnel:" "$HOME/.cloudflared/config.yml" 2>/dev/null | awk '{print $2}' || echo "CONFIGURE_ME")
            export TUNNEL_UUID
        fi
        substitute_template "$REPO_DIR/config/cloudflared.yml.template" "$HOME/.cloudflared/config.yml"
        log "Generated ~/.cloudflared/config.yml"
    fi
}

# ── Install Systemd Services ────────────────────────────────

install_services() {
    step "Systemd Services"

    # Process all .template files
    for tmpl in "$REPO_DIR"/services/*.template; do
        local name
        name=$(basename "$tmpl" .template)
        substitute_template "$tmpl" "/tmp/wsl-llm-$name"
        sudo cp "/tmp/wsl-llm-$name" "/etc/systemd/system/$name"
        rm -f "/tmp/wsl-llm-$name"
        log "Installed $name"
    done

    # Copy non-template files
    sudo cp "$REPO_DIR/services/llama-server-watchdog.timer" /etc/systemd/system/
    log "Installed llama-server-watchdog.timer"

    # Make scripts executable
    chmod +x "$REPO_DIR/services/llama-server.sh"

    # Generate the watchdog script from template
    substitute_template "$REPO_DIR/services/llama-server-watchdog.sh.template" \
        "$REPO_DIR/services/llama-server-watchdog.sh"
    chmod +x "$REPO_DIR/services/llama-server-watchdog.sh"

    sudo systemctl daemon-reload

    # Enable services
    sudo systemctl enable llama-server
    sudo systemctl enable llama-server-watchdog.timer
    sudo systemctl enable litellm
    [ -n "$DOMAIN" ] && sudo systemctl enable cloudflared

    log "All services enabled"
}

# ── Setup Sudoers ────────────────────────────────────────────

setup_sudoers() {
    step "Sudoers (passwordless service management)"

    local user
    user=$(whoami)
    local sudoers="/etc/sudoers.d/wsl-llm"

    cat <<EOF | sudo tee "$sudoers" > /dev/null
$user ALL=(root) NOPASSWD: /bin/systemctl start llama-server
$user ALL=(root) NOPASSWD: /bin/systemctl stop llama-server
$user ALL=(root) NOPASSWD: /bin/systemctl restart llama-server
$user ALL=(root) NOPASSWD: /bin/systemctl status *
$user ALL=(root) NOPASSWD: /bin/systemctl start litellm
$user ALL=(root) NOPASSWD: /bin/systemctl stop litellm
$user ALL=(root) NOPASSWD: /bin/systemctl restart litellm
$user ALL=(root) NOPASSWD: /bin/systemctl start cloudflared
$user ALL=(root) NOPASSWD: /bin/systemctl stop cloudflared
$user ALL=(root) NOPASSWD: /bin/systemctl restart cloudflared
$user ALL=(root) NOPASSWD: /bin/systemctl start llama-server-watchdog.timer
$user ALL=(root) NOPASSWD: /bin/systemctl stop llama-server-watchdog.timer
$user ALL=(root) NOPASSWD: /bin/systemctl daemon-reload
EOF
    sudo chmod 0440 "$sudoers"
    sudo visudo -cf "$sudoers"
    log "Passwordless sudo configured for service management"
}

# ── Install CLI ──────────────────────────────────────────────

install_cli() {
    step "Installing llm CLI"
    mkdir -p "$HOME/.local/bin"
    chmod +x "$REPO_DIR/bin/llm"
    ln -sf "$REPO_DIR/bin/llm" "$HOME/.local/bin/llm"
    log "Installed: llm -> $REPO_DIR/bin/llm"
}

# ── Save Install Metadata ───────────────────────────────────

save_metadata() {
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/install.env" <<EOF
# Generated by install.sh on $(date -Iseconds)
# DO NOT commit this file
LLAMA_API_KEY=$LLAMA_API_KEY
LITELLM_MASTER_KEY=$LITELLM_MASTER_KEY
LITELLM_SALT_KEY=$LITELLM_SALT_KEY
DB_PASSWORD=$DB_PASSWORD
ADMIN_USER=$ADMIN_USER
ADMIN_PASSWORD=$ADMIN_PASSWORD
ADMIN_EMAIL=$ADMIN_EMAIL
DOMAIN=$DOMAIN
REPO_DIR=$REPO_DIR
EOF
    chmod 600 "$CONFIG_DIR/install.env"
    log "Secrets saved to $CONFIG_DIR/install.env"
}

# ── Start Services ───────────────────────────────────────────

start_services() {
    step "Starting Services"

    sudo systemctl start llama-server
    log "Started llama-server (model loading may take 30-60s...)"

    sudo systemctl start llama-server-watchdog.timer
    log "Started watchdog timer"

    # Wait for llama-server health
    echo -n "  Waiting for llama-server..."
    for i in $(seq 1 60); do
        if curl -s --max-time 2 -H "Authorization: Bearer $LLAMA_API_KEY" \
            http://localhost:8080/health &>/dev/null; then
            echo " ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""

    if command -v docker &>/dev/null && [ "$SKIP_DOCKER" = false ]; then
        sudo systemctl start litellm
        log "Started LiteLLM"
    fi

    if [ -n "$DOMAIN" ]; then
        sudo systemctl start cloudflared
        log "Started Cloudflare tunnel"
    fi
}

# ── Print Summary ────────────────────────────────────────────

print_summary() {
    echo ""
    echo -e "${BOLD}============================================${NC}"
    echo -e "${BOLD}  WSL-LLM Installation Complete${NC}"
    echo -e "${BOLD}============================================${NC}"
    echo ""
    echo "Endpoints:"
    echo "  llama-server:    http://localhost:8080"
    echo "  LiteLLM proxy:   http://localhost:4000"
    echo "  LiteLLM admin:   http://localhost:4000/ui"
    echo "  Open WebUI:      http://localhost:3000"
    [ -n "$DOMAIN" ] && echo "  Public:          https://$DOMAIN"
    echo ""
    echo "Credentials:"
    echo "  llama-server key:  $LLAMA_API_KEY"
    echo "  LiteLLM master:    $LITELLM_MASTER_KEY"
    echo "  Admin login:       $ADMIN_USER / $ADMIN_PASSWORD"
    echo ""
    echo "Quick commands:"
    echo "  llm status         Check all services"
    echo "  llm health         Health check endpoints"
    echo "  llm restart        Restart llama-server"
    echo "  llm logs           Tail server logs"
    echo "  llm bench quick    Run performance benchmark"
    echo "  llm config edit    Edit server config"
    echo "  llm key create X   Create API key for user X"
    echo ""
    echo "Credentials saved: $CONFIG_DIR/install.env"
    echo -e "${BOLD}============================================${NC}"
}

# ── Main ─────────────────────────────────────────────────────

main() {
    echo -e "${BOLD}WSL-LLM Installer${NC}"
    echo ""

    preflight
    generate_secrets
    install_deps
    setup_cuda

    if [ "$SKIP_BUILD" = false ]; then
        step "Building Inference Engines"
        # Primary: Madreag's TurboQuant CUDA fork (turbo3 KV cache, full 262k ctx)
        "$REPO_DIR/scripts/build-turboquant.sh"
        # Secondary: ik_llama.cpp + upstream llama.cpp as fallback engines
        "$REPO_DIR/scripts/build-engines.sh"
    else
        log "Skipping engine builds (--skip-build)"
    fi

    if [ "$SKIP_MODEL" = false ]; then
        step "Model Download"
        if [ "$YES" = true ]; then
            echo "Y" | "$REPO_DIR/scripts/download-model.sh"
        else
            "$REPO_DIR/scripts/download-model.sh"
        fi
    else
        log "Skipping model download (--skip-model)"
    fi

    if [ "$SKIP_VENV" = false ]; then
        step "Python Environment"
        "$REPO_DIR/scripts/setup-venv.sh"
    else
        log "Skipping venv setup (--skip-venv)"
    fi

    generate_configs
    setup_sudoers
    install_services
    install_cli
    save_metadata

    if [ "$SKIP_DOCKER" = false ]; then
        step "Docker Containers"
        "$REPO_DIR/scripts/setup-docker.sh"
    else
        log "Skipping Docker setup (--skip-docker)"
    fi

    start_services
    print_summary
}

main
