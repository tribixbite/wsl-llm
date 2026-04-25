#!/bin/bash
set -euo pipefail

# WSL-LLM uninstaller — cleanly removes all components

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

echo "WSL-LLM Uninstaller"
echo ""
echo "This will remove:"
echo "  - Systemd services (llama-server, litellm, cloudflared, watchdog)"
echo "  - Sudoers entry"
echo "  - CLI symlink (~/.local/bin/llm)"
echo "  - Generated configs (~/llama-server.conf, ~/litellm_config.yaml)"
echo "  - Install metadata (~/.config/wsl-llm/)"
echo ""
read -rp "Continue? [y/N] " answer
if [[ ! "${answer:-N}" =~ ^[Yy] ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Stop and disable services
log "Stopping services..."
sudo systemctl stop llama-server litellm cloudflared llama-server-watchdog.timer 2>/dev/null || true
sudo systemctl disable llama-server litellm cloudflared llama-server-watchdog.timer 2>/dev/null || true

# Remove service files
log "Removing service files..."
sudo rm -f /etc/systemd/system/llama-server.service
sudo rm -f /etc/systemd/system/llama-server-watchdog.service
sudo rm -f /etc/systemd/system/llama-server-watchdog.timer
sudo rm -f /etc/systemd/system/litellm.service
sudo rm -f /etc/systemd/system/cloudflared.service
sudo systemctl daemon-reload

# Remove sudoers
log "Removing sudoers entry..."
sudo rm -f /etc/sudoers.d/wsl-llm

# Remove CLI
log "Removing llm CLI..."
rm -f "$HOME/.local/bin/llm"

# Remove generated configs
log "Removing configs..."
rm -f "$HOME/llama-server.conf" "$HOME/llama-server.conf.bak"
rm -f "$HOME/litellm_config.yaml" "$HOME/litellm_config.yaml.bak"
rm -rf "$HOME/.config/wsl-llm"

echo ""

# Optional: Docker containers
read -rp "Remove Docker containers (litellm-pg, open-webui)? [y/N] " answer
if [[ "${answer:-N}" =~ ^[Yy] ]]; then
    docker stop litellm-pg open-webui 2>/dev/null || true
    docker rm litellm-pg open-webui 2>/dev/null || true
    log "Docker containers removed"
fi

# Optional: Engine builds
read -rp "Remove engine builds (~3.8GB: ~/ik_llama.cpp, ~/llama.cpp)? [y/N] " answer
if [[ "${answer:-N}" =~ ^[Yy] ]]; then
    rm -rf "$HOME/ik_llama.cpp" "$HOME/llama.cpp"
    log "Engine builds removed"
fi

# Optional: Models
read -rp "Remove downloaded models (~20GB: ~/models/)? [y/N] " answer
if [[ "${answer:-N}" =~ ^[Yy] ]]; then
    rm -rf "$HOME/models"
    log "Models removed"
fi

# Optional: Python venv
read -rp "Remove Python venv (~2GB: ~/bench_env/)? [y/N] " answer
if [[ "${answer:-N}" =~ ^[Yy] ]]; then
    rm -rf "$HOME/bench_env"
    log "Python venv removed"
fi

echo ""
log "Uninstall complete."
