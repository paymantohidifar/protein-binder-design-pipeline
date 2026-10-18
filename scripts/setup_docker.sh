#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Terminal Output Formatting Helpers
# ------------------------------------------------------------------------------
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; } # Blue section header
ok()   { printf '   \033[1;32mok\033[0m  %s\n' "$*"; }  # Green success badge
warn() { printf '   \033[1;33m--\033[0m  %s\n' "$*"; }  # Yellow warning badge

# ------------------------------------------------------------------------------
# Docker Engine Installation Check
# ------------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    warn "Docker executable not found on PATH."
    log "Installing Docker via official convenience script..."
    curl -fsSL https://get.docker.com | sudo sh
    ok "Docker installation completed."
else
    ok "Docker $(docker --version | awk '{print $3}' | tr -d ,)"
fi

# ------------------------------------------------------------------------------
# Docker Group Privileges & Non-Interactive Self-Elevation
# ------------------------------------------------------------------------------
# Checks if the active subshell process has 'docker' group permissions.
# If missing, adds $USER to the docker group and re-executes the script instantly
# using 'sg docker', avoiding manual logouts or terminal restarts on cloud VMs.
if ! groups | grep -qw docker; then
    if ! id -nG "$USER" | grep -qw docker; then
        log "Adding $USER to the 'docker' group..."
        sudo usermod -aG docker "$USER"
        ok "Added $USER to the 'docker' group."
        newgrp docker
    fi

    log "Re-executing script with active 'docker' group privileges..."
    exec sg docker -c "$0 $*"
fi

ok "Docker environment verified and active."
warn "Run 'newgrp docker' in your interactive terminal to apply the group changes."
warn "Run 'docker images' to verify user permissions."