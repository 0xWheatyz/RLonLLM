#!/usr/bin/env bash
#
# deploy.sh - Deploy RLonLLM on a DigitalOcean GPU droplet (Ubuntu 22.04)
#
# This script is idempotent: safe to re-run without side effects.
# It will:
#   1. Verify NVIDIA GPU and drivers are available
#   2. Install Docker Engine if not present
#   3. Install nvidia-container-toolkit if not present
#   4. Clone or update the repo
#   5. Build and run the GRPO trainer via docker compose
#   6. Tail the logs
#
# Usage:
#   chmod +x deploy.sh && ./deploy.sh
#
set -euo pipefail

REPO_URL="https://github.com/0xWheatyz/RLonLLM.git"
DEPLOY_DIR="/opt/RLonLLM"

echo "=== RLonLLM GPU Deployment ==="

# ---------------------------------------------------------------
# 1. Check for NVIDIA GPU and drivers
# ---------------------------------------------------------------
echo ""
echo "[1/5] Checking NVIDIA GPU and drivers..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers must be pre-installed on the GPU droplet."
    echo "       DigitalOcean GPU droplets ship with drivers; if missing, install manually."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "NVIDIA drivers OK."

# ---------------------------------------------------------------
# 2. Install Docker Engine (if not already installed)
# ---------------------------------------------------------------
echo ""
echo "[2/5] Ensuring Docker is installed..."

if ! command -v docker &>/dev/null; then
    echo "Docker not found. Installing Docker Engine..."
    apt-get update -y
    apt-get install -y ca-certificates curl gnupg

    install -m 0755 -d /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        chmod a+r /etc/apt/keyrings/docker.gpg
    fi

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list

    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    echo "Docker installed successfully."
else
    echo "Docker already installed: $(docker --version)"
fi

# ---------------------------------------------------------------
# 3. Install nvidia-container-toolkit (if not already installed)
# ---------------------------------------------------------------
echo ""
echo "[3/5] Ensuring nvidia-container-toolkit is installed..."

if ! dpkg -s nvidia-container-toolkit &>/dev/null 2>&1; then
    echo "nvidia-container-toolkit not found. Installing..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "nvidia-container-toolkit installed and configured."
else
    echo "nvidia-container-toolkit already installed."
fi

# ---------------------------------------------------------------
# 4. Clone or update the repository
# ---------------------------------------------------------------
echo ""
echo "[4/5] Setting up repository at ${DEPLOY_DIR}..."

if [ -d "${DEPLOY_DIR}/.git" ]; then
    echo "Repo exists. Pulling latest changes..."
    git -C "${DEPLOY_DIR}" pull --ff-only
else
    echo "Cloning repository..."
    git clone "${REPO_URL}" "${DEPLOY_DIR}"
fi

cd "${DEPLOY_DIR}"
mkdir -p results

# ---------------------------------------------------------------
# 5. Build and run with docker compose
# ---------------------------------------------------------------
echo ""
echo "[5/5] Building and starting GRPO trainer..."

docker compose down --remove-orphans 2>/dev/null || true
docker compose build --no-cache
docker compose up -d

echo ""
echo "=== Deployment complete ==="
echo "Container is running. Tailing logs (Ctrl+C to stop)..."
echo ""
docker compose logs -f
