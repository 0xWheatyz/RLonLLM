FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# NVIDIA container runtime env vars
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.11 and system deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    procps \
    git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support first, then remaining deps
COPY RL/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the RL source code
COPY RL/*.py .

# Pre-download default model weights during build so GPU time isn't wasted
# on network I/O at runtime. Cache lives in the image layer.
ARG PREFETCH_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${PREFETCH_MODEL}', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('${PREFETCH_MODEL}', trust_remote_code=True)"

# Create results directory
RUN mkdir -p /app/results && chmod 777 /app/results

# Default: run GRPO training
CMD ["python", "grpo_train.py"]
