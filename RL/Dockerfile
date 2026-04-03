FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch to avoid pulling CUDA)
COPY RL/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the RL source code
COPY RL/*.py .

# Create results directory
RUN mkdir -p /app/results && chmod 777 /app/results

# Default: run the prompt-guided baseline
CMD ["python", "main.py"]
