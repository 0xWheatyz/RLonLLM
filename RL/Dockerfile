FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY RL/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RL source code
COPY RL/*.py .

# Create results directory
RUN mkdir -p /app/results && chmod 777 /app/results

# Command to run the training
CMD ["python", "main.py"]
