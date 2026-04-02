FROM ollama/ollama:latest

# Pull the model during the build process
# 1. Start ollama serve in the background
# 2. Wait for it to be ready
# 3. Pull the requested model
# 4. Kill the server process to complete the build step
RUN (ollama serve &) && sleep 10 && ollama pull qwen2.5-coder:1.5b && pkill ollama

# Expose the default ollama port
EXPOSE 11434

# The base image already has ENTRYPOINT ["ollama"] and CMD ["serve"],
# but we can explicitly set them to be sure.
ENTRYPOINT ["ollama"]
CMD ["serve"]
