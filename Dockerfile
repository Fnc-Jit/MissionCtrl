FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy project
COPY . .

# Set PYTHONPATH and disable output buffering (critical for Docker log visibility)
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7860

# Health check - generous timeouts for Hugging Face Spaces
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=10 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run server (the app.py lifespan prints the full banner)
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
