FROM python:3.11-slim

# Cache bust - v3
ARG CACHEBUST=3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    echo "=== Installed packages ===" && pip list | grep -E "(yt-dlp|bgutil)" && \
    echo "=== yt-dlp version ===" && python -c "import yt_dlp; print(yt_dlp.version.__version__)" && \
    # Ensure __init__.py files exist for plugin discovery
    touch /usr/local/lib/python3.11/site-packages/yt_dlp_plugins/__init__.py && \
    touch /usr/local/lib/python3.11/site-packages/yt_dlp_plugins/extractor/__init__.py && \
    echo "=== Plugin directory contents ===" && \
    ls -la /usr/local/lib/python3.11/site-packages/yt_dlp_plugins/ && \
    ls -la /usr/local/lib/python3.11/site-packages/yt_dlp_plugins/extractor/

# Copy application
COPY . .

# Health check (Railway sets PORT dynamically)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE 8000

# Use PORT from environment (Railway sets this)
CMD ["sh", "-c", "echo Starting app... && python app.py"]

# Deploy trigger 1769779123
