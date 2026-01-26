FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (no whisper - using Groq API)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    yt-dlp \
    python-multipart \
    aiofiles \
    httpx \
    openai \
    groq

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use PORT from environment (Railway sets this)
CMD python app.py
