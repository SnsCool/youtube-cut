FROM python:3.11-slim

# Install FFmpeg and dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Cache bust argument - change this to force rebuild
ARG CACHE_BUST=v3_no_whisper

# Copy requirements and install (without whisper)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && echo $CACHE_BUST

# Copy application code
COPY . .

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Run the application using python (handles PORT env var)
CMD ["python", "app.py"]
