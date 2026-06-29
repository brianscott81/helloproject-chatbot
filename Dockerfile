# Dockerfile for the Hello! Project Wiki chatbot web interface.
#
# Build context expects:
#   - chat.py, conversation.py, llm.py, query.py, web.py (source)
#   - web/  (frontend HTML/CSS/JS)
#   - helloproject.db  (SQLite index, ~107MB)
#   - chroma/  (ChromaDB vector index, ~301MB)
#   - requirements.txt  (Python deps)
#
# Build:
#   docker build -t helloproject-wiki .
#
# Run locally:
#   docker run --rm -p 8000:8000 \
#     -e ANTHROPIC_BASE_URL=https... \
#     -e ANTHROPIC_AUTH_TOKEN=*** \
#     helloproject-wiki
#
# The image is ~700MB because it bundles:
#   - Python 3.11 slim base (~120MB)
#   - numpy, chromadb, anthropic, openai (~250MB installed)
#   - the pre-built SQLite + ChromaDB indexes (~400MB)
#   - the source code

FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr.
# Buffering is important for Fly.io log streaming.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# We need build tools for some Python packages (numpy, chromadb) and
# curl for the healthcheck.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching — code changes
# don't invalidate the pip install layer).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY chat.py conversation.py llm.py query.py web.py ./
COPY web/ ./web/

# Copy the pre-built indexes. These are gitignored but must be
# present in the build context. If you don't have them, run
# `python build_index.py` and `python build_embeddings.py` first.
COPY helloproject.db ./
COPY chroma/ ./chroma/

# Run as a non-root user for security. The chroma directory and db
# need to be readable by this user.
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the web server port. Fly.io's internal_port must match.
EXPOSE 8000

# Healthcheck for Fly.io. The / path returns 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/help || exit 1

# Default to binding on all interfaces so the platform's reverse
# proxy can reach the app. The web.py entry point handles the rest.
CMD ["python", "web.py", "--host", "0.0.0.0", "--port", "8000"]
