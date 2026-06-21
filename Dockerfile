# ─── Build stage ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile pdfplumber/pypdf native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Runtime stage ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="POD-JD Intelligence"
LABEL org.opencontainers.image.description="Multi-agent pipeline: POD emails → JD analysis → Student & Faculty briefs"
LABEL org.opencontainers.image.source="https://github.com/mitaoe-aiml/pod-jd-intelligence"

# Create non-root user
RUN groupadd -r podjd && useradd -r -g podjd -d /app podjd

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=podjd:podjd . .

# Output directories (will be volume-mounted in production)
RUN mkdir -p output/jds output/drafts && chown -R podjd:podjd output/

USER podjd

# Default command: run pipeline once (override in docker-compose for MCP server)
CMD ["python", "main.py", "run"]
