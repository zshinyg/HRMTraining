# Multi-stage Dockerfile for HRM-CodeGen
# Production-ready image with support for both CPU and GPU environments

# -----------------------------------------------------------------------------
# Base stage with shared dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Builder stage for dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# GPU-specific builder stage
# -----------------------------------------------------------------------------
FROM builder AS gpu-builder

# Install CUDA-specific dependencies
RUN pip install --no-cache-dir torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# -----------------------------------------------------------------------------
# CPU-specific builder stage
# -----------------------------------------------------------------------------
FROM builder AS cpu-builder

# Install CPU-specific dependencies
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# -----------------------------------------------------------------------------
# Runtime stage - GPU version
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    CONFIG_DIR=/app/configs \
    LOG_DIR=/app/logs \
    CHECKPOINT_DIR=/app/checkpoints \
    PORT=8000 \
    WORKERS=4 \
    MAX_BATCH_SIZE=16 \
    MAX_SEQUENCE_LENGTH=2048 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ca-certificates \
    curl \
    netcat-openbsd \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser && \
    mkdir -p /app/models /app/data /app/configs /app/logs /app/checkpoints && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=gpu-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=gpu-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/data /app/configs /app/logs /app/checkpoints && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port for API
EXPOSE ${PORT}

# Set up entrypoint script
COPY --chown=appuser:appuser scripts/docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/docker_entrypoint.sh"]

# Default command (can be overridden)
CMD ["serve"]

# -----------------------------------------------------------------------------
# Runtime stage - CPU version
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS cpu-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    CONFIG_DIR=/app/configs \
    LOG_DIR=/app/logs \
    CHECKPOINT_DIR=/app/checkpoints \
    PORT=8000 \
    WORKERS=4 \
    MAX_BATCH_SIZE=8 \
    MAX_SEQUENCE_LENGTH=1024

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    netcat-openbsd \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser && \
    mkdir -p /app/models /app/data /app/configs /app/logs /app/checkpoints && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=cpu-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=cpu-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/data /app/configs /app/logs /app/checkpoints && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port for API
EXPOSE ${PORT}

# Set up entrypoint script
COPY --chown=appuser:appuser scripts/docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/docker_entrypoint.sh"]

# Default command (can be overridden)
CMD ["serve"]

# -----------------------------------------------------------------------------
# Final stage - selectable at build time with --target
# Default to GPU version
# -----------------------------------------------------------------------------
FROM gpu-runtime
