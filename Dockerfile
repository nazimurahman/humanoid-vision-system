# Default Dockerfile for inference
# Multi-stage build for optimized production deployment

# Stage 1: Builder
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY . /tmp/app/
WORKDIR /tmp/app

# Install the package
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r robot && useradd -r -g robot -m -u 1000 robot
USER robot
WORKDIR /home/robot/app

# Copy application code
COPY --chown=robot:robot src/ ./src/
COPY --chown=robot:robot configs/ ./configs/
COPY --chown=robot:robot models/ ./models/
COPY --chown=robot:robot scripts/ ./scripts/
COPY --chown=robot:robot docker/entrypoint.sh ./
COPY --chown=robot:robot docker/healthcheck.py ./

# Set environment variables
ENV PYTHONPATH=/home/robot/app:/home/robot/app/src
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Create necessary directories
RUN mkdir -p /home/robot/app/logs /home/robot/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python /home/robot/app/healthcheck.py

# Expose ports
# FastAPI REST API
EXPOSE 8000  
# gRPC server
EXPOSE 50051 

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]