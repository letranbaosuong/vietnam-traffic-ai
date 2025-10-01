# Multi-stage build for optimized image size
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p data/models data/cache data/uploads

# Set Python path
ENV PYTHONPATH=/app:/root/.local/lib/python3.9/site-packages
ENV PATH=/root/.local/bin:$PATH

# Download models
RUN python main.py download-models || true

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000", "--env", "production"]