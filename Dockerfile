# Stage 1: Builder - for Cython build only (keeps build tools out of the runtime image)
FROM python:3.10-slim AS builder

WORKDIR /build

# System tools required for Cython build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install only the packages needed for the Cython bbox module build
RUN pip install --no-cache-dir Cython numpy

# Copy Cython source files and build the bbox extension
COPY widerface_evaluate/box_overlaps.pyx ./widerface_evaluate/
COPY widerface_evaluate/setup.py ./widerface_evaluate/
RUN cd widerface_evaluate && python setup.py build_ext --inplace


# Stage 2: Runtime - GPU evaluation environment
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

LABEL maintainer="kyeongha-git"
LABEL description="OurModel Face Detection - WiderFace Evaluation"

WORKDIR /app

# OpenCV runtime dependencies (libGL, libglib)
# Note: Ubuntu 24.04 (Noble) replaced libgl1-mesa-glx with libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Docker-specific requirements)
# Note: --break-system-packages is required because the base image (Python 3.12)
#       enforces PEP 668, which prevents system-wide pip installs by default.
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.docker.txt

# Copy source code
COPY src/ ./src/
COPY tools/ ./tools/
COPY utils/ ./utils/

# Apply Docker-specific config: pretrain=False (pretrain weights not included in image)
# This overwrites config.py inside the container only; the host original is preserved.
COPY src/utils/config_docker.py ./src/utils/config.py

# Copy WiderFace evaluation module
COPY widerface_evaluate/ ./widerface_evaluate/

# Copy Cython .so built in Stage 1 (build tools excluded from runtime)
COPY --from=builder /build/widerface_evaluate/bbox*.so ./widerface_evaluate/

# Copy trained weights (Final model only, ~2.5 MB)
COPY weights/mobilenet0.25_eca_cbam_Final.pth ./weights/

# Create output directories
RUN mkdir -p \
    ./widerface_evaluate/widerface_txt \
    ./results \
    ./eval

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy entrypoint script and make it executable
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command: run WiderFace evaluation
# Note: mount the validation dataset at runtime:
#   -v /path/to/widerface/val:/app/data/widerface/val:ro
CMD ["evaluate"]
