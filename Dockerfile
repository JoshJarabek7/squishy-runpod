FROM --platform=linux/amd64 nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Configure apt and install packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13-full \
    python3.13-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.6/compat/

# Create persistent directories for model storage
RUN mkdir -p /models/sam \
    /models/huggingface \
    /root/.cache/huggingface

# Set up environment variables for model and cache locations
ENV BASE_PATH="/root/.cache" \
    HF_DATASETS_CACHE="/models/huggingface/datasets" \
    HUGGINGFACE_HUB_CACHE="/models/huggingface/hub" \
    HF_HOME="/models/huggingface" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    ULTRALYTICS_DIR="/models/sam"

# Install Python dependencies with caching
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install --upgrade -r /requirements.txt --no-cache-dir

# Copy handler code
COPY rp_handler.py /rp_handler.py

# Pre-download models during build
RUN python3.13 -c "from ultralytics import SAM; SAM('sam2.1_l.pt')" && \
    python3.13 -c "from transformers import Owlv2Processor, Owlv2ForObjectDetection; Owlv2Processor.from_pretrained('google/owlv2-large-patch14'); Owlv2ForObjectDetection.from_pretrained('google/owlv2-large-patch14')"

# Set the working directory
WORKDIR /

# Start the handler
CMD ["python3.13", "-u", "/rp_handler.py"]