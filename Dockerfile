FROM --platform=linux/amd64 nvidia/cuda:12.1.0-base-ubuntu22.04

# Configure apt and install packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.1/compat/

# Set up environment variables for HuggingFace cache
ENV BASE_PATH="/runpod-volume" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies with caching
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir

# Copy handler code
COPY rp_handler.py /rp_handler.py

# Set the working directory
WORKDIR /

# Start the handler
CMD ["python3.11", "-u", "/rp_handler.py"]
