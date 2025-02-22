FROM --platform=linux/amd64 nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Configure apt and install packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    libgl1 \
    libglib2.0-0 \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13-full \
    python3.13-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.6/compat/

# Create persistent directories for model storage and caching
RUN mkdir -p /models/sam /models/huggingface /root/.cache/huggingface /root/.cache/ultralytics

# Set up environment variables for model caching and offline mode
ENV HF_HOME="/models/huggingface" \
    TRANSFORMERS_CACHE="/models/huggingface" \
    HF_HUB_CACHE="/models/huggingface" \
    HF_DATASETS_CACHE="/models/huggingface/datasets" \
    ULTRALYTICS_CONFIG="/models/ultralytics" \
    ULTRALYTICS_ASSETS="/models/ultralytics/assets" \
    ULTRALYTICS_CACHE_DIR="/models/ultralytics/cache" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_RETRY_COUNT=5 \
    PYTHONUNBUFFERED=1

# Install Python dependencies with caching
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    python3.13 -m pip install --no-cache-dir 'huggingface_hub[cli]' hf_transfer

# Copy handler code and download script
COPY rp_handler.py /rp_handler.py
COPY download_models.py /download_models.py
COPY utils.py /utils.py

# Make the download script executable and run it with retries
RUN chmod +x /download_models.py && \
    (python3.13 /download_models.py || \
     (sleep 5 && python3.13 /download_models.py) || \
     (sleep 10 && python3.13 /download_models.py))

# Set the working directory
WORKDIR /

# Start the handler
CMD ["python3.13", "-u", "/rp_handler.py"]