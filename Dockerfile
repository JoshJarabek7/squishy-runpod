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

# Create persistent directories for model storage
RUN mkdir -p /models/sam /models/huggingface /root/.cache/huggingface

# Set up environment variables for HuggingFace cache locations
ENV BASE_PATH="/root/.cache" \
    HF_DATASETS_CACHE="/models/huggingface/datasets" \
    HF_HOME="/models/huggingface" \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies with caching
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install --upgrade -r /requirements.txt --no-cache-dir

# Copy handler code
COPY rp_handler.py /rp_handler.py

# Download SAM model and move it to /models/sam
RUN python3.13 -c "from ultralytics import SAM; model = SAM('sam2.1_l.pt'); print(f'Model downloaded to: {model.ckpt_path}')" && \
    find / -name "sam2.1_l.pt" -exec mv {} /models/sam/sam2.1_l.pt \; || echo "SAM model not found, assuming it's already in place"

# Download OWLv2 model files using transformers directly
RUN python3.13 -c "import torch; \
    from transformers import Owlv2Processor, Owlv2ForObjectDetection; \
    processor_result = Owlv2Processor.from_pretrained('google/owlv2-large-patch14', cache_dir='/models/huggingface'); \
    global_processor = processor_result[0] if isinstance(processor_result, tuple) else processor_result; \
    global_owlv2_model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-large-patch14', cache_dir='/models/huggingface')"

# Set the working directory
WORKDIR /

# Start the handler
CMD ["python3.13", "-u", "/rp_handler.py"]