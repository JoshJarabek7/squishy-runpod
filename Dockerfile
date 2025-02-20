FROM --platform=linux/amd64 runpod/base:0.6.3-cuda11.8.0

# Install Python 3.13 with force-overwrite
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    # Download the packages first
    apt-get download python3.13-minimal python3.13 libpython3.13-minimal libpython3.13-stdlib python3.13-dev python3.13-venv python3.13-distutils && \
    # Force install them with dpkg
    find . -name "*.deb" -exec dpkg --force-overwrite -i {} \; && \
    # Fix any remaining dependencies
    apt-get install -y -f && \
    # Install pip using get-pip.py
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f *.deb

# Define build arguments
ARG CUDA_VISIBLE_DEVICES
ARG MAX_CONCURRENCY
ARG GPU_MEMORY_UTILIZATION
ARG TRUST_REMOTE_CODE
ARG DEFAULT_BATCH_SIZE
ARG DEFAULT_MIN_BATCH_SIZE
ARG DEFAULT_BATCH_SIZE_GROWTH_FACTOR
ARG DISABLE_LOG_STATS
ARG DISABLE_LOG_REQUESTS
ARG RAW_OPENAI_OUTPUT
ARG TOKENIZER_MODE
ARG BLOCK_SIZE
ARG SWAP_SPACE
ARG ENFORCE_EAGER
ARG MAX_SEQ_LEN_TO_CAPTURE
ARG DISABLE_CUSTOM_ALL_REDUCE

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
ENV MAX_CONCURRENCY=${MAX_CONCURRENCY}
ENV GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
ENV TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
ENV DEFAULT_BATCH_SIZE=${DEFAULT_BATCH_SIZE}
ENV DEFAULT_MIN_BATCH_SIZE=${DEFAULT_MIN_BATCH_SIZE}
ENV DEFAULT_BATCH_SIZE_GROWTH_FACTOR=${DEFAULT_BATCH_SIZE_GROWTH_FACTOR}
ENV DISABLE_LOG_STATS=${DISABLE_LOG_STATS}
ENV DISABLE_LOG_REQUESTS=${DISABLE_LOG_REQUESTS}
ENV RAW_OPENAI_OUTPUT=${RAW_OPENAI_OUTPUT}
ENV TOKENIZER_MODE=${TOKENIZER_MODE}
ENV BLOCK_SIZE=${BLOCK_SIZE}
ENV SWAP_SPACE=${SWAP_SPACE}
ENV ENFORCE_EAGER=${ENFORCE_EAGER}
ENV MAX_SEQ_LEN_TO_CAPTURE=${MAX_SEQ_LEN_TO_CAPTURE}
ENV DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}

COPY requirements.txt /requirements.txt

RUN python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install --upgrade -r requirements.txt --no-cache-dir && \
    rm /requirements.txt

COPY rp_handler.py .

CMD [ "python3.13", "-u", "rp_handler.py" ]