FROM --platform=arm64 runpod/base:0.6.3-cuda11.8.0

COPY requirements.txt /requirements.txt

RUN python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install --upgrade -r requirements.txt --no-cache-dir && \
    rm /requirements.txt

COPY rp_handler.py .

CMD [ "python3.13", "-u", "rp_handler.py" ]