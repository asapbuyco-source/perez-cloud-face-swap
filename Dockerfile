# Perez Live Cam - Cloud Face Swap Server
# GPU image for RunPod / Vast.ai / any NVIDIA Docker host.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY download_models.py .
RUN python3 download_models.py

COPY server.py .

EXPOSE 8000
CMD ["python3", "server.py"]