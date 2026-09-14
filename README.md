# Perez Live Cam — Cloud Face Swap

GPU runs in the cloud; the user's PC only sends webcam frames and receives
the swapped face back. No models on the client (keeps the installer at ~100 MB).

## How it works

```
User's PC (Perez app)                  Your GPU server (RunPod / Vast.ai / any NVIDIA box)
─────────────────────                  ─────────────────────────────────────────────
 pick face photo ── POST /api/face ──► ArcFace embed (w600k_r50) ── returns emb[512]
 Live Camera ON ── WS connect ──────► /ws/frame
   send JPEG frame (base64)  ───────► inswapper_128 (GPU, ~5-15ms) ──► swap
   ◄── swapped JPEG frame ───────────
 draw on preview + push to virtual camera (SHM)  →  call apps see the swapped face
```

Latency target: ~150-250 ms end-to-end at 10 fps (60-80 ms on RTX 4090 GPU).

## Deploy (RunPod — ~30 seconds)

1. Install Docker, then from this folder:

   ```bash
   docker build -t <your-dockerhub-user>/perez-face-swap:latest .
   docker push <your-dockerhub-user>/perez-face-swap:latest
   ```

2. RunPod → Deploy → Pod → **RTX 4090** → Docker image `<your-dockerhub-user>/perez-face-swap:latest`
3. Set **HTTP Port: 8000** → Deploy.
4. RunPod gives you a proxy URL like:
   `https://<podid>-8000.proxy.runpod.net`
5. In Perez app → **Settings → Face Swap (Cloud)** → paste
   `wss://<podid>-8000.proxy.runpod.net/ws/frame` (client converts to the REST/WS paths automatically)
6. Pick a face photo → **Start Cloud Swap** → enable your virtual camera → calls see the swap.

## Vast.ai (cheaper spot)

Same image; after renting, open the pod's port 8000 (HTTP expose) and use its URL.

## Cost estimate

- RTX 4090 community cloud ≈ **$0.34/hr** while the pod is running
- One person streaming 10 fps ≈ ~4% GPU utilization — a single 4090 handles
  several concurrent users comfortably
- Stop the pod when not in use → billed per second

## Env / config

- Models are baked into the image (`models/inswapper_128.onnx`, `models/w600k_r50.onnx`).
- Server listens on `0.0.0.0:8000` (FastAPI + uvicorn).
- Uses `CUDAExecutionProvider` with CPU fallback.

## Security note

The server binds publicly. Add an auth token when going production:
pass `?token=` in the WS/HTTP URLs and check it in the handlers (see the
`check_token` TODO in `server.py`). For a personal pod, runpod's random proxy
URL is enough.