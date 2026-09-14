"""
Perez Live Cam - Cloud Face Swap Server
========================================
Receives webcam frames over WebSocket, swaps the face to a user-chosen
photo (inswapper_128), and streams the result back as JPEG.

Client flow:
  1. POST /api/face   { photo: <base64 jpeg> }      -> { emb: [512 floats] }
  2. WS  /ws/frame?emb=<hex or send once>           -> send { b64, emb } -> { b64 }

Runs on a GPU (RTX 4090 etc). ~5-15ms per swap on 4090.

Deploy (RunPod):
  docker build -t <user>/perez-face-swap:latest .
  docker push <user>/perez-face-swap:latest
  RunPod -> Deploy pod -> RTX 4090 -> Docker image above -> expose HTTP port 8000
  The pod gives you https://<id>-8000.proxy.runpod.net - use wss://<that>/ws/frame
"""

import asyncio
import base64
import io
import json
import logging
import time

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("faceswap")

MODEL_DIR = "models"

app = FastAPI(title="Perez Face Swap")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
arcface = ort.InferenceSession(f"{MODEL_DIR}/w600k_r50.onnx", providers=providers)
inswapper = ort.InferenceSession(f"{MODEL_DIR}/inswapper_128.onnx", providers=providers)
log.info("Models loaded. arcface input: %s, inswapper inputs: %s",
         arcface.get_inputs()[0].name, [i.name for i in inswapper.get_inputs()])

_face_cache: dict[str, np.ndarray] = {}  # client_id -> embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _b64_to_jpeg(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _embed(face_bgr: np.ndarray) -> np.ndarray:
    """ArcFace embedding from a BGR image (any size, centered 112 crop)."""
    h, w = face_bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = face_bgr[y0:y0 + side, x0:x0 + side]
    face = cv2.resize(crop, (112, 112))
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0..255
    inp = rgb.transpose(2, 0, 1)[None]  # 1,3,112,112
    out = arcface.run(None, {arcface.get_inputs()[0].name: inp})[0][0]
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def _swap(src_emb: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
    """Run inswapper on the whole frame (source emb + target image)."""
    S = 128
    small = cv2.resize(frame_bgr, (S, S))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp = ((rgb / 127.5) - 1.0).transpose(2, 0, 1)[None].astype(np.float32)  # 1,3,128,128
    feeds = {"source": src_emb[None].astype(np.float32), "target": inp}
    out = inswapper.run(None, feeds)[0][0]  # 3,128,128 in -1..1
    out = ((out + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out_bgr


# ---------------------------------------------------------------------------
# REST: register a face
# ---------------------------------------------------------------------------
class FaceIn(BaseModel):
    photo: str  # base64 jpeg


@app.post("/api/face")
async def register_face(body: FaceIn):
    try:
        img = _b64_to_jpeg(body.photo)
        emb = _embed(img)
        return {"ok": True, "emb": emb.tolist()}
    except Exception as e:  # noqa: BLE001
        log.exception("register_face failed")
        return {"ok": False, "error": str(e)}


@app.get("/health")
async def health():
    return {"ok": True, "gpu": ort.get_device()}


# ---------------------------------------------------------------------------
# WS: streaming swap
# ---------------------------------------------------------------------------
@app.websocket("/ws/frame")
async def ws_frame(ws: WebSocket):
    await ws.accept()
    emb: np.ndarray | None = None
    client_id = f"ws-{id(ws)}"
    frames = 0
    last_swap = 0.0
    t0 = time.time()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "face":  # first message: register source face
                if msg.get("emb"):
                    emb = np.asarray(msg["emb"], dtype=np.float32)
                else:
                    img = _b64_to_jpeg(msg["photo"])
                    emb = _embed(img)
                _face_cache[client_id] = emb
                await ws.send_text(json.dumps({"type": "face_ok"}))
                continue
            if emb is None:
                await ws.send_text(json.dumps({"type": "error", "message": "send face first"}))
                continue
            frame = _b64_to_jpeg(msg["b64"])
            t = time.time()
            swapped = _swap(emb, frame)
            last_swap = time.time() - t
            ok, jpg = cv2.imencode(".jpg", swapped, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            frames += 1
            await ws.send_text(json.dumps({
                "type": "frame",
                "b64": base64.b64encode(jpg.tobytes()).decode(),
                "w": swapped.shape[1],
                "h": swapped.shape[0],
                "swap_ms": round(last_swap * 1000, 1),
                "fps": round(frames / max(time.time() - t0, 0.001), 1),
            }))
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("ws error")
    finally:
        _face_cache.pop(client_id, None)
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")