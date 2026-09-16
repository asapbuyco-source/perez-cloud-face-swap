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
import os
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
gfpgan = None
if os.path.exists(f"{MODEL_DIR}/gfpgan_1.4.onnx"):
    gfpgan = ort.InferenceSession(f"{MODEL_DIR}/gfpgan_1.4.onnx", providers=providers)
log.info("Models loaded. arcface input: %s, inswapper inputs: %s, gfpgan: %s",
         arcface.get_inputs()[0].name, [i.name for i in inswapper.get_inputs()], bool(gfpgan))

_face_cache: dict[str, np.ndarray] = {}  # client_id -> embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _b64_to_jpeg(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# Haar cascade for frontal face detection (bundled with opencv-python-headless).
_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        import os
        p = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
        _cascade = cv2.CascadeClassifier(p)
    return _cascade


def _detect_face(frame_bgr: np.ndarray):
    """Returns (x, y, w, h) of the largest frontal face, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


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


def _enhance(face_bgr: np.ndarray) -> np.ndarray:
    """GFPGAN v1.4: restore detail/clarity of the swapped face. 512x512."""
    if gfpgan is None:
        return face_bgr
    S = 512
    resized = cv2.resize(face_bgr, (S, S))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = rgb.transpose(2, 0, 1)[None].astype(np.float32)  # 1,3,512,512
    out = gfpgan.run(None, {gfpgan.get_inputs()[0].name: inp})[0][0]
    out = (out.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return cv2.resize(out_bgr, (face_bgr.shape[1], face_bgr.shape[0]))


def _feathered_paste(base: np.ndarray, patch: np.ndarray, x0: int, y0: int, feather: int = 25) -> None:
    """Paste patch onto base with a feathered elliptical alpha mask (no hard box)."""
    ph, pw = patch.shape[:2]
    x0 = max(0, min(x0, base.shape[1] - pw))
    y0 = max(0, min(y0, base.shape[0] - ph))
    mask = np.zeros((ph, pw), dtype=np.float32)
    center = (pw / 2, ph / 2)
    axes = (max(pw / 2 - feather, 1), max(ph / 2 - feather, 1))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    # soft feather border
    kernel = cv2.getGaussianKernel(feather * 2 + 1, feather / 3)
    mask = cv2.filter2D(mask, -1, kernel)
    mask = np.clip(mask, 0, 1)[:, :, None].astype(np.float32)
    region = base[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    base[y0:y0 + ph, x0:x0 + pw] = (patch.astype(np.float32) * mask + region * (1 - mask)).astype(np.uint8)


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
            box = _detect_face(frame)
            if box is None:
                # no face: pass the frame through untouched
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    continue
                frames += 1
                await ws.send_text(json.dumps({
                    "type": "frame",
                    "b64": base64.b64encode(jpg.tobytes()).decode(),
                    "w": frame.shape[1],
                    "h": frame.shape[0],
                    "face": None,
                }))
                continue
            fx, fy, fw, fh = box
            # pad the crop a bit so the hairline blends
            pad = 0.35
            cx, cy = fx + fw / 2, fy + fh / 2
            half = max(fw, fh) * (1 + pad) / 2
            x0 = max(0, int(cx - half))
            y0 = max(0, int(cy - half))
            x1 = min(frame.shape[1], int(cx + half))
            y1 = min(frame.shape[0], int(cy + half))
            crop = frame[y0:y1, x0:x1]
            swapped_crop = _swap(emb, crop) if crop.size else crop
            # GFPGAN: restore detail on the swapped face
            swapped_crop = _enhance(swapped_crop)
            # feathered paste back onto the full-res frame (no hard box)
            out = frame.copy()
            _feathered_paste(out, swapped_crop, x0, y0)
            last_swap = time.time() - t
            ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            frames += 1
            if frames % 25 == 0:
                log.info("swapped %d frames, %s ms/frame, ~%.1f fps", frames, round(last_swap * 1000, 1), round(frames / max(time.time() - t0, 0.001), 1))
            await ws.send_text(json.dumps({
                "type": "frame",
                "b64": base64.b64encode(jpg.tobytes()).decode(),
                "w": out.shape[1],
                "h": out.shape[0],
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