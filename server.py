"""
Perez Live Cam - Cloud Face Swap Server (Deep-Live-Cam pipeline)
================================================================
Same stack as Deep-Live-Cam / facefusion:
  - insightface FaceAnalysis (buffalo_l)  -> detection + 106 landmarks + embedding
  - inswapper_128 (fp16 when present)     -> face swap with paste_back blending
  - GFPGAN v1.4                           -> face enhancement (sharp detail)
  - mouth-mask blending                   -> keeps YOUR lip sync when talking

Client flow:
  1. POST /api/face   { photo: <base64 jpeg> }  -> { emb: [512 floats] }
  2. WS  /ws/frame    send { b64 } frames      -> { b64 } swapped frames
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
# Models - insightface pipeline (buffalo_l + inswapper) with GFPGAN enhance
# ---------------------------------------------------------------------------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

_fa = None
_swapper = None
_gfpgan = None
_swapper_path = None


def _pick_model(name):
    """Pick fp16 variant when present (faster), else full fp32."""
    fp16 = os.path.join(MODEL_DIR, name.replace(".onnx", "_fp16.onnx"))
    if os.path.exists(fp16):
        return fp16
    full = os.path.join(MODEL_DIR, name)
    return full if os.path.exists(full) else None


def load_models():
    global _fa, _swapper, _gfpgan, _swapper_path
    try:
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        _fa = FaceAnalysis(name="buffalo_l", root=MODEL_DIR, providers=providers)
        _fa.prepare(ctx_id=0, det_size=(640, 640))

        _swapper_path = _pick_model("inswapper_128.onnx")
        if _swapper_path:
            _swapper = get_model(_swapper_path, providers=providers)
        log.info("insightface ready. swapper=%s", os.path.basename(_swapper_path) if _swapper_path else None)
    except Exception as e:  # noqa: BLE001
        log.exception("insightface load failed: %s", e)
        _fa = None

    g = os.path.join(MODEL_DIR, "gfpgan_1.4.onnx")
    if os.path.exists(g):
        _gfpgan = ort.InferenceSession(g, providers=providers)
        log.info("gfpgan: True")
    else:
        log.info("gfpgan: False")


def _b64_to_jpeg(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _embed(face_img_bgr: np.ndarray):
    """ArcFace embedding of the source face photo (the user's chosen identity)."""
    faces = _fa.get(face_img_bgr)
    if not faces:
        raise RuntimeError("no face found in photo")
    return faces[0].normed_embedding


def _enhance(face_bgr: np.ndarray) -> np.ndarray:
    """GFPGAN v1.4: restore detail of the swapped face. The ONNX model is
    fixed at 512x512 input. Throttling (every Nth frame) is the speed lever."""
    if _gfpgan is None:
        return face_bgr
    h, w = face_bgr.shape[:2]
    S = 512
    resized = cv2.resize(face_bgr, (S, S))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = rgb.transpose(2, 0, 1)[None].astype(np.float32)  # 1,3,512,512
    out = _gfpgan.run(None, {_gfpgan.get_inputs()[0].name: inp})[0][0]
    out = (out.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return cv2.resize(out_bgr, (w, h))


def _swap_frame(emb: np.ndarray, frame_bgr: np.ndarray, enhance: bool = True) -> np.ndarray:
    """Swap the face in a full-res frame using insightface's paste_back, then
    enhance with GFPGAN and blend back with a landmark mask."""
    if _fa is None or _swapper is None:
        return frame_bgr

    faces = _fa.get(frame_bgr)
    if not faces:
        return frame_bgr
    target_face = max(faces, key=lambda f: f.bbox[3] - f.bbox[1])  # largest face

    # Source face: reuse the target's geometry, swap in the user's identity
    # embedding. normed_embedding is a read-only property derived from
    # `embedding`, so we only set `embedding` (the swapper normalizes again).
    from insightface.app.common import Face  # noqa: PLC0415

    src = Face(bbox=target_face.bbox, kps=target_face.kps, det_score=1.0)
    src.embedding = emb

    swapped = _swapper.get(frame_bgr, target_face, src, paste_back=True)
    if swapped is None:
        return frame_bgr

    # ---- landmark-based mask for clean blending (DLC-style) ----
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    lm = target_face.landmark_2d_106
    if lm is not None:
        hull = cv2.convexHull(lm[0:33].astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        mask = cv2.GaussianBlur(mask, (9, 9), 4)

    # mouth-mask: keep original mouth (lip sync while talking)
    if lm is not None and lm.shape[0] >= 64:
        mouth = lm[52:64].astype(np.float32)
        mm = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mm, cv2.convexHull(mouth).astype(np.int32), 255)
        mm = cv2.dilate(mm, np.ones((5, 5), np.uint8), iterations=2)
        mm = cv2.GaussianBlur(mm, (7, 7), 3)
        mask[mm > 128] = 0  # hole at the mouth -> keep original talking mouth

    # GFPGAN enhance the swapped face region (throttled)
    if enhance:
        y_idx, x_idx = np.where(mask > 0)
        if len(x_idx) > 0:
            y0, y1 = int(y_idx.min()), int(y_idx.max()) + 1
            x0, x1 = int(x_idx.min()), int(x_idx.max()) + 1
            pad = 20
            y0, y1 = max(0, y0 - pad), min(frame_bgr.shape[0], y1 + pad)
            x0, x1 = max(0, x0 - pad), min(frame_bgr.shape[1], x1 + pad)
            face_crop = swapped[y0:y1, x0:x1]
            enhanced = _enhance(face_crop)
            swapped[y0:y1, x0:x1] = enhanced

    # feathered alpha blend
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (swapped.astype(np.float32) * mask_f + frame_bgr.astype(np.float32) * (1 - mask_f)).astype(np.uint8)


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
    return {"ok": True, "gpu": ort.get_device(), "swapper": bool(_swapper)}


# ---------------------------------------------------------------------------
# WS: streaming swap
# ---------------------------------------------------------------------------
@app.websocket("/ws/frame")
async def ws_frame(ws: WebSocket):
    await ws.accept()
    emb: np.ndarray | None = None
    frames = 0
    t0 = time.time()
    enhance_every = 3  # GFPGAN every Nth frame (speed vs sharpness)
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "face":
                if msg.get("emb"):
                    emb = np.asarray(msg["emb"], dtype=np.float32)
                else:
                    img = _b64_to_jpeg(msg["photo"])
                    emb = _embed(img)
                await ws.send_text(json.dumps({"type": "face_ok"}))
                continue
            if emb is None:
                await ws.send_text(json.dumps({"type": "error", "message": "send face first"}))
                continue
            frame = _b64_to_jpeg(msg["b64"])
            t = time.time()
            out = _swap_frame(emb, frame, enhance=(frames % enhance_every == 0))
            elapsed = time.time() - t
            ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            frames += 1
            if frames % 25 == 0:
                log.info("swapped %d frames, %s ms/frame, ~%.1f fps", frames, round(elapsed * 1000, 1), round(frames / max(time.time() - t0, 0.001), 1))
            await ws.send_text(json.dumps({
                "type": "frame",
                "b64": base64.b64encode(jpg.tobytes()).decode(),
                "w": out.shape[1],
                "h": out.shape[0],
            }))
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("ws error")
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


load_models()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")