"""
Perez Live Cam - Cloud Face Swap Server
========================================
Runs the real Deep-Live-Cam-VFX frame processors (vendored under ./dlc)
on a RunPod GPU:

  modules/processors/frame/face_swapper.py   -> DLC face swap (mask/blend)
  modules/processors/frame/face_enhancer.py  -> DLC GFPGAN enhancement

Client flow:
  1. POST /api/face   { photo: <base64 jpeg> }  -> { ok: true }
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

import dlc_engine

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


def _b64_to_jpeg(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# REST: register a face (photo -> DLC source face)
# ---------------------------------------------------------------------------
class FaceIn(BaseModel):
    photo: str  # base64 jpeg


@app.post("/api/face")
async def register_face(body: FaceIn):
    try:
        img = _b64_to_jpeg(body.photo)
        ok = dlc_engine.set_source_photo(img)
        return {"ok": ok, "error": None if ok else "no face found in photo"}
    except Exception as e:  # noqa: BLE001
        log.exception("register_face failed")
        return {"ok": False, "error": str(e)}


@app.get("/health")
async def health():
    return {"ok": True, "gpu": ort.get_device(), "source": dlc_engine.has_source()}


# ---------------------------------------------------------------------------
# WS: streaming swap
# ---------------------------------------------------------------------------
@app.websocket("/ws/frame")
async def ws_frame(ws: WebSocket):
    await ws.accept()
    frames = 0
    t0 = time.time()
    enhance_every = 3  # GFPGAN every Nth frame (speed vs sharpness)
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "face":
                if msg.get("photo"):
                    img = _b64_to_jpeg(msg["photo"])
                    ok = dlc_engine.set_source_photo(img)
                    await ws.send_text(json.dumps(
                        {"type": "face_ok", "ok": ok,
                         "error": None if ok else "no face found in photo"}
                    ))
                else:
                    await ws.send_text(json.dumps({"type": "face_ok", "ok": True}))
                continue
            if not dlc_engine.has_source():
                await ws.send_text(json.dumps({"type": "error", "message": "send face first"}))
                continue
            frame = _b64_to_jpeg(msg["b64"])
            t = time.time()
            out = dlc_engine.process_frame(frame, enhance=(frames % enhance_every == 0))
            elapsed = time.time() - t
            ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
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


dlc_engine.init()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")