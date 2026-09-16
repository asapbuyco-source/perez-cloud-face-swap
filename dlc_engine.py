"""Deep-Live-Cam-VFX engine (vendored DLC code) running headless on the pod.

Uses DLC's real frame processors:
  modules/processors/frame/face_swapper.py  -> process_frame(source_face, frame)
  modules/processors/frame/face_enhancer.py -> process_frame(source_face, frame)

The vendored modules live in ./dlc/modules; only core.py is a stub (the real
one imports tkinter + tensorflow). globals are configured below exactly the
way DLC's live-mode UI sets them.
"""

import os
import sys
import logging
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlc"))

import modules.globals  # noqa: E402

modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
modules.globals.execution_threads = 0
modules.globals.many_faces = False
modules.globals.map_faces = False
modules.globals.opacity = 1.0
modules.globals.sharpness = 0.5
modules.globals.mouth_mask = False
modules.globals.poisson_blend = False
modules.globals.live_mirror = False

from modules.face_analyser import get_one_face  # noqa: E402
from modules.processors.frame import face_swapper  # noqa: E402
from modules.processors.frame import face_enhancer  # noqa: E402

log = logging.getLogger("dlc")
_logger_ready = False
_source_face_lock = threading.Lock()
_source_face = None


def init(source_photo_bgr: np.ndarray | None = None) -> None:
    """Load DLC models. Optionally set the source face from a photo."""
    global _logger_ready
    if not _logger_ready:
        logging.basicConfig(level=logging.INFO)
        _logger_ready = True

    log.info("loading face_swapper model (DLC)")
    if face_swapper.get_face_swapper() is None:
        raise RuntimeError("DLC face_swapper failed to load")

    log.info("warmup enhancer (GFPGAN)")
    try:
        face_enhancer.get_face_enhancer()
        log.info("face_enhancer ready")
    except Exception as e:  # noqa: BLE001
        log.warning("face_enhancer unavailable: %s (continuing without it)", e)

    if source_photo_bgr is not None:
        set_source_photo(source_photo_bgr)


def set_source_photo(photo_bgr: np.ndarray) -> bool:
    """Extract the source identity (DLC's get_one_face = leftmost face)."""
    global _source_face
    face = get_one_face(photo_bgr)
    if face is None:
        log.error("no face found in source photo")
        return False
    with _source_face_lock:
        _source_face = face
    log.info("source face set: %s", face.bbox.astype(int).tolist())
    return True


def process_frame(frame_bgr: np.ndarray, enhance: bool = True) -> np.ndarray:
    """DLC's live pipeline: swap (their mask/blend/sharpen), then enhance."""
    with _source_face_lock:
        source = _source_face
    if source is None:
        return frame_bgr

    out = face_swapper.process_frame(source, frame_bgr)
    if enhance:
        try:
            out = face_enhancer.process_frame(source, out)
        except Exception as e:  # noqa: BLE001
            log.debug("enhance skipped: %s", e)
    return out


def has_source() -> bool:
    with _source_face_lock:
        return _source_face is not None