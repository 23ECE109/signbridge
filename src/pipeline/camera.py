"""
Camera Capture & Frame Buffer
==============================
Captures frames from the built-in camera at 30 FPS.
Maintains a circular RAM buffer of 64 frames (~2.1 seconds).
Raw frames are NEVER written to disk — privacy by design.
"""

import threading
import logging
import time
from collections import deque
from typing import Optional, Callable

import cv2
import numpy as np

log = logging.getLogger("signbridge.pipeline.camera")

BUFFER_SIZE = 64         # ~2.1 seconds at 30 FPS
TARGET_FPS  = 30
FRAME_W     = 640
FRAME_H     = 480


class FrameBuffer:
    """Thread-safe circular buffer for raw video frames (RAM only)."""

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append(frame)

    def get_window(self) -> list[np.ndarray]:
        """Return a copy of the current frame window."""
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class CameraCapture:
    """
    Manages camera access and feeds frames into a FrameBuffer.

    Usage:
        cap = CameraCapture(on_frame=my_callback)
        cap.start()
        ...
        cap.stop()
    """

    def __init__(
        self,
        device_id: int = 0,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.device_id = device_id
        self.on_frame = on_frame
        self.buffer = FrameBuffer()

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._fps_actual = 0.0

    def start(self) -> bool:
        """Open camera and start capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            log.error(f"Failed to open camera device {self.device_id}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera opened: {actual_w}×{actual_h} @ device {self.device_id}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop capture and release camera. Buffer is cleared."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self.buffer.clear()  # Privacy: clear all frames from RAM
        log.info("Camera stopped and frame buffer cleared")

    def _capture_loop(self) -> None:
        interval = 1.0 / TARGET_FPS
        t_last = time.perf_counter()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                log.warning("Frame read failed — skipping")
                continue

            now = time.perf_counter()
            self._fps_actual = 1.0 / max(now - t_last, 1e-6)
            t_last = now

            # Convert BGR → RGB for downstream models
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.buffer.push(frame_rgb)

            if self.on_frame:
                self.on_frame(frame_rgb)

            # Yield remaining frame budget back to OS
            elapsed = time.perf_counter() - now
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def fps(self) -> float:
        return self._fps_actual

    def check_lighting_quality(self, frame: np.ndarray) -> float:
        """
        Returns a lighting quality score in [0, 1].
        Score < 0.3 → prompt user to improve lighting.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mean_brightness = gray.mean() / 255.0
        # Penalize too dark (< 0.2) or overexposed (> 0.9)
        if mean_brightness < 0.2:
            return mean_brightness / 0.2
        if mean_brightness > 0.9:
            return (1.0 - mean_brightness) / 0.1
        return 1.0
