"""
Landmark Extractor — Live MediaPipe Holistic (CPU, works immediately)
=====================================================================
Extracts hand + pose landmarks from a single RGB frame using
MediaPipe Holistic running on CPU. No ONNX, no Qualcomm account needed
for development. Drop-in replaceable with QNN version for Snapdragon demo.

Output per frame: flat float32 vector [225]
  - Left hand:  21 landmarks × 3 = 63
  - Right hand: 21 landmarks × 3 = 63
  - Pose:       33 landmarks × 3 = 99
  Total: 225
"""

import logging
import threading
from typing import Optional
import numpy as np

log = logging.getLogger("signbridge.pipeline.landmark_extractor")

N_HAND  = 21
N_POSE  = 33
DIM     = 3
SEQ_DIM = (N_HAND * 2 + N_POSE) * DIM   # 225
SEQ_LEN = 64


class LandmarkResult:
    def __init__(self,
                 left_hand:  Optional[np.ndarray] = None,
                 right_hand: Optional[np.ndarray] = None,
                 pose:       Optional[np.ndarray] = None):
        self.left_hand  = left_hand   # [21,3] or None
        self.right_hand = right_hand  # [21,3] or None
        self.pose       = pose        # [33,3] or None

    def to_flat_vector(self) -> np.ndarray:
        lh = self.left_hand.flatten()  if self.left_hand  is not None else np.zeros(N_HAND * DIM, dtype=np.float32)
        rh = self.right_hand.flatten() if self.right_hand is not None else np.zeros(N_HAND * DIM, dtype=np.float32)
        ps = self.pose.flatten()       if self.pose        is not None else np.zeros(N_POSE * DIM, dtype=np.float32)
        return np.concatenate([lh, rh, ps]).astype(np.float32)

    @property
    def any_hand(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None

    def dominant_hand(self) -> Optional[np.ndarray]:
        """Return right hand if available, else left hand."""
        if self.right_hand is not None:
            return self.right_hand
        return self.left_hand


class LandmarkExtractor:
    """
    Live MediaPipe Holistic landmark extractor.
    Thread-safe. Call load() once, then extract() per frame.
    """

    def __init__(self, complexity: int = 0):
        """
        complexity=0 → fastest (good for real-time demo)
        complexity=1 → more accurate
        """
        self.complexity = complexity
        self._holistic  = None
        self._lock      = threading.Lock()
        self._loaded    = False

    def load(self) -> bool:
        try:
            import mediapipe as mp
            self._mp_holistic = mp.solutions.holistic
            self._holistic = self._mp_holistic.Holistic(
                static_image_mode        = False,
                model_complexity         = self.complexity,
                smooth_landmarks         = True,
                min_detection_confidence = 0.5,
                min_tracking_confidence  = 0.5,
            )
            self._loaded = True
            log.info(f"MediaPipe Holistic loaded (complexity={self.complexity})")
            return True
        except ImportError:
            log.error("mediapipe not installed. Run: pip install mediapipe")
            return False
        except Exception as e:
            log.error(f"MediaPipe load failed: {e}")
            return False

    def extract(self, frame_rgb: np.ndarray) -> LandmarkResult:
        """
        Extract landmarks from one RGB frame.
        Args:
            frame_rgb: uint8 [H, W, 3]
        Returns:
            LandmarkResult
        """
        if not self._loaded or self._holistic is None:
            return LandmarkResult()

        with self._lock:
            results = self._holistic.process(frame_rgb)

        lh = rh = ps = None

        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z]
                           for lm in results.left_hand_landmarks.landmark],
                          dtype=np.float32)   # [21,3]

        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z]
                           for lm in results.right_hand_landmarks.landmark],
                          dtype=np.float32)   # [21,3]

        if results.pose_landmarks:
            ps = np.array([[lm.x, lm.y, lm.z]
                           for lm in results.pose_landmarks.landmark[:N_POSE]],
                          dtype=np.float32)   # [33,3]

        return LandmarkResult(lh, rh, ps)

    def normalize_sequence(self, sequence: list) -> np.ndarray:
        """
        Normalize a list of flat [225] vectors → [64, 225] tensor.
        Zero-mean + unit-std per feature, pad/trim to SEQ_LEN.
        """
        if not sequence:
            return np.zeros((SEQ_LEN, SEQ_DIM), dtype=np.float32)

        arr = np.stack(sequence, axis=0).astype(np.float32)  # [T, 225]
        arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-6)

        T = arr.shape[0]
        if T < SEQ_LEN:
            arr = np.concatenate([arr, np.zeros((SEQ_LEN - T, SEQ_DIM), dtype=np.float32)])
        elif T > SEQ_LEN:
            idx = np.linspace(0, T - 1, SEQ_LEN, dtype=int)
            arr = arr[idx]
        return arr.astype(np.float32)

    def close(self):
        if self._holistic:
            self._holistic.close()
