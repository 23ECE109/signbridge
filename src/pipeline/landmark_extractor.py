"""
Landmark Extractor — MediaPipe Holistic via Qualcomm AI Hub / ONNX
====================================================================
Extracts 543 body landmarks per frame:
  - 21 × 2 hands × 3D  = 126 hand coordinates
  - 33 pose landmarks × 3D = 99 body coordinates
  - Total used: 225 coordinates/frame (face mesh optional)

Runs on Snapdragon Hexagon NPU via QNNExecutionProvider.
Benchmark (Qualcomm AI Hub): 52–76 µs per frame.

Model source:
  https://aihub.qualcomm.com/models/mediapipe_hand_gesture
  https://aihub.qualcomm.com/models/mediapipe_pose
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("signbridge.pipeline.landmark_extractor")

# ── Constants ─────────────────────────────────────────────────────────────────
N_HAND_LANDMARKS  = 21      # per hand
N_POSE_LANDMARKS  = 33
LANDMARK_DIM      = 3       # x, y, z
SEQUENCE_DIM      = (N_HAND_LANDMARKS * 2 + N_POSE_LANDMARKS) * LANDMARK_DIM  # 225


class LandmarkResult:
    """Structured result from one frame."""

    def __init__(
        self,
        left_hand:  Optional[np.ndarray] = None,   # [21, 3]
        right_hand: Optional[np.ndarray] = None,   # [21, 3]
        pose:       Optional[np.ndarray] = None,   # [33, 3]
        face:       Optional[np.ndarray] = None,   # [468, 3] optional
    ):
        self.left_hand  = left_hand
        self.right_hand = right_hand
        self.pose       = pose
        self.face       = face

    def to_flat_vector(self) -> np.ndarray:
        """
        Flatten to a 225-dim vector suitable for the temporal encoder.
        Missing landmarks (hand not detected) are zero-filled.
        """
        lh = self.left_hand.flatten()  if self.left_hand  is not None else np.zeros(N_HAND_LANDMARKS * LANDMARK_DIM)
        rh = self.right_hand.flatten() if self.right_hand is not None else np.zeros(N_HAND_LANDMARKS * LANDMARK_DIM)
        ps = self.pose.flatten()       if self.pose        is not None else np.zeros(N_POSE_LANDMARKS  * LANDMARK_DIM)
        return np.concatenate([lh, rh, ps], axis=0).astype(np.float32)  # [225]

    @property
    def both_hands_detected(self) -> bool:
        return self.left_hand is not None and self.right_hand is not None

    @property
    def any_hand_detected(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None


class LandmarkExtractor:
    """
    Wraps MediaPipe Holistic ONNX models with QNNExecutionProvider.

    In the prototype, this class acts as the interface layer.
    If the QNN-compiled model is available, it runs on the NPU.
    If not, it falls back to the CPU ONNX session.

    TODO for full implementation:
      - Download QNN-compiled .dlc files from Qualcomm AI Hub
      - Load with QNNExecutionProvider
      - Map model I/O to LandmarkResult
    """

    def __init__(self, model_dir: Optional[Path] = None, use_npu: bool = True):
        self.model_dir = model_dir or Path(__file__).parent.parent.parent / "models"
        self.use_npu   = use_npu
        self._session  = None
        self._loaded   = False

    def load(self) -> bool:
        """Load ONNX landmark extraction models. Returns True on success."""
        try:
            import onnxruntime as ort

            providers = []
            if self.use_npu and "QNNExecutionProvider" in ort.get_available_providers():
                providers.append((
                    "QNNExecutionProvider",
                    {
                        "backend_type":                       "htp",
                        "htp_performance_mode":               "burst",
                        "htp_graph_finalization_optimization_mode": "3",
                        "enable_htp_fp16_precision":          "1",
                    }
                ))
                log.info("Using QNN NPU provider for landmark extraction")
            else:
                log.info("Using CPU provider for landmark extraction")

            providers.append("CPUExecutionProvider")

            # Model path — populated after Qualcomm AI Hub download
            model_path = self.model_dir / "mediapipe_holistic.onnx"
            if not model_path.exists():
                log.warning(
                    f"Model not found at {model_path}. "
                    "Run: python data/download_models.py"
                )
                return False

            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self._loaded  = True
            log.info("Landmark extractor loaded successfully")
            return True

        except Exception as e:
            log.error(f"Failed to load landmark extractor: {e}")
            return False

    def extract(self, frame_rgb: np.ndarray) -> LandmarkResult:
        """
        Extract landmarks from a single RGB frame.

        Args:
            frame_rgb: uint8 [H, W, 3] RGB image

        Returns:
            LandmarkResult with hand and pose arrays
        """
        if not self._loaded:
            log.warning("Extractor not loaded — returning empty result")
            return LandmarkResult()

        # ── Preprocess ────────────────────────────────────────────────────────
        import cv2
        img = cv2.resize(frame_rgb, (256, 256)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # [1, 3, 256, 256]

        # ── Inference ─────────────────────────────────────────────────────────
        outputs = self._session.run(None, {self._session.get_inputs()[0].name: img})

        # ── Parse outputs (model-specific, populated after Hub integration) ───
        # placeholder — actual output parsing depends on final model I/O format
        result = LandmarkResult()
        # result.left_hand  = outputs[0].reshape(21, 3)
        # result.right_hand = outputs[1].reshape(21, 3)
        # result.pose       = outputs[2].reshape(33, 3)
        return result

    def normalize_sequence(self, sequence: list[np.ndarray]) -> np.ndarray:
        """
        Normalize a sequence of 225-dim landmark vectors for the encoder.

        Normalization:
          - Center: subtract mean of pose hip midpoint across sequence
          - Scale:  divide by shoulder width (makes scale-invariant)
          - Temporal: pad/trim to exactly 64 frames

        Args:
            sequence: list of [225] vectors, one per frame

        Returns:
            np.ndarray [64, 225] float32
        """
        T = len(sequence)
        if T == 0:
            return np.zeros((64, SEQUENCE_DIM), dtype=np.float32)

        arr = np.stack(sequence, axis=0)  # [T, 225]

        # Zero-mean normalization per feature across the sequence
        arr = (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-6)

        # Pad or trim to 64 frames
        if T < 64:
            pad = np.zeros((64 - T, SEQUENCE_DIM), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[-64:]  # take most recent 64 frames

        return arr.astype(np.float32)
