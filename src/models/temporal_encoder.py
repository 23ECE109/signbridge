"""
Temporal Encoder — 4-Layer Transformer
========================================
Encodes a [64, 225] landmark sequence into a [256] embedding.

Architecture:
  Input: [batch, 64, 225]  (64 frames × 225 landmark features)
  Positional embedding → 4× Transformer encoder layers (d=256, heads=8)
  Global average pooling → [batch, 256]

Trained on: WLASL (pre-training) + custom 150-sign dataset (fine-tune)
Exported to ONNX INT8 for Snapdragon NPU deployment.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("signbridge.models.temporal_encoder")

D_MODEL    = 256
SEQ_LEN    = 64
INPUT_DIM  = 225
USER_DIM   = 64


class TemporalEncoder:
    """
    Wraps the ONNX temporal encoder for inference.
    Also provides the personalization fusion layer.
    """

    def __init__(self, model_dir: Optional[Path] = None, use_npu: bool = True):
        self.model_dir = model_dir or Path(__file__).parent.parent.parent / "models"
        self.use_npu   = use_npu
        self._enc_session  = None
        self._fuse_session = None
        self._loaded   = False

    def load(self) -> bool:
        """Load encoder and fusion MLP ONNX models."""
        try:
            import onnxruntime as ort
            providers = self._get_providers(ort)

            enc_path  = self.model_dir / "temporal_encoder.onnx"
            fuse_path = self.model_dir / "personalization_mlp.onnx"

            if not enc_path.exists():
                log.warning(f"Encoder model not found: {enc_path}")
                log.warning("Train and export first: python training/train_encoder.py")
                return False

            self._enc_session = ort.InferenceSession(
                str(enc_path), providers=providers
            )
            log.info("Temporal encoder loaded")

            if fuse_path.exists():
                self._fuse_session = ort.InferenceSession(
                    str(fuse_path), providers=[("CPUExecutionProvider", {})]
                )
                log.info("Personalization MLP loaded")
            else:
                log.warning("Personalization MLP not found — will use identity fusion")

            self._loaded = True
            return True

        except Exception as e:
            log.error(f"Encoder load failed: {e}")
            return False

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        """
        Encode a normalized landmark sequence.

        Args:
            sequence: float32 [64, 225]

        Returns:
            embedding: float32 [256]
        """
        if not self._loaded or self._enc_session is None:
            return np.zeros(D_MODEL, dtype=np.float32)

        inp = sequence[np.newaxis, ...]  # [1, 64, 225]
        out = self._enc_session.run(
            None,
            {self._enc_session.get_inputs()[0].name: inp}
        )
        return out[0].squeeze(0)  # [256]

    def fuse_user(self, seq_emb: np.ndarray, user_emb: np.ndarray) -> np.ndarray:
        """
        Fuse sequence embedding with user profile embedding.

        Args:
            seq_emb:  float32 [256]
            user_emb: float32 [64]

        Returns:
            fused: float32 [256]
        """
        if self._fuse_session is None:
            # Identity fallback: ignore user embedding
            return seq_emb

        combined = np.concatenate([seq_emb, user_emb], axis=0)[np.newaxis, :]  # [1, 320]
        out = self._fuse_session.run(
            None,
            {self._fuse_session.get_inputs()[0].name: combined}
        )
        return out[0].squeeze(0)  # [256]

    def _get_providers(self, ort) -> list:
        providers = []
        if self.use_npu and "QNNExecutionProvider" in ort.get_available_providers():
            providers.append((
                "QNNExecutionProvider",
                {
                    "backend_type":     "htp",
                    "htp_performance_mode": "burst",
                    "enable_htp_fp16_precision": "1",
                }
            ))
        providers.append("CPUExecutionProvider")
        return providers
