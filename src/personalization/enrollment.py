"""
User Enrollment — 5-Minute Personalization Setup
==================================================
Guides the user through signing 30 reference signs.
Extracts embeddings, computes a 64-dim user profile vector.
Stores encrypted in local SQLite DB.

This is one of SignBridge's three core differentiators:
the system adapts to each individual's signing style.
"""

import logging
import json
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np

log = logging.getLogger("signbridge.personalization.enrollment")

# 30 reference signs chosen to cover key hand configurations
ENROLLMENT_SIGNS = [
    # Open hand variants
    "HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES",
    # Closed fist variants
    "NO", "STOP", "WAIT", "WANT", "NEED",
    # Pointing variants
    "I", "YOU", "HERE", "WHERE", "WHAT",
    # Two-hand variants
    "HELP", "UNDERSTAND", "REPEAT", "TOGETHER", "BOTH",
    # Motion variants
    "COME", "GO", "UP", "DOWN", "AGAIN",
    # Number shapes
    "ONE", "TWO", "THREE", "FOUR", "FIVE",
]

REPS_PER_SIGN = 3  # Record each sign 3 times


class EnrollmentSession:
    """
    Manages the guided enrollment flow.

    Callbacks:
        on_prompt:   Callable[[str, int, int], None]  (sign, current, total)
        on_capture:  Callable[[int, int], None]        (rep, total_reps)
        on_complete: Callable[[np.ndarray], None]       (user_embedding)
    """

    def __init__(
        self,
        landmark_extractor,
        camera_buffer,
        on_prompt:   Optional[Callable[[str, int, int], None]] = None,
        on_capture:  Optional[Callable[[int, int], None]]      = None,
        on_complete: Optional[Callable[[np.ndarray], None]]    = None,
    ):
        self.extractor   = landmark_extractor
        self.camera      = camera_buffer
        self.on_prompt   = on_prompt
        self.on_capture  = on_capture
        self.on_complete = on_complete

        self._embeddings: list[np.ndarray] = []
        self._completed = False

    def run(self) -> Optional[np.ndarray]:
        """
        Run the full enrollment flow (blocking).
        Returns the 64-dim user embedding on success, None on failure.
        """
        log.info(f"Starting enrollment: {len(ENROLLMENT_SIGNS)} signs × {REPS_PER_SIGN} reps")
        total = len(ENROLLMENT_SIGNS)

        for i, sign in enumerate(ENROLLMENT_SIGNS):
            if self.on_prompt:
                self.on_prompt(sign, i + 1, total)

            for rep in range(REPS_PER_SIGN):
                if self.on_capture:
                    self.on_capture(rep + 1, REPS_PER_SIGN)

                # Collect 64 frames (~2.1 seconds) for this rep
                time.sleep(0.5)  # Brief pause between reps
                frames = self.camera.get_window()

                if len(frames) < 32:
                    log.warning(f"Insufficient frames for {sign} rep {rep + 1}")
                    continue

                # Extract landmarks and encode
                landmark_seq = []
                for frame in frames:
                    result = self.extractor.extract(frame)
                    landmark_seq.append(result.to_flat_vector())

                if not landmark_seq:
                    continue

                # Compute embedding for this rep (mean-pooled landmarks)
                seq_arr = np.stack(landmark_seq, axis=0)  # [T, 225]
                emb     = seq_arr.mean(axis=0)             # [225] — simple prototype
                self._embeddings.append(emb)

            time.sleep(0.3)  # Pause between signs

        if len(self._embeddings) < 10:
            log.error("Too few enrollment embeddings captured — enrollment failed")
            return None

        user_embedding = self._compute_user_embedding()
        log.info(f"Enrollment complete: {len(self._embeddings)} samples → {user_embedding.shape} embedding")

        if self.on_complete:
            self.on_complete(user_embedding)

        self._completed = True
        return user_embedding

    def _compute_user_embedding(self) -> np.ndarray:
        """
        Compute a 64-dim user profile vector from collected embeddings.

        Method:
          1. Stack all enrollment embeddings [N, 225]
          2. Reduce to 64-dim via PCA (top 64 principal components)
          3. Store the mean of projected embeddings as the user vector

        In production: use the temporal encoder's embedding space directly.
        For the prototype: PCA over raw landmark embeddings.
        """
        stack = np.stack(self._embeddings, axis=0).astype(np.float32)  # [N, 225]

        # Center
        mean = stack.mean(axis=0, keepdims=True)
        centered = stack - mean

        # PCA via SVD — reduce to 64 dims
        n_components = min(64, centered.shape[0] - 1, centered.shape[1])
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            components = Vt[:n_components]               # [64, 225]
            projected  = centered @ components.T         # [N, 64]
            user_vec   = projected.mean(axis=0)          # [64]
        except np.linalg.LinAlgError:
            log.warning("SVD failed — using mean embedding as user vector")
            user_vec = centered.mean(axis=0)[:64]

        # Normalize to unit sphere
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm

        return user_vec.astype(np.float32)  # [64]
