"""
Data Augmentation Utilities
=============================
Standalone augmentation functions for landmark sequences.
Used by preprocessing.py during training data generation.
"""

import numpy as np

SEQ_LEN      = 64
LANDMARK_DIM = 225


def jitter(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to landmark coordinates."""
    return seq + np.random.randn(*seq.shape).astype(np.float32) * sigma


def horizontal_flip(seq: np.ndarray) -> np.ndarray:
    """Mirror x-coordinates (simulates left-handed signer)."""
    flipped = seq.copy()
    for i in range(0, LANDMARK_DIM, 3):
        flipped[:, i] = 1.0 - flipped[:, i]
    return flipped


def speed_up(seq: np.ndarray, drop_every: int = 4) -> np.ndarray:
    """Drop frames to simulate faster signing."""
    frames = [seq[i] for i in range(seq.shape[0]) if i % drop_every != 0]
    return _pad_or_trim(frames)


def slow_down(seq: np.ndarray) -> np.ndarray:
    """Repeat frames to simulate slower signing."""
    frames = []
    for frame in seq:
        frames.append(frame)
        frames.append(frame)
    return _pad_or_trim(frames)


def temporal_shift(seq: np.ndarray, shift: int = 5) -> np.ndarray:
    """Shift sequence start by N frames (simulates segmentation offset)."""
    shift = min(abs(shift), seq.shape[0] // 4)
    return np.roll(seq, shift, axis=0)


def rotation_2d(seq: np.ndarray, angle_deg: float = 10.0) -> np.ndarray:
    """Rotate 2D projection of landmarks by angle degrees."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = seq.copy()
    for i in range(0, LANDMARK_DIM, 3):
        x = rotated[:, i]
        y = rotated[:, i + 1]
        rotated[:, i]     = cos_a * x - sin_a * y
        rotated[:, i + 1] = sin_a * x + cos_a * y
    return rotated


def apply_all(seq: np.ndarray) -> list[np.ndarray]:
    """Apply all augmentations. Returns list of 6 variants (original + 5)."""
    return [
        seq,
        jitter(seq),
        horizontal_flip(seq),
        speed_up(seq),
        slow_down(seq),
        temporal_shift(seq, shift=np.random.randint(-5, 6)),
    ]


def _pad_or_trim(frames: list[np.ndarray]) -> np.ndarray:
    """Normalize frame list to exactly SEQ_LEN frames."""
    if not frames:
        return np.zeros((SEQ_LEN, LANDMARK_DIM), dtype=np.float32)
    arr = np.stack(frames, axis=0).astype(np.float32)
    T   = arr.shape[0]
    if T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, LANDMARK_DIM), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif T > SEQ_LEN:
        indices = np.linspace(0, T - 1, SEQ_LEN, dtype=int)
        arr     = arr[indices]
    return arr
