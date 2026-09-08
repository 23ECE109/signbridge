"""
Data Preprocessing Pipeline
=============================
Converts raw collected video clips into normalized landmark sequences
ready for model training.

Input:  data/collected/{split}/{sign_label}/{clip_id}.mp4
Output: data/processed/{split}/{clip_id}.npy   (float32 [64, 225])
                                {clip_id}.json  ({"label": "HELP", "signer": "s01", ...})

Run:
    python data/preprocessing.py --split train
    python data/preprocessing.py --split val
    python data/preprocessing.py --split test
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("signbridge.data.preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

ROOT         = Path(__file__).resolve().parent.parent
COLLECTED    = ROOT / "data" / "collected"
PROCESSED    = ROOT / "data" / "processed"
SEQ_LEN      = 64
LANDMARK_DIM = 225   # 21 LH + 21 RH + 33 pose, each × 3


def extract_landmarks_from_video(video_path: Path) -> list[np.ndarray]:
    """
    Extract per-frame landmark vectors from a video clip.
    Uses MediaPipe Holistic running on CPU (preprocessing, not real-time).

    Returns: list of float32 [225] vectors, one per frame.
    """
    try:
        import cv2
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
        cap = cv2.VideoCapture(str(video_path))
        landmarks = []

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = holistic.process(frame_rgb)
                vec       = _pack_landmarks(results)
                landmarks.append(vec)

        cap.release()
        return landmarks

    except ImportError:
        log.error("MediaPipe not installed: pip install mediapipe")
        return []
    except Exception as e:
        log.error(f"Landmark extraction failed for {video_path.name}: {e}")
        return []


def _pack_landmarks(results) -> np.ndarray:
    """Pack MediaPipe holistic results into a flat [225] float32 vector."""
    def _hand(hand_landmarks) -> np.ndarray:
        if hand_landmarks is None:
            return np.zeros(21 * 3, dtype=np.float32)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    def _pose(pose_landmarks) -> np.ndarray:
        if pose_landmarks is None:
            return np.zeros(33 * 3, dtype=np.float32)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark[:33]],
            dtype=np.float32
        ).flatten()

    lh = _hand(results.left_hand_landmarks)
    rh = _hand(results.right_hand_landmarks)
    ps = _pose(results.pose_landmarks)
    return np.concatenate([lh, rh, ps])  # [225]


def normalize_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """
    Normalize and pad/trim a landmark sequence to [64, 225].

    Steps:
      1. Zero-mean per feature across the sequence
      2. Scale-normalize by std
      3. Pad with zeros or trim to exactly SEQ_LEN frames
    """
    if not frames:
        return np.zeros((SEQ_LEN, LANDMARK_DIM), dtype=np.float32)

    arr = np.stack(frames, axis=0).astype(np.float32)   # [T, 225]

    # Normalize
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-6
    arr  = (arr - mean) / std

    T = arr.shape[0]
    if T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, LANDMARK_DIM), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif T > SEQ_LEN:
        # Uniformly sample SEQ_LEN frames to preserve temporal structure
        indices = np.linspace(0, T - 1, SEQ_LEN, dtype=int)
        arr     = arr[indices]

    return arr.astype(np.float32)   # [64, 225]


def augment(seq: np.ndarray) -> list[np.ndarray]:
    """
    Generate 5 augmented variants of a landmark sequence.
    Returns list of [64, 225] arrays including the original.
    """
    variants = [seq]

    # 1. Landmark jitter
    jitter = seq + np.random.randn(*seq.shape).astype(np.float32) * 0.01
    variants.append(jitter)

    # 2. Horizontal flip (mirror x-coordinates)
    flipped = seq.copy()
    for i in range(0, LANDMARK_DIM, 3):   # every x coordinate
        flipped[:, i] = 1.0 - flipped[:, i]
    variants.append(flipped)

    # 3. Speed up (drop every 4th frame, re-pad)
    fast_frames = [seq[i] for i in range(seq.shape[0]) if i % 4 != 0]
    variants.append(normalize_sequence(fast_frames))

    # 4. Slow down (repeat every other frame)
    slow_frames = []
    for frame in seq:
        slow_frames.append(frame)
        slow_frames.append(frame)
    variants.append(normalize_sequence(slow_frames))

    return variants


def process_split(split: str, augment_data: bool = True):
    """Process all clips for a given split."""
    in_dir  = COLLECTED / split
    out_dir = PROCESSED / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        log.error(f"Input directory not found: {in_dir}")
        log.error(f"Record data first: python data/collect.py --split {split}")
        return

    clip_paths = list(in_dir.rglob("*.mp4")) + list(in_dir.rglob("*.avi"))
    log.info(f"Processing {len(clip_paths)} clips for split '{split}'...")

    processed = 0
    skipped   = 0

    for clip_path in clip_paths:
        # Parse label from directory name: collected/train/HELP/s01_001.mp4
        sign_label = clip_path.parent.name.upper()

        t0     = time.perf_counter()
        frames = extract_landmarks_from_video(clip_path)
        elapsed = (time.perf_counter() - t0) * 1000

        if len(frames) < 10:
            log.warning(f"  Skipping {clip_path.name} — too few frames ({len(frames)})")
            skipped += 1
            continue

        seq = normalize_sequence(frames)
        variants = augment(seq) if augment_data and split == "train" else [seq]

        for i, var in enumerate(variants):
            clip_id  = f"{clip_path.stem}_aug{i}" if i > 0 else clip_path.stem
            npy_out  = out_dir / f"{clip_id}.npy"
            json_out = out_dir / f"{clip_id}.json"

            np.save(str(npy_out), var)
            json_out.write_text(json.dumps({
                "label":      sign_label,
                "source":     clip_path.name,
                "augment_id": i,
                "n_frames":   len(frames),
                "split":      split,
            }))

        processed += 1
        if processed % 50 == 0:
            log.info(f"  Processed {processed}/{len(clip_paths)} clips...")

    log.info(f"Split '{split}' done: {processed} processed, {skipped} skipped")
    if augment_data and split == "train":
        log.info(f"  Total augmented samples: ~{processed * 5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SignBridge dataset")
    parser.add_argument("--split",     choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--no-augment", action="store_true", help="Skip augmentation")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        process_split(s, augment_data=not args.no_augment)

    log.info("\nPreprocessing complete.")
    log.info("Next: python training/train_encoder.py")
