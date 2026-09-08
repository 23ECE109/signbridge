"""
Dataset Collection Script
==========================
Guided recording tool for the SignBridge custom 150-sign dataset.
Shows on-screen prompts for each sign, records from the camera,
and saves clips with quality metadata.

Usage:
    python data/collect.py --signer s01 --split train
    python data/collect.py --signer s02 --split val --signs HELP PAIN DOCTOR

Collection protocol per sign:
  - 5 repetitions
  - 3 background conditions (swap manually between sessions)
  - 2 lighting conditions
  - ~2.5 seconds per clip at 30 FPS = ~75 frames captured
"""

import argparse
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("signbridge.data.collect")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

ROOT       = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocabulary.json"
OUT_BASE   = ROOT / "data" / "collected"

REPS_PER_SIGN    = 5
CLIP_DURATION_S  = 2.5   # seconds per clip
FPS              = 30
FRAME_W          = 640
FRAME_H          = 480

# Colors (BGR for OpenCV)
GREEN  = (0, 220, 100)
YELLOW = (0, 200, 230)
RED    = (60, 60, 220)
WHITE  = (240, 240, 240)
DARK   = (20, 20, 40)


def load_vocabulary() -> list[str]:
    if not VOCAB_PATH.exists():
        log.error(f"Vocabulary not found: {VOCAB_PATH}")
        return []
    data  = json.loads(VOCAB_PATH.read_text())
    signs = []
    for cat in data.get("categories", {}).values():
        signs.extend(cat.get("signs", []))
    return signs


def record_clip(cap, sign_label: str, rep: int, out_path: Path) -> dict:
    """
    Record one clip of a sign.
    Shows countdown on-screen, then records for CLIP_DURATION_S seconds.
    Returns quality metadata dict.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        log.error("OpenCV not installed: pip install opencv-python")
        return {}

    n_frames = int(FPS * CLIP_DURATION_S)
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer   = cv2.VideoWriter(str(out_path), fourcc, FPS, (FRAME_W, FRAME_H))

    # ── Countdown (3 seconds) ──────────────────────────────────────────────
    for count in [3, 2, 1]:
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()
            cv2.putText(display, f"GET READY: {sign_label}", (20, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, WHITE, 2)
            cv2.putText(display, str(count), (FRAME_W // 2 - 30, FRAME_H // 2 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 5.0, YELLOW, 6)
            cv2.putText(display, f"Rep {rep}/{REPS_PER_SIGN}", (20, FRAME_H - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, WHITE, 1)
            cv2.imshow("SignBridge — Data Collection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                writer.release()
                return {}

    # ── Record ────────────────────────────────────────────────────────────
    frames_written = 0
    brightness_vals = []

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frames_written += 1
        brightness_vals.append(frame.mean())

        display = frame.copy()
        progress = int((frames_written / n_frames) * (FRAME_W - 40))
        cv2.rectangle(display, (20, FRAME_H - 30), (FRAME_W - 20, FRAME_H - 10), DARK, -1)
        cv2.rectangle(display, (20, FRAME_H - 30), (20 + progress, FRAME_H - 10), GREEN, -1)
        cv2.putText(display, f"RECORDING: {sign_label}  rep {rep}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, GREEN, 2)
        cv2.imshow("SignBridge — Data Collection", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()

    mean_brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.0
    quality_ok      = mean_brightness > 30.0 and frames_written >= int(n_frames * 0.9)

    return {
        "frames":     frames_written,
        "brightness": round(mean_brightness, 1),
        "quality_ok": quality_ok,
    }


def run_collection(signer_id: str, split: str, signs: list[str]):
    try:
        import cv2
    except ImportError:
        log.error("OpenCV not installed: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        log.error("Cannot open camera")
        return

    log.info(f"Starting collection — signer: {signer_id}, split: {split}, signs: {len(signs)}")
    log.info("Press Q at any time to quit. Press SPACE between signs to continue.")

    session_meta = []

    for sign_idx, sign_label in enumerate(signs):
        log.info(f"\n[{sign_idx + 1}/{len(signs)}]  Sign: {sign_label}")

        for rep in range(1, REPS_PER_SIGN + 1):
            clip_name = f"{signer_id}_{sign_label}_{rep:02d}.mp4"
            out_path  = OUT_BASE / split / sign_label / clip_name

            if out_path.exists():
                log.info(f"  Skipping existing: {clip_name}")
                continue

            meta = record_clip(cap, sign_label, rep, out_path)
            if not meta:
                log.warning("  Recording aborted")
                break

            status = "✅" if meta.get("quality_ok") else "⚠️ LOW QUALITY"
            log.info(f"  Rep {rep}: {meta['frames']} frames, brightness={meta['brightness']} {status}")

            session_meta.append({
                "signer":    signer_id,
                "sign":      sign_label,
                "rep":       rep,
                "split":     split,
                "clip_file": str(out_path.relative_to(ROOT)),
                **meta,
            })

            # Short pause between reps
            time.sleep(0.5)

        # Pause between signs — user presses SPACE or waits 2s
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "NEXT SIGN IN 2s — press SPACE to skip wait", (20, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, YELLOW, 1)
            cv2.imshow("SignBridge — Data Collection", frame)
        cv2.waitKey(2000)

    cap.release()
    cv2.destroyAllWindows()

    # Save session metadata
    meta_path = OUT_BASE / split / f"session_{signer_id}_{int(time.time())}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(session_meta, indent=2))
    log.info(f"\nSession metadata saved: {meta_path}")
    log.info(f"Total clips recorded: {len(session_meta)}")
    log.info("Next: python data/preprocessing.py --split train")


if __name__ == "__main__":
    all_signs = load_vocabulary()

    parser = argparse.ArgumentParser(description="Collect SignBridge dataset")
    parser.add_argument("--signer", required=True, help="Signer ID, e.g. s01")
    parser.add_argument("--split",  choices=["train", "val", "test"], default="train")
    parser.add_argument("--signs",  nargs="*", default=None,
                        help="Subset of signs to record (default: all vocabulary)")
    args = parser.parse_args()

    signs_to_record = args.signs if args.signs else all_signs
    if not signs_to_record:
        log.error("No signs found. Check data/vocabulary.json.")
    else:
        run_collection(args.signer, args.split, signs_to_record)
