"""
SignBridge — Main Application Entry Point
==========================================
Private, Personalized, On-Device Communication Intelligence for Sign Language Users
Snapdragon AI Lab Build & Present Challenge

Architecture:
  Camera → Landmarks (NPU) → Temporal Encoder → Personalization Layer
  → Context Engine → Translation + Confidence → TTS / Disambiguation
  Microphone → Distil-Whisper (NPU) → Text Display

All inference runs on-device. Nothing leaves the device.
"""

import sys
import os
import logging
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signbridge.main")


def check_environment() -> bool:
    """Verify required dependencies and model files are present."""
    missing = []

    # Check ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        log.info(f"ONNX Runtime {ort.__version__} — available providers: {providers}")
        if "QNNExecutionProvider" in providers:
            log.info("✅  Qualcomm NPU (QNN) provider detected")
        else:
            log.warning("⚠️   QNNExecutionProvider not found — will fall back to CPU")
    except ImportError:
        missing.append("onnxruntime")

    # Check OpenCV
    try:
        import cv2
        log.info(f"OpenCV {cv2.__version__} detected")
    except ImportError:
        missing.append("opencv-python")

    # Check PyQt6
    try:
        from PyQt6.QtWidgets import QApplication  # noqa: F401
        log.info("PyQt6 detected")
    except ImportError:
        missing.append("PyQt6")

    if missing:
        log.error(f"Missing dependencies: {missing}")
        log.error("Run: pip install -r requirements.txt")
        return False

    return True


def main():
    """Application entry point."""
    log.info("=" * 60)
    log.info("  SignBridge — Starting Up")
    log.info("  On-Device Sign Language Communication Intelligence")
    log.info("=" * 60)

    if not check_environment():
        sys.exit(1)

    # Import here so missing deps don't crash before the check
    from PyQt6.QtWidgets import QApplication
    from ui.app import SignBridgeApp

    app = QApplication(sys.argv)
    app.setApplicationName("SignBridge")
    app.setApplicationVersion("1.0.0")

    window = SignBridgeApp()
    window.show()

    log.info("Application window launched")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
