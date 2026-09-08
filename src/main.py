"""
SignBridge — Main Entry Point
==============================
Launch: python setup.py          (recommended — installs deps first)
   OR:  python src/main.py       (if deps already installed)
"""

import sys
import os
import logging
from pathlib import Path

# ── Ensure src/ is on the path ────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-32s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("signbridge.main")


def check_deps() -> bool:
    missing = []
    for pkg, imp in [("mediapipe", "mediapipe"), ("opencv-python", "cv2"), ("PyQt6", "PyQt6")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(f"Missing packages: {missing}")
        log.error("Run:  python setup.py  — to install everything automatically")
        return False
    return True


def main():
    log.info("=" * 55)
    log.info("  SignBridge — Real-Time Sign Language Communication")
    log.info("  Snapdragon AI Lab Build & Present Challenge")
    log.info("=" * 55)

    if not check_deps():
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication
    from ui.app import SignBridgeApp

    app = QApplication(sys.argv)
    app.setApplicationName("SignBridge")
    app.setApplicationVersion("1.0.0")

    window = SignBridgeApp()
    window.show()

    log.info("SignBridge window opened ✅")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
