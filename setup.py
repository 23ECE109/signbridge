"""
SignBridge — One-Command Setup & Launcher
==========================================
Run this once to install all dependencies and launch the app:

    python setup.py

Or just install deps:
    python setup.py --install-only

Or just run (after install):
    python setup.py --run-only
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def banner():
    print("\n" + "="*60)
    print("  🤝  SignBridge — Real-Time Sign Language Communication")
    print("  Snapdragon AI Lab Hackathon")
    print("="*60 + "\n")


def check_python():
    major, minor = sys.version_info[:2]
    print(f"Python {major}.{minor} detected", end="  ")
    if major < 3 or minor < 10:
        print("❌  Python 3.10+ required")
        sys.exit(1)
    print("✅")


def install_deps():
    print("\n📦  Installing dependencies...")
    req = ROOT / "requirements.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"],
        capture_output=False
    )
    if result.returncode != 0:
        print("❌  pip install failed. Try manually: pip install -r requirements.txt")
        sys.exit(1)
    print("✅  Dependencies installed\n")


def check_camera():
    print("🎥  Checking camera...", end="  ")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            print("✅  Camera found")
            return True
        else:
            print("⚠️   No camera detected — connect a webcam")
            return False
    except Exception:
        print("⚠️   Could not check camera")
        return False


def check_mediapipe():
    print("🤖  Checking MediaPipe...", end="  ")
    try:
        import mediapipe
        print(f"✅  v{mediapipe.__version__}")
        return True
    except ImportError:
        print("❌  Not installed")
        return False


def check_whisper():
    print("🎤  Checking Whisper...", end="  ")
    try:
        import whisper
        print("✅  openai-whisper ready (model downloads on first run ~150 MB)")
        return True
    except ImportError:
        print("⚠️   Not installed — partner ASR disabled (pip install openai-whisper)")
        return False


def check_tts():
    print("🔊  Checking TTS...", end="  ")
    try:
        import pyttsx3
        e = pyttsx3.init()
        e.stop()
        print("✅  pyttsx3 (Windows SAPI5) ready")
        return True
    except Exception:
        print("⚠️   TTS unavailable — translations shown as text only")
        return False


def run_app():
    print("\n🚀  Launching SignBridge...\n")
    src = ROOT / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)
    result = subprocess.run(
        [sys.executable, str(src / "main.py")],
        env=env,
        cwd=str(ROOT)
    )
    return result.returncode


def main():
    banner()

    parser = argparse.ArgumentParser()
    parser.add_argument("--install-only", action="store_true", help="Install deps and exit")
    parser.add_argument("--run-only",     action="store_true", help="Run app without installing")
    parser.add_argument("--check-only",   action="store_true", help="Check environment only")
    args = parser.parse_args()

    check_python()

    if not args.run_only:
        install_deps()

    # Environment checks
    print("🔍  Checking environment...\n")
    cam = check_camera()
    mp  = check_mediapipe()
    wh  = check_whisper()
    tts = check_tts()

    print()
    if not mp:
        print("❌  MediaPipe is required. Run: pip install mediapipe")
        sys.exit(1)
    if not cam:
        print("⚠️   No camera — the app will open but video will be blank")

    if args.install_only or args.check_only:
        print("\n✅  Setup complete. Run 'python setup.py --run-only' to launch.")
        return

    print("\n" + "-"*60)
    print("  Controls:")
    print("  • Sign in front of camera — translation appears automatically")
    print("  • Partner speaks → transcript shown on right panel")
    print("  • 'Incorrect' button → correct a wrong translation")
    print("  • Topic selector → bias for Medical / Emergency / etc.")
    print("  • Press Speak Last to re-hear the last translation")
    print("-"*60 + "\n")

    sys.exit(run_app())


if __name__ == "__main__":
    main()
