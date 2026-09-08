"""
Text-to-Speech — pyttsx3 (works immediately, no download needed)
                + piper-tts (optional, higher quality)
=================================================================
pyttsx3 uses Windows built-in SAPI5 voices — works with zero setup.
piper-tts is used if its binary exists in models/piper/.

For hackathon demo: pyttsx3 works out of the box on Windows.
"""

import logging
import threading
from typing import Optional

log = logging.getLogger("signbridge.models.tts")


class TTSEngine:
    def __init__(self):
        self._engine    = None
        self._lock      = threading.Lock()
        self._available = False
        self._use_piper = False

    def load(self) -> bool:
        # Try piper first (better quality)
        from pathlib import Path
        piper_exe = Path(__file__).parent.parent.parent / "models" / "piper" / "piper.exe"
        piper_mdl = Path(__file__).parent.parent.parent / "models" / "piper" / "en_US-lessac-medium.onnx"

        if piper_exe.exists() and piper_mdl.exists():
            self._piper_exe = str(piper_exe)
            self._piper_mdl = str(piper_mdl)
            self._piper_cfg = str(piper_mdl) + ".json"
            self._use_piper  = True
            self._available  = True
            log.info("TTS: piper-tts loaded ✅")
            return True

        # Fall back to pyttsx3 (Windows SAPI5, zero install)
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 160)   # slightly slower for clarity
            self._engine.setProperty("volume", 0.95)
            # Pick a clear voice if available
            voices = self._engine.getProperty("voices")
            for v in voices:
                if "zira" in v.name.lower() or "david" in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break
            self._available = True
            log.info("TTS: pyttsx3 (Windows SAPI5) loaded ✅")
            return True
        except ImportError:
            log.error("pyttsx3 not installed. Run: pip install pyttsx3")
            return False
        except Exception as e:
            log.error(f"TTS load failed: {e}")
            return False

    def speak(self, text: str, blocking: bool = False):
        if not self._available:
            log.warning(f"TTS unavailable — would say: {text!r}")
            return
        if blocking:
            self._do_speak(text)
        else:
            t = threading.Thread(target=self._do_speak, args=(text,), daemon=True)
            t.start()

    def _do_speak(self, text: str):
        with self._lock:
            try:
                if self._use_piper:
                    self._speak_piper(text)
                else:
                    self._speak_pyttsx3(text)
            except Exception as e:
                log.error(f"TTS speak error: {e}")

    def _speak_pyttsx3(self, text: str):
        import pyttsx3
        # Re-init per call to avoid threading issues
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()

    def _speak_piper(self, text: str):
        import subprocess, tempfile, os
        import sounddevice as sd
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                [self._piper_exe, "--model", self._piper_mdl,
                 "--config", self._piper_cfg, "--output_file", tmp],
                input=text.encode(), capture_output=True, timeout=10
            )
            data, sr = sf.read(tmp, dtype="float32")
            sd.play(data, sr, blocking=True)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._available
