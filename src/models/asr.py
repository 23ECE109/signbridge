"""
Partner ASR — Whisper (openai-whisper, CPU)
===========================================
Transcribes the hearing partner's speech in real time using
openai-whisper running on CPU. No GPU needed for demo.

Uses whisper "base" model (~150 MB, downloads once automatically).
Runs in a background thread, emits transcript via callback.

For Snapdragon production: swap for Distil-Whisper via QNN.
"""

import logging
import threading
import queue
import time
from typing import Optional, Callable

import numpy as np

log = logging.getLogger("signbridge.models.asr")

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 4       # transcribe in 4-second chunks
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


class WhisperASR:
    """
    Real-time ASR using openai-whisper (CPU).
    Emits transcript strings via on_transcript callback.
    """

    def __init__(self, model_size: str = "base",
                 on_transcript: Optional[Callable[[str], None]] = None):
        self.model_size    = model_size
        self.on_transcript = on_transcript
        self._model        = None
        self._loaded       = False
        self._running      = False
        self._audio_q: queue.Queue = queue.Queue(maxsize=10)
        self._capture_thread: Optional[threading.Thread] = None
        self._transcribe_thread: Optional[threading.Thread] = None

    def load(self) -> bool:
        try:
            import whisper
            log.info(f"Loading Whisper '{self.model_size}' model (downloads ~150 MB on first run)...")
            self._model  = whisper.load_model(self.model_size)
            self._loaded = True
            log.info("Whisper loaded ✅")
            return True
        except ImportError:
            log.error("openai-whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            log.error(f"Whisper load failed: {e}")
            return False

    def start(self):
        if not self._loaded:
            log.warning("Whisper not loaded — ASR disabled")
            return
        self._running = True
        self._capture_thread    = threading.Thread(target=self._capture_loop,    daemon=True, name="sb-mic")
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True, name="sb-asr")
        self._capture_thread.start()
        self._transcribe_thread.start()
        log.info("Whisper ASR started (listening for partner speech)")

    def stop(self):
        self._running = False

    # ── Capture ───────────────────────────────────────────────────────────────

    def _capture_loop(self):
        try:
            import sounddevice as sd
            log.info("Microphone capture started")
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
                while self._running:
                    data, _ = stream.read(CHUNK_SAMPLES)
                    chunk   = data.flatten()
                    # Skip near-silent chunks (noise gate)
                    if np.abs(chunk).mean() < 0.005:
                        continue
                    try:
                        self._audio_q.put_nowait(chunk)
                    except queue.Full:
                        pass  # Drop if transcription can't keep up
        except ImportError:
            log.error("sounddevice not installed. Run: pip install sounddevice")
        except Exception as e:
            log.error(f"Microphone error: {e}")

    # ── Transcribe ────────────────────────────────────────────────────────────

    def _transcribe_loop(self):
        while self._running:
            try:
                chunk = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._model is None:
                continue

            try:
                result = self._model.transcribe(
                    chunk,
                    language      = "en",
                    fp16          = False,   # CPU mode
                    condition_on_previous_text = False,
                )
                text = result.get("text", "").strip()
                if text and len(text) > 2 and self.on_transcript:
                    log.info(f"🎤 Partner: {text!r}")
                    self.on_transcript(text)
            except Exception as e:
                log.error(f"Transcription error: {e}")
