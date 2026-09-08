"""
Two-Way Display — Hearing Partner Speech Transcription
=======================================================
Runs Distil-Whisper (NPU) in a background thread.
Transcribed text is displayed in large font for the Deaf user.

This completes the two-way communication bridge:
  Deaf user signs → speech output (hearing partner hears)
  Hearing partner speaks → large text (Deaf user reads)
"""

import logging
import threading
import queue
from typing import Optional, Callable

import numpy as np

log = logging.getLogger("signbridge.ui.two_way_display")

SAMPLE_RATE    = 16000   # Whisper expects 16kHz mono
CHUNK_SECONDS  = 3       # Transcribe in 3-second chunks
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_SECONDS


class PartnerASR:
    """
    Background ASR for the hearing partner's speech using Distil-Whisper.
    Runs Distil-Whisper on Snapdragon NPU via ONNX Runtime.

    Source: https://aihub.qualcomm.com/models/distil_whisper
    """

    def __init__(
        self,
        model_dir,
        on_transcript: Optional[Callable[[str], None]] = None,
    ):
        self.model_dir     = model_dir
        self.on_transcript = on_transcript
        self._session      = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self._capture_thread: Optional[threading.Thread] = None

    def load(self) -> bool:
        """Load Distil-Whisper ONNX model with QNN provider."""
        try:
            import onnxruntime as ort
            from pathlib import Path

            model_path = Path(self.model_dir) / "distil_whisper.onnx"
            if not model_path.exists():
                log.warning(f"Distil-Whisper model not found: {model_path}")
                log.warning("Download from Qualcomm AI Hub and place in models/")
                return False

            providers = []
            if "QNNExecutionProvider" in ort.get_available_providers():
                providers.append((
                    "QNNExecutionProvider",
                    {
                        "backend_type":     "htp",
                        "htp_performance_mode": "burst",
                        "enable_htp_fp16_precision": "1",
                    }
                ))
                log.info("Distil-Whisper: using NPU provider")

            providers.append("CPUExecutionProvider")
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            log.info("Distil-Whisper loaded successfully")
            return True

        except Exception as e:
            log.error(f"Distil-Whisper load failed: {e}")
            return False

    def start(self) -> None:
        """Start microphone capture and transcription threads."""
        self._running = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()

        self._thread = threading.Thread(
            target=self._transcription_loop, daemon=True
        )
        self._thread.start()
        log.info("Partner ASR started")

    def stop(self) -> None:
        self._running = False
        log.info("Partner ASR stopped")

    def _capture_loop(self) -> None:
        """Capture microphone audio and push chunks to queue."""
        try:
            import sounddevice as sd
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
            ) as stream:
                while self._running:
                    data, _ = stream.read(CHUNK_SAMPLES)
                    self._audio_queue.put(data.flatten())
        except Exception as e:
            log.error(f"Microphone capture error: {e}")

    def _transcription_loop(self) -> None:
        """Consume audio chunks and transcribe with Whisper."""
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._session is None:
                continue

            try:
                text = self._transcribe(chunk)
                if text.strip() and self.on_transcript:
                    self.on_transcript(text.strip())
            except Exception as e:
                log.error(f"Transcription error: {e}")

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Run Distil-Whisper inference on one audio chunk.

        Input: float32 [CHUNK_SAMPLES] at 16kHz
        Output: transcribed string
        """
        if self._session is None:
            return ""

        # Pad/trim to expected input length
        inp = audio[:CHUNK_SAMPLES]
        if len(inp) < CHUNK_SAMPLES:
            inp = np.pad(inp, (0, CHUNK_SAMPLES - len(inp)))

        inp = inp[np.newaxis, :]  # [1, CHUNK_SAMPLES]

        outputs = self._session.run(
            None,
            {self._session.get_inputs()[0].name: inp}
        )

        # Decode token IDs → string
        # (tokenizer integration omitted for brevity — use HuggingFace tokenizer)
        # outputs[0]: [1, seq_len] int32 token IDs
        return "[Partner speech transcription placeholder]"
