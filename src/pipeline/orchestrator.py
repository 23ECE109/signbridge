"""
Pipeline Orchestrator — Real-Time Demo
=======================================
Connects:
  Camera → MediaPipe Landmarks → Gesture Classifier → Confidence →
  TTS Output / Disambiguation / Re-sign Request

Also runs Whisper ASR in parallel for partner speech → text display.

Emits results via callbacks (safe to connect to Qt signals).
"""

import logging
import threading
import time
from typing import Optional, Callable

import numpy as np

from pipeline.camera import CameraCapture
from pipeline.landmark_extractor import LandmarkExtractor
from models.gesture_classifier import GestureClassifier, label_to_sentence
from models.confidence import ConfidenceEstimator, ConfidenceTier
from personalization.correction_memory import CorrectionMemory
from context_engine.context_window import ContextWindow
from context_engine.topic_tag import TopicTag

log = logging.getLogger("signbridge.pipeline.orchestrator")

CLASSIFY_INTERVAL  = 1.2   # seconds between translation attempts
MIN_FRAMES_BUFFER  = 20    # minimum frames before classifying


class TranslationResult:
    def __init__(self, text, sentence, confidence, tier, alternatives, latency_ms):
        self.text         = text          # sign label e.g. "HELP"
        self.sentence     = sentence      # natural language e.g. "Help me, please."
        self.confidence   = confidence
        self.tier         = tier
        self.alternatives = alternatives  # [(label, conf)]
        self.latency_ms   = latency_ms

    def __repr__(self):
        return f"TranslationResult({self.sentence!r} conf={self.confidence:.2f} {self.tier.name})"


class SignBridgePipeline:
    """
    Central real-time pipeline orchestrator.

    Callbacks (called from background thread — marshal to Qt with signals):
        on_translation : Callable[[TranslationResult], None]
        on_frame       : Callable[[np.ndarray, any], None]  — (frame_rgb, landmark_result)
        on_fps_update  : Callable[[float], None]
        on_asr_text    : Callable[[str], None]
    """

    def __init__(self,
                 user_id:       str = "default",
                 on_translation: Optional[Callable] = None,
                 on_frame:       Optional[Callable] = None,
                 on_fps_update:  Optional[Callable] = None,
                 on_asr_text:    Optional[Callable] = None):

        self.user_id        = user_id
        self.on_translation = on_translation
        self.on_frame       = on_frame
        self.on_fps_update  = on_fps_update
        self.on_asr_text    = on_asr_text

        self.camera      = CameraCapture()
        self.extractor   = LandmarkExtractor(complexity=0)
        self.classifier  = GestureClassifier(history_len=20)
        self.confidence  = ConfidenceEstimator()
        self.corrections = CorrectionMemory(user_id)
        self.context     = ContextWindow()
        self.topic       = TopicTag()

        self._running              = False
        self._pipeline_thread: Optional[threading.Thread] = None
        self._last_classify_time   = 0.0
        self._last_label           = ""
        self._same_sign_count      = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        log.info("Starting SignBridge real-time pipeline...")

        if not self.extractor.load():
            log.error("MediaPipe failed to load — install mediapipe: pip install mediapipe")
            return False

        self.corrections.load()

        if not self.camera.start():
            log.error("Camera failed to open")
            return False

        self._running = True
        self._pipeline_thread = threading.Thread(
            target=self._loop, daemon=True, name="sb-pipeline"
        )
        self._pipeline_thread.start()
        log.info("Pipeline running ✅")
        return True

    def stop(self):
        self._running = False
        self.camera.stop()
        self.extractor.close()
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=3.0)
        log.info("Pipeline stopped")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            frames = self.camera.buffer.get_window()

            if not frames:
                time.sleep(0.03)
                continue

            # Use the most recent frame for landmark extraction
            latest_frame = frames[-1]
            result       = self.extractor.extract(latest_frame)
            self.classifier.update(result)

            # Emit frame + landmarks for UI rendering
            if self.on_frame:
                self.on_frame(latest_frame, result)

            if self.on_fps_update:
                self.on_fps_update(self.camera.fps)

            # Classify on interval
            now = time.perf_counter()
            if (now - self._last_classify_time < CLASSIFY_INTERVAL or
                    len(frames) < MIN_FRAMES_BUFFER):
                time.sleep(0.03)
                continue

            self._last_classify_time = now
            t0 = time.perf_counter()

            gesture = self.classifier.classify()
            if gesture is None:
                time.sleep(0.05)
                continue

            # Avoid repeating the same sign endlessly
            if gesture.label == self._last_label:
                self._same_sign_count += 1
                if self._same_sign_count > 3:
                    time.sleep(0.1)
                    continue
            else:
                self._same_sign_count = 0
                self._last_label = gesture.label

            sentence = label_to_sentence(gesture.label)

            # Check correction memory
            dummy_emb = np.zeros(64, dtype=np.float32)
            override  = self.corrections.lookup(dummy_emb, threshold=0.99)  # high threshold = rarely fires unless explicit

            # Build top-k as (text, prob) for confidence estimator
            top_k = [(sentence, gesture.confidence)]
            for lbl, c in gesture.alternatives:
                top_k.append((label_to_sentence(lbl), c))

            conf_score, tier = self.confidence.score(top_k, correction_override=override)
            latency = (time.perf_counter() - t0) * 1000

            tr = TranslationResult(
                text        = gesture.label,
                sentence    = override if override else sentence,
                confidence  = conf_score,
                tier        = tier,
                alternatives= [label_to_sentence(lbl) for lbl, _ in gesture.alternatives],
                latency_ms  = latency,
            )

            log.info(f"🤟 {tr}")

            if self.on_translation:
                self.on_translation(tr)

            # Update context on high confidence
            if tier == ConfidenceTier.HIGH:
                self.context.add(tr.sentence)

            self.classifier.reset()
            time.sleep(0.05)

    # ── Feedback ──────────────────────────────────────────────────────────────

    def submit_correction(self, correct_sentence: str):
        dummy_emb = np.zeros(64, dtype=np.float32)
        self.corrections.store(dummy_emb, correct_sentence)
        log.info(f"Correction stored: {correct_sentence!r}")

    def set_topic(self, topic: str):
        self.topic.set(topic)
