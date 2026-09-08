"""
Pipeline Orchestrator
======================
Coordinates the full SignBridge inference pipeline:

  Camera → Landmarks → Encoder → Personalization → Context
  → Translation + Confidence → Output Decision

Runs the perception loop in a background thread.
Emits signals to the UI thread via Qt signals.
"""

import logging
import threading
import time
from typing import Optional, Callable

import numpy as np

from pipeline.camera import CameraCapture
from pipeline.landmark_extractor import LandmarkExtractor
from models.temporal_encoder import TemporalEncoder
from models.translation_decoder import TranslationDecoder
from models.confidence import ConfidenceEstimator, ConfidenceTier
from personalization.user_embedding import UserProfileManager
from personalization.correction_memory import CorrectionMemory
from context_engine.context_window import ContextWindow
from context_engine.topic_tag import TopicTag

log = logging.getLogger("signbridge.pipeline.orchestrator")

# Minimum frames in buffer before attempting translation
MIN_FRAMES_FOR_TRANSLATION = 32
# Minimum gap between translation attempts (seconds)
TRANSLATION_COOLDOWN = 1.5


class TranslationResult:
    """Output from one complete translation cycle."""

    def __init__(
        self,
        text: str,
        confidence: float,
        tier: ConfidenceTier,
        alternatives: list[str],
        latency_ms: float,
    ):
        self.text         = text
        self.confidence   = confidence
        self.tier         = tier
        self.alternatives = alternatives
        self.latency_ms   = latency_ms

    def __repr__(self) -> str:
        return (
            f"TranslationResult(text={self.text!r}, "
            f"conf={self.confidence:.2f}, tier={self.tier.name}, "
            f"latency={self.latency_ms:.0f}ms)"
        )


class SignBridgePipeline:
    """
    Central orchestrator for the SignBridge AI pipeline.

    Callbacks (called from background thread, marshal to UI thread as needed):
        on_translation:   Callable[[TranslationResult], None]
        on_landmarks:     Callable[[np.ndarray], None]  — for live preview
        on_fps_update:    Callable[[float], None]
    """

    def __init__(
        self,
        user_id: str = "default",
        on_translation: Optional[Callable[[TranslationResult], None]] = None,
        on_landmarks:   Optional[Callable[[np.ndarray], None]] = None,
        on_fps_update:  Optional[Callable[[float], None]] = None,
    ):
        self.user_id        = user_id
        self.on_translation = on_translation
        self.on_landmarks   = on_landmarks
        self.on_fps_update  = on_fps_update

        # ── Sub-components ────────────────────────────────────────────────────
        self.camera      = CameraCapture()
        self.extractor   = LandmarkExtractor()
        self.encoder     = TemporalEncoder()
        self.decoder     = TranslationDecoder()
        self.confidence  = ConfidenceEstimator()
        self.user_mgr    = UserProfileManager(user_id)
        self.corrections = CorrectionMemory(user_id)
        self.context     = ContextWindow()
        self.topic       = TopicTag()

        # ── State ─────────────────────────────────────────────────────────────
        self._running         = False
        self._pipeline_thread: Optional[threading.Thread] = None
        self._last_translation_time = 0.0
        self._landmark_buffer: list[np.ndarray] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Load all models and start the pipeline. Returns True on success."""
        log.info("Starting SignBridge pipeline...")

        ok = all([
            self.extractor.load(),
            self.encoder.load(),
            self.decoder.load(),
        ])

        if not ok:
            log.error("One or more models failed to load — check model directory")
            return False

        self.user_mgr.load()
        self.corrections.load()

        ok_cam = self.camera.start()
        if not ok_cam:
            log.error("Camera failed to open")
            return False

        self._running = True
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop, daemon=True
        )
        self._pipeline_thread.start()
        log.info("Pipeline running")
        return True

    def stop(self) -> None:
        """Gracefully stop the pipeline."""
        self._running = False
        self.camera.stop()
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=3.0)
        log.info("Pipeline stopped")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _pipeline_loop(self) -> None:
        """
        Background loop: reads buffered frames → extracts landmarks →
        accumulates sequence → triggers translation when ready.
        """
        while self._running:
            frames = self.camera.buffer.get_window()

            if len(frames) < MIN_FRAMES_FOR_TRANSLATION:
                time.sleep(0.05)
                continue

            # ── Per-frame landmark extraction ─────────────────────────────────
            landmark_sequence = []
            for frame in frames:
                result = self.extractor.extract(frame)
                vec    = result.to_flat_vector()
                landmark_sequence.append(vec)

                if self.on_landmarks:
                    self.on_landmarks(vec)

            # ── Report FPS ────────────────────────────────────────────────────
            if self.on_fps_update:
                self.on_fps_update(self.camera.fps)

            # ── Translation cooldown ──────────────────────────────────────────
            now = time.perf_counter()
            if now - self._last_translation_time < TRANSLATION_COOLDOWN:
                time.sleep(0.1)
                continue

            self._last_translation_time = now

            # ── Encode sequence ───────────────────────────────────────────────
            t0       = time.perf_counter()
            norm_seq = self.extractor.normalize_sequence(landmark_sequence)  # [64, 225]
            seq_emb  = self.encoder.encode(norm_seq)                          # [256]

            # ── Personalization fusion ────────────────────────────────────────
            user_emb = self.user_mgr.get_embedding()                          # [64]
            fused    = self.encoder.fuse_user(seq_emb, user_emb)              # [256]

            # ── Context conditioning ──────────────────────────────────────────
            ctx_emb  = self.context.get_embedding()
            ctx_fused = np.concatenate([fused, ctx_emb], axis=0) if ctx_emb is not None else fused

            # ── Translate ─────────────────────────────────────────────────────
            top_k_results = self.decoder.decode_topk(ctx_fused, k=3)  # [(text, prob), ...]

            # ── Correction memory override ────────────────────────────────────
            override = self.corrections.lookup(seq_emb, threshold=0.85)

            # ── Confidence estimation ─────────────────────────────────────────
            conf_score, tier = self.confidence.score(
                top_k_results,
                correction_override=override,
            )

            best_text = override if override else top_k_results[0][0]
            alts      = [t for t, _ in top_k_results[1:]]
            latency   = (time.perf_counter() - t0) * 1000  # ms

            result = TranslationResult(
                text        = best_text,
                confidence  = conf_score,
                tier        = tier,
                alternatives= alts,
                latency_ms  = latency,
            )

            log.info(f"Translation: {result}")

            if self.on_translation:
                self.on_translation(result)

            # ── Update context window ─────────────────────────────────────────
            if tier == ConfidenceTier.HIGH:
                self.context.add(best_text)

            time.sleep(0.05)

    # ── User feedback ─────────────────────────────────────────────────────────

    def submit_correction(self, seq_embedding: np.ndarray, correct_text: str) -> None:
        """Called when user taps 'Incorrect' and provides the right translation."""
        self.corrections.store(seq_embedding, correct_text)
        log.info(f"Correction stored: {correct_text!r}")

    def set_topic(self, topic: str) -> None:
        """Set session topic tag (Medical / General / Emergency / Work / Education)."""
        self.topic.set(topic)
        log.info(f"Topic set to: {topic}")
