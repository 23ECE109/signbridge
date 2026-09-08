"""
SignBridge — Main Application Window
======================================
Real-time demo UI showing:
  LEFT  — Live camera feed with MediaPipe landmark overlay
  RIGHT — Partner speech transcript (Whisper)
  BOTTOM — Translation output + confidence indicator
  FOOTER — Topic selector, enroll, FPS, NPU/CPU badge
"""

import logging
import threading
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFrame, QSizePolicy,
    QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor

log = logging.getLogger("signbridge.ui.app")


class Signals(QObject):
    translation_ready = pyqtSignal(object)
    frame_ready       = pyqtSignal(object, object)   # (frame_rgb, landmark_result)
    fps_updated       = pyqtSignal(float)
    asr_text          = pyqtSignal(str)


class SignBridgeApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self._signals   = Signals()
        self._pipeline  = None
        self._asr       = None
        self._tts       = None
        self._last_result = None

        self._build_ui()
        self._connect_signals()

        # Start pipeline in background so UI appears instantly
        QTimer.singleShot(500, self._start_pipeline)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("SignBridge — Real-Time Sign Language Communication")
        self.setMinimumSize(1200, 720)
        self.setStyleSheet(self._css())

        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(12)

        # ── Title bar ─────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        logo = QLabel("🤝  SignBridge")
        logo.setFont(QFont("Segoe UI", 18, QFont.Weight.Black))
        logo.setStyleSheet("color: #6c63ff;")
        title_row.addWidget(logo)

        self.status_label = QLabel("⏳  Starting pipeline...")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #888;")
        title_row.addStretch()
        title_row.addWidget(self.status_label)
        vbox.addLayout(title_row)

        # ── Main row: camera + partner ────────────────────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(14)

        # Camera panel
        cam_panel = QVBoxLayout()
        cam_title = QLabel("📷  Live Camera + Landmarks")
        cam_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        cam_title.setStyleSheet("color: #a0c4ff;")
        cam_panel.addWidget(cam_title)

        self.camera_view = QLabel()
        self.camera_view.setFixedSize(560, 380)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setStyleSheet(
            "background:#0d0d22; border-radius:12px; border:1px solid #1e1e40;"
        )
        self.camera_view.setText("Camera starting...")
        self.camera_view.setFont(QFont("Segoe UI", 13))
        cam_panel.addWidget(self.camera_view)
        main_row.addLayout(cam_panel)

        # Partner speech panel
        partner_panel = QVBoxLayout()
        p_title = QLabel("🎤  Partner Speech  (Whisper ASR)")
        p_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        p_title.setStyleSheet("color: #a0c4ff;")
        partner_panel.addWidget(p_title)

        self.partner_label = QLabel("Waiting for partner to speak...")
        self.partner_label.setFont(QFont("Segoe UI", 20))
        self.partner_label.setWordWrap(True)
        self.partner_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.partner_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.partner_label.setStyleSheet(
            "background:#0f3460; border-radius:12px; padding:18px; color:#e0e0ff;"
            "border:1px solid #1e1e50;"
        )
        partner_panel.addWidget(self.partner_label)
        main_row.addLayout(partner_panel)

        vbox.addLayout(main_row)

        # ── Translation output ─────────────────────────────────────────────
        trans_title = QLabel("💬  Translation Output")
        trans_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        trans_title.setStyleSheet("color: #a0c4ff; margin-top:4px;")
        vbox.addWidget(trans_title)

        self.translation_label = QLabel("Sign to begin — face the camera and sign clearly")
        self.translation_label.setFont(QFont("Segoe UI", 34, QFont.Weight.Black))
        self.translation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translation_label.setWordWrap(True)
        self.translation_label.setMinimumHeight(110)
        self.translation_label.setStyleSheet(
            "background:#16213e; border-radius:16px; padding:22px;"
            "color:#ffffff; border:2px solid #0f3460;"
        )
        vbox.addWidget(self.translation_label)

        # ── Confidence row ─────────────────────────────────────────────────
        conf_row = QHBoxLayout()
        self.conf_label = QLabel("Confidence: —")
        self.conf_label.setFont(QFont("Segoe UI", 12))
        self.conf_label.setStyleSheet("color:#888;")
        conf_row.addWidget(self.conf_label)

        self.latency_label = QLabel("Latency: —")
        self.latency_label.setFont(QFont("Segoe UI", 11))
        self.latency_label.setStyleSheet("color:#555;")
        conf_row.addWidget(self.latency_label)

        conf_row.addStretch()

        self.incorrect_btn = QPushButton("✗  Incorrect")
        self.incorrect_btn.setFont(QFont("Segoe UI", 11))
        self.incorrect_btn.setFixedHeight(38)
        self.incorrect_btn.setStyleSheet(
            "QPushButton{background:#e94560;border-radius:8px;color:white;padding:0 18px;}"
            "QPushButton:hover{background:#c73652;}"
        )
        self.incorrect_btn.clicked.connect(self._on_incorrect)
        conf_row.addWidget(self.incorrect_btn)

        vbox.addLayout(conf_row)

        # ── Controls row ───────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()

        ctrl_row.addWidget(QLabel("Topic:"))
        self.topic_combo = QComboBox()
        self.topic_combo.addItems(["General", "Medical", "Emergency", "Work", "Education"])
        self.topic_combo.setFixedHeight(34)
        self.topic_combo.currentTextChanged.connect(self._on_topic)
        ctrl_row.addWidget(self.topic_combo)

        ctrl_row.addSpacing(12)

        speak_btn = QPushButton("🔊  Speak Last")
        speak_btn.setFixedHeight(34)
        speak_btn.clicked.connect(self._speak_last)
        ctrl_row.addWidget(speak_btn)

        ctrl_row.addStretch()

        self.fps_label = QLabel("FPS: —")
        self.fps_label.setFont(QFont("Segoe UI", 11))
        self.fps_label.setStyleSheet("color:#555;")
        ctrl_row.addWidget(self.fps_label)

        self.compute_label = QLabel("⚡  CPU")
        self.compute_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.compute_label.setStyleSheet("color:#48c774;")
        ctrl_row.addWidget(self.compute_label)

        vbox.addLayout(ctrl_row)

        # ── Signs cheat sheet ──────────────────────────────────────────────
        cheat = QLabel(
            "Demo signs: HELLO  YES  NO  HELP  STOP  WATER  PLEASE  THANK_YOU  "
            "GOOD  MORE  WHAT  EMERGENCY  FINISHED  I/ME  UNDERSTAND  REPEAT  PAIN"
        )
        cheat.setFont(QFont("Segoe UI", 9))
        cheat.setStyleSheet("color:#444; margin-top:2px;")
        cheat.setWordWrap(True)
        vbox.addWidget(cheat)

    def _connect_signals(self):
        self._signals.translation_ready.connect(self._on_translation)
        self._signals.frame_ready.connect(self._on_frame)
        self._signals.fps_updated.connect(self._on_fps)
        self._signals.asr_text.connect(self._on_asr)

    # ── Pipeline startup ──────────────────────────────────────────────────────

    def _start_pipeline(self):
        def _init():
            from pipeline.orchestrator import SignBridgePipeline
            from models.tts import TTSEngine
            from models.asr import WhisperASR

            # TTS
            self._tts = TTSEngine()
            self._tts.load()

            # Pipeline
            self._pipeline = SignBridgePipeline(
                on_translation = lambda r: self._signals.translation_ready.emit(r),
                on_frame       = lambda f, l: self._signals.frame_ready.emit(f, l),
                on_fps_update  = lambda fps: self._signals.fps_updated.emit(fps),
            )
            ok = self._pipeline.start()

            if ok:
                self._signals.fps_updated.emit(0)
                # ASR — load in background (downloads model on first run)
                self._asr = WhisperASR(
                    model_size    = "base",
                    on_transcript = lambda t: self._signals.asr_text.emit(t),
                )
                asr_ok = self._asr.load()
                if asr_ok:
                    self._asr.start()
                    self._signals.asr_text.emit("🎤 Whisper ASR ready — partner can speak now")
                else:
                    self._signals.asr_text.emit("⚠ Whisper not available (pip install openai-whisper)")
            else:
                self._signals.asr_text.emit("❌ Pipeline failed — check camera + mediapipe")

        t = threading.Thread(target=_init, daemon=True, name="sb-init")
        t.start()

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_translation(self, result):
        from models.confidence import ConfidenceTier
        self._last_result = result

        if result.tier == ConfidenceTier.HIGH:
            self.translation_label.setText(result.sentence)
            self.translation_label.setStyleSheet(
                "background:#1b4332;border-radius:16px;padding:22px;"
                "color:#ffffff;border:2px solid #2d6a4f;"
            )
            self.conf_label.setText(
                f"Confidence: {result.confidence:.0%}  ✓  HIGH  |  Sign: {result.text}"
            )
            self.conf_label.setStyleSheet("color:#48c774;")
            self.status_label.setText("✅  Translating")
            self.status_label.setStyleSheet("color:#48c774;")
            # Speak it
            if self._tts:
                self._tts.speak(result.sentence, blocking=False)

        elif result.tier == ConfidenceTier.MED:
            from ui.disambiguation import DisambiguationDialog
            dlg = DisambiguationDialog(result.sentence, result.alternatives, parent=self)
            if dlg.exec() and dlg.selected_text:
                self.translation_label.setText(dlg.selected_text)
                self.translation_label.setStyleSheet(
                    "background:#1a3a2a;border-radius:16px;padding:22px;"
                    "color:#fff;border:2px solid #2d6a4f;"
                )
                if self._tts:
                    self._tts.speak(dlg.selected_text, blocking=False)
                # If user picked something different, store correction
                if dlg.selected_text != result.sentence and self._pipeline:
                    self._pipeline.submit_correction(dlg.selected_text)
            self.conf_label.setText(f"Confidence: {result.confidence:.0%}  ⚠  MED — clarified")
            self.conf_label.setStyleSheet("color:#ffdd57;")

        else:  # LOW
            self.translation_label.setText("🔄  Please sign that again clearly...")
            self.translation_label.setStyleSheet(
                "background:#3d1515;border-radius:16px;padding:22px;"
                "color:#fff;border:2px solid #e94560;"
            )
            self.conf_label.setText(f"Confidence: {result.confidence:.0%}  ✗  LOW — re-sign")
            self.conf_label.setStyleSheet("color:#e94560;")

        self.latency_label.setText(f"Latency: {result.latency_ms:.0f} ms")

    def _on_frame(self, frame_rgb, landmark_result):
        """Render camera frame with landmark overlay onto camera_view QLabel."""
        try:
            import cv2
            frame = frame_rgb.copy()

            # Draw hand landmarks
            if landmark_result.left_hand is not None:
                self._draw_hand(frame, landmark_result.left_hand, (100, 220, 255))
            if landmark_result.right_hand is not None:
                self._draw_hand(frame, landmark_result.right_hand, (100, 255, 180))

            # Resize to fit panel
            h_panel, w_panel = 380, 560
            frame_resized = cv2.resize(frame, (w_panel, h_panel))

            # Convert to QPixmap
            img   = QImage(frame_resized.data, w_panel, h_panel,
                           w_panel * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.camera_view.setPixmap(pixmap)
        except Exception as e:
            pass  # Don't crash on frame render errors

    def _draw_hand(self, frame: np.ndarray, hand: np.ndarray, color: tuple):
        """Draw 21 hand landmarks as dots on the frame."""
        import cv2
        h, w = frame.shape[:2]
        # MediaPipe connections (simplified subset)
        connections = [
            (0,1),(1,2),(2,3),(3,4),       # thumb
            (0,5),(5,6),(6,7),(7,8),       # index
            (5,9),(9,10),(10,11),(11,12),  # middle
            (9,13),(13,14),(14,15),(15,16),# ring
            (13,17),(17,18),(18,19),(19,20),# pinky
            (0,17),
        ]
        pts = [(int(hand[i,0]*w), int(hand[i,1]*h)) for i in range(21)]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

    def _on_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        col = "#48c774" if fps >= 20 else "#ffdd57" if fps >= 10 else "#e94560"
        self.fps_label.setStyleSheet(f"color:{col};")
        if fps > 0 and self.status_label.text() == "⏳  Starting pipeline...":
            self.status_label.setText("✅  Pipeline running")
            self.status_label.setStyleSheet("color:#48c774;")

    def _on_asr(self, text: str):
        self.partner_label.setText(text)

    def _on_topic(self, topic: str):
        if self._pipeline:
            self._pipeline.set_topic(topic)

    def _speak_last(self):
        if self._last_result and self._tts:
            self._tts.speak(self._last_result.sentence, blocking=False)
        elif self._tts:
            self._tts.speak("SignBridge is ready.", blocking=False)

    def _on_incorrect(self):
        if self._last_result is None:
            return
        text, ok = QInputDialog.getText(
            self, "Correct Translation",
            f"Current: \"{self._last_result.sentence}\"\nEnter correct translation:"
        )
        if ok and text.strip():
            if self._pipeline:
                self._pipeline.submit_correction(text.strip())
            QMessageBox.information(self, "Saved", f"Got it — I'll remember:\n\"{text.strip()}\"")

    # ── Stylesheet ────────────────────────────────────────────────────────────

    @staticmethod
    def _css() -> str:
        return """
        QMainWindow,QWidget{background:#060612;color:#e0e0f0;font-family:'Segoe UI';}
        QLabel{color:#e0e0f0;}
        QComboBox{background:#1a1a2e;border:1px solid #1e1e40;border-radius:6px;
                  color:#e0e0f0;padding:4px 10px;}
        QPushButton{background:#0f3460;border-radius:8px;color:white;padding:0 16px;}
        QPushButton:hover{background:#1a4a80;}
        QInputDialog{background:#0d0d22;}
        """

    def closeEvent(self, event):
        if self._asr:
            self._asr.stop()
        if self._pipeline:
            self._pipeline.stop()
        event.accept()
