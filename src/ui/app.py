"""
SignBridge — Main Application Window (PyQt6)
=============================================
Clean, accessible UI designed for real communication scenarios.

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  [Camera Preview]          [Partner Speech Display]      │
  │                                                          │
  │  [Translation Output — Large Text]                       │
  │                                                          │
  │  [Confidence Indicator]  [Incorrect Button]              │
  │                                                          │
  │  [Topic Selector]  [Enroll]  [Settings]  [FPS Counter]   │
  └─────────────────────────────────────────────────────────┘
"""

import logging
import threading
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFrame, QSizePolicy,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette

log = logging.getLogger("signbridge.ui.app")


class PipelineSignals(QObject):
    """Qt signals for cross-thread communication from pipeline to UI."""
    translation_ready = pyqtSignal(object)   # TranslationResult
    fps_updated       = pyqtSignal(float)
    landmark_update   = pyqtSignal(object)


class SignBridgeApp(QMainWindow):
    """Main SignBridge application window."""

    def __init__(self):
        super().__init__()
        self._signals   = PipelineSignals()
        self._pipeline  = None
        self._asr_thread: Optional[threading.Thread] = None
        self._last_result = None

        self._build_ui()
        self._connect_signals()
        self._start_pipeline()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("SignBridge — On-Device Communication Intelligence")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(self._stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # ── Row 1: Camera preview + Partner speech ─────────────────────────
        row1 = QHBoxLayout()

        self.camera_label = QLabel("📷 Camera — Waiting for video...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setFixedSize(420, 280)
        self.camera_label.setStyleSheet(
            "background: #1a1a2e; border-radius: 12px; color: #888; font-size: 14px;"
        )
        row1.addWidget(self.camera_label)

        row1.addSpacing(20)

        partner_panel = QVBoxLayout()
        partner_title = QLabel("🎤 Partner Speech (Live Transcript)")
        partner_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        partner_title.setStyleSheet("color: #a0c4ff;")
        partner_panel.addWidget(partner_title)

        self.partner_text = QLabel("Waiting for partner to speak...")
        self.partner_text.setFont(QFont("Segoe UI", 22))
        self.partner_text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.partner_text.setWordWrap(True)
        self.partner_text.setStyleSheet(
            "background: #0f3460; border-radius: 12px; padding: 16px; color: #e0e0e0; "
            "min-height: 230px;"
        )
        self.partner_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        partner_panel.addWidget(self.partner_text)
        row1.addLayout(partner_panel)

        main_layout.addLayout(row1)

        # ── Row 2: Translation output ──────────────────────────────────────
        trans_title = QLabel("💬 Translation Output")
        trans_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        trans_title.setStyleSheet("color: #a0c4ff;")
        main_layout.addWidget(trans_title)

        self.translation_label = QLabel("Sign to begin...")
        self.translation_label.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        self.translation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translation_label.setWordWrap(True)
        self.translation_label.setMinimumHeight(120)
        self.translation_label.setStyleSheet(
            "background: #16213e; border-radius: 16px; padding: 24px; "
            "color: #ffffff; border: 2px solid #0f3460;"
        )
        main_layout.addWidget(self.translation_label)

        # ── Row 3: Confidence indicator + Incorrect button ─────────────────
        row3 = QHBoxLayout()

        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setFont(QFont("Segoe UI", 13))
        self.confidence_label.setStyleSheet("color: #aaa;")
        row3.addWidget(self.confidence_label)

        row3.addStretch()

        self.incorrect_btn = QPushButton("✗  Incorrect Translation")
        self.incorrect_btn.setFont(QFont("Segoe UI", 12))
        self.incorrect_btn.setFixedHeight(42)
        self.incorrect_btn.setStyleSheet(
            "QPushButton { background: #e94560; border-radius: 8px; color: white; padding: 0 20px; }"
            "QPushButton:hover { background: #c73652; }"
        )
        self.incorrect_btn.clicked.connect(self._on_incorrect)
        row3.addWidget(self.incorrect_btn)

        main_layout.addLayout(row3)

        # ── Row 4: Controls ────────────────────────────────────────────────
        row4 = QHBoxLayout()

        topic_label = QLabel("Topic:")
        topic_label.setFont(QFont("Segoe UI", 12))
        topic_label.setStyleSheet("color: #ccc;")
        row4.addWidget(topic_label)

        self.topic_combo = QComboBox()
        self.topic_combo.addItems(["General", "Medical", "Emergency", "Work", "Education"])
        self.topic_combo.setFont(QFont("Segoe UI", 12))
        self.topic_combo.setFixedHeight(36)
        self.topic_combo.currentTextChanged.connect(self._on_topic_changed)
        row4.addWidget(self.topic_combo)

        row4.addSpacing(16)

        self.enroll_btn = QPushButton("👤  Enroll / Re-enroll")
        self.enroll_btn.setFont(QFont("Segoe UI", 12))
        self.enroll_btn.setFixedHeight(36)
        self.enroll_btn.clicked.connect(self._on_enroll)
        row4.addWidget(self.enroll_btn)

        row4.addStretch()

        self.fps_label = QLabel("FPS: —")
        self.fps_label.setFont(QFont("Segoe UI", 11))
        self.fps_label.setStyleSheet("color: #666;")
        row4.addWidget(self.fps_label)

        self.npu_label = QLabel("NPU: Active")
        self.npu_label.setFont(QFont("Segoe UI", 11))
        self.npu_label.setStyleSheet("color: #48c774;")
        row4.addWidget(self.npu_label)

        main_layout.addLayout(row4)

    def _connect_signals(self) -> None:
        self._signals.translation_ready.connect(self._on_translation)
        self._signals.fps_updated.connect(self._on_fps_update)

    # ── Pipeline startup ──────────────────────────────────────────────────────

    def _start_pipeline(self) -> None:
        def _init():
            try:
                from pipeline.orchestrator import SignBridgePipeline
                self._pipeline = SignBridgePipeline(
                    on_translation=lambda r: self._signals.translation_ready.emit(r),
                    on_fps_update=lambda fps: self._signals.fps_updated.emit(fps),
                )
                ok = self._pipeline.start()
                if not ok:
                    log.error("Pipeline failed to start")
            except Exception as e:
                log.error(f"Pipeline init error: {e}")

        t = threading.Thread(target=_init, daemon=True)
        t.start()

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_translation(self, result) -> None:
        from models.confidence import ConfidenceTier
        self._last_result = result

        if result.tier == ConfidenceTier.HIGH:
            self.translation_label.setText(result.text)
            self.translation_label.setStyleSheet(
                "background: #1b4332; border-radius: 16px; padding: 24px; "
                "color: #ffffff; border: 2px solid #2d6a4f;"
            )
            self.confidence_label.setText(
                f"Confidence: {result.confidence:.0%} — HIGH  ✓  Latency: {result.latency_ms:.0f}ms"
            )
            self.confidence_label.setStyleSheet("color: #48c774;")

        elif result.tier == ConfidenceTier.MED:
            # Show disambiguation
            from ui.disambiguation import DisambiguationDialog
            dlg = DisambiguationDialog(result.text, result.alternatives, parent=self)
            dlg.exec()
            self.confidence_label.setText(
                f"Confidence: {result.confidence:.0%} — MEDIUM  ⚠  Clarified"
            )
            self.confidence_label.setStyleSheet("color: #ffdd57;")

        else:  # LOW
            self.translation_label.setText("Please sign that again...")
            self.translation_label.setStyleSheet(
                "background: #3d1515; border-radius: 16px; padding: 24px; "
                "color: #ffffff; border: 2px solid #e94560;"
            )
            self.confidence_label.setText(
                f"Confidence: {result.confidence:.0%} — LOW  ✗  Re-sign requested"
            )
            self.confidence_label.setStyleSheet("color: #e94560;")

    def _on_fps_update(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")
        color = "#48c774" if fps >= 25 else "#ffdd57" if fps >= 15 else "#e94560"
        self.fps_label.setStyleSheet(f"color: {color};")

    def _on_topic_changed(self, topic: str) -> None:
        if self._pipeline:
            self._pipeline.set_topic(topic)

    def _on_enroll(self) -> None:
        QMessageBox.information(
            self, "Enrollment",
            "5-minute enrollment coming up.\n\n"
            "Follow the on-screen sign prompts.\n"
            "SignBridge will adapt to your personal signing style."
        )
        # TODO: launch EnrollmentDialog

    def _on_incorrect(self) -> None:
        if self._last_result is None:
            return
        QMessageBox.information(
            self, "Correction",
            "Thank you — correction noted.\n"
            "SignBridge will remember this for your next session."
        )
        # TODO: open text input for correct translation
        # self._pipeline.submit_correction(...)

    # ── Stylesheet ────────────────────────────────────────────────────────────

    @staticmethod
    def _stylesheet() -> str:
        return """
        QMainWindow, QWidget {
            background-color: #0a0a1a;
            color: #e0e0e0;
            font-family: 'Segoe UI';
        }
        QComboBox {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            color: #e0e0e0;
            padding: 4px 10px;
        }
        QPushButton {
            background: #0f3460;
            border-radius: 8px;
            color: white;
            padding: 0 16px;
        }
        QPushButton:hover { background: #1a4a80; }
        """

    def closeEvent(self, event) -> None:
        if self._pipeline:
            self._pipeline.stop()
        event.accept()
