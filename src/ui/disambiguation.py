"""
Disambiguation Dialog
======================
Shown when confidence is in the MED tier (0.50–0.74).
Presents top-2 interpretation options in large, accessible text.
User selects the correct one. Selection is stored as a correction if the
model's top-1 was wrong.

This is a core UX innovation: making AI uncertainty visible and actionable.
"""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

log = logging.getLogger("signbridge.ui.disambiguation")


class DisambiguationDialog(QDialog):
    """
    Dialog shown for MED-confidence translations.

    Example:
      "Did you mean:"
      [A] I have an allergy
      [B] I have a reaction
    """

    def __init__(
        self,
        top_text: str,
        alternatives: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self.selected_text: str | None = None
        self._top_text    = top_text
        self._alternatives = alternatives

        self.setWindowTitle("Clarification Needed")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setStyleSheet("""
            QDialog { background: #16213e; color: #e0e0e0; font-family: 'Segoe UI'; }
            QPushButton {
                background: #0f3460; border-radius: 10px; color: white;
                font-size: 20px; padding: 18px 24px; text-align: left;
            }
            QPushButton:hover { background: #1a4a80; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(18)

        # Header
        header = QLabel("⚠  Ambiguous sign detected — did you mean:")
        header.setFont(QFont("Segoe UI", 14))
        header.setStyleSheet("color: #ffdd57;")
        layout.addWidget(header)

        # Option A — top-1 prediction
        btn_a = QPushButton(f"  A)  {top_text}")
        btn_a.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        btn_a.clicked.connect(lambda: self._select(top_text, correct=True))
        layout.addWidget(btn_a)

        # Option B — alternative
        alt_text = alternatives[0] if alternatives else "(No alternative)"
        btn_b = QPushButton(f"  B)  {alt_text}")
        btn_b.setFont(QFont("Segoe UI", 20))
        btn_b.clicked.connect(lambda: self._select(alt_text, correct=False))
        layout.addWidget(btn_b)

        # Skip option
        skip_btn = QPushButton("  Skip")
        skip_btn.setFont(QFont("Segoe UI", 13))
        skip_btn.setStyleSheet("background: #333; color: #aaa; border-radius: 8px; padding: 10px;")
        skip_btn.clicked.connect(self.reject)
        layout.addWidget(skip_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _select(self, text: str, correct: bool) -> None:
        self.selected_text = text
        log.info(f"User selected: {text!r} (was_top1={correct})")
        self.accept()
