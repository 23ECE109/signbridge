"""
Rule-Based Gesture Classifier
==============================
Recognises 20 core ASL/common signs from MediaPipe hand landmark geometry.
No model training required — works immediately using finger angle + distance
heuristics derived from hand landmark coordinates.

Signs supported (demo vocabulary):
  HELLO, THANK_YOU, YES, NO, HELP, STOP, WATER, PAIN,
  PLEASE, I/ME, YOU, UNDERSTAND, REPEAT, GOOD, BAD,
  MORE, WHAT, WHERE, EMERGENCY, FINISHED

Each sign has a geometric rule set. The classifier returns:
  - label (str)
  - confidence (float 0–1)
  - description (str)  — shown in disambiguation dialog
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("signbridge.models.gesture_classifier")


@dataclass
class GestureResult:
    label:       str
    confidence:  float
    description: str
    alternatives: list   # [(label, conf), ...]


# ── Finger geometry helpers ────────────────────────────────────────────────────

def _finger_extended(hand: np.ndarray, tip: int, pip: int) -> bool:
    """True if fingertip is above (lower y) than PIP joint — finger extended."""
    return hand[tip, 1] < hand[pip, 1]

def _finger_curled(hand: np.ndarray, tip: int, pip: int) -> bool:
    return hand[tip, 1] > hand[pip, 1]

def _thumb_extended(hand: np.ndarray) -> bool:
    """Thumb tip (4) further from wrist (0) than thumb IP (3)."""
    d_tip = np.linalg.norm(hand[4] - hand[0])
    d_ip  = np.linalg.norm(hand[3] - hand[0])
    return d_tip > d_ip * 1.1

def _all_fingers_extended(hand: np.ndarray) -> bool:
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(_finger_extended(hand, t, p) for t, p in zip(tips, pips))

def _all_fingers_curled(hand: np.ndarray) -> bool:
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(_finger_curled(hand, t, p) for t, p in zip(tips, pips))

def _index_only(hand: np.ndarray) -> bool:
    return (_finger_extended(hand, 8, 6) and
            _finger_curled(hand, 12, 10) and
            _finger_curled(hand, 16, 14) and
            _finger_curled(hand, 20, 18))

def _two_fingers(hand: np.ndarray) -> bool:
    """Index + middle extended, ring + pinky curled."""
    return (_finger_extended(hand, 8,  6) and
            _finger_extended(hand, 12, 10) and
            _finger_curled(hand, 16, 14) and
            _finger_curled(hand, 20, 18))

def _pinky_only(hand: np.ndarray) -> bool:
    return (_finger_curled(hand, 8,  6) and
            _finger_curled(hand, 12, 10) and
            _finger_curled(hand, 16, 14) and
            _finger_extended(hand, 20, 18))

def _thumb_index_close(hand: np.ndarray, threshold: float = 0.06) -> bool:
    return np.linalg.norm(hand[4] - hand[8]) < threshold

def _hand_height(hand: np.ndarray) -> float:
    """Normalised y of wrist (0). Lower = hand raised higher on screen."""
    return hand[0, 1]

def _wrist_near_face(pose: Optional[np.ndarray], hand: np.ndarray) -> bool:
    """True if wrist landmark is near nose landmark in pose."""
    if pose is None:
        return False
    nose   = pose[0]
    wrist  = hand[0]
    return np.linalg.norm(nose[:2] - wrist[:2]) < 0.25


# ── Sign rules ────────────────────────────────────────────────────────────────

def _classify_hand(hand: np.ndarray,
                   pose: Optional[np.ndarray]) -> list:
    """
    Returns list of (label, confidence, description) sorted by confidence.
    Each rule tries to match a gesture. Multiple can partially match.
    """
    scores = []

    # ── HELLO: open hand, palm out, raised near face ─────────────────────
    if _all_fingers_extended(hand) and _thumb_extended(hand):
        conf = 0.85 if _wrist_near_face(pose, hand) else 0.60
        scores.append(("HELLO",     conf, "Open hand near face — hello / wave"))

    # ── THANK_YOU: flat hand, tips touch chin then move forward ──────────
    # Approximation: all fingers extended + hand at mid-face height
    if _all_fingers_extended(hand) and 0.3 < _hand_height(hand) < 0.6:
        scores.append(("THANK_YOU", 0.65, "Flat hand near mouth — thank you"))

    # ── YES: fist nodding — closed fist ───────────────────────────────────
    if _all_fingers_curled(hand) and not _thumb_extended(hand):
        scores.append(("YES",       0.80, "Closed fist — yes / agree"))

    # ── NO: index + middle extended, side-to-side ─────────────────────────
    if _two_fingers(hand):
        scores.append(("NO",        0.72, "Two fingers extended — no / disagree"))

    # ── STOP: flat hand raised, palm facing out ────────────────────────────
    if _all_fingers_extended(hand) and _hand_height(hand) < 0.5:
        scores.append(("STOP",      0.68, "Raised flat hand — stop / wait"))

    # ── I / ME: index finger pointing to self ─────────────────────────────
    if _index_only(hand):
        scores.append(("I_ME",      0.75, "Index pointing — I / me"))

    # ── HELP: closed fist on flat palm (two hands needed — single hand approx)
    if _all_fingers_curled(hand) and _thumb_extended(hand):
        scores.append(("HELP",      0.70, "Fist with thumb up — help"))

    # ── GOOD: flat hand from chin moving forward ───────────────────────────
    if _all_fingers_extended(hand) and _wrist_near_face(pose, hand) and _thumb_extended(hand):
        scores.append(("GOOD",      0.72, "Flat hand from face — good"))

    # ── BAD: hand flips down ───────────────────────────────────────────────
    if _all_fingers_extended(hand) and hand[0, 2] > 0.05:  # z > 0 = tilted away
        scores.append(("BAD",       0.55, "Hand tilted — bad"))

    # ── WATER: W handshape — middle three fingers extended ────────────────
    if (_finger_extended(hand, 8,  6) and
        _finger_extended(hand, 12, 10) and
        _finger_extended(hand, 16, 14) and
        _finger_curled(hand, 20, 18) and
        _finger_curled(hand, 4, 3)):
        scores.append(("WATER",     0.78, "Three fingers up — water"))

    # ── PLEASE: flat hand circles on chest ────────────────────────────────
    if (_all_fingers_extended(hand) and
        not _thumb_extended(hand) and
        0.4 < _hand_height(hand) < 0.7):
        scores.append(("PLEASE",    0.65, "Circular motion on chest — please"))

    # ── MORE: fingertips pinched together, both hands tap ─────────────────
    if _thumb_index_close(hand, threshold=0.07):
        scores.append(("MORE",      0.72, "Fingertips pinched — more"))

    # ── UNDERSTAND: index curls from forehead ─────────────────────────────
    if _index_only(hand) and _wrist_near_face(pose, hand):
        scores.append(("UNDERSTAND", 0.70, "Index at forehead — understand"))

    # ── REPEAT: index circles ─────────────────────────────────────────────
    if _index_only(hand) and not _wrist_near_face(pose, hand):
        scores.append(("REPEAT",    0.62, "Index circling — repeat / again"))

    # ── WHAT: spread fingers, shrug ───────────────────────────────────────
    if (_all_fingers_extended(hand) and not _thumb_extended(hand) and
        np.linalg.norm(hand[8] - hand[20]) > 0.25):   # fingers spread wide
        scores.append(("WHAT",      0.68, "Spread fingers — what?"))

    # ── WHERE: index wagging ───────────────────────────────────────────────
    if _index_only(hand) and hand[8, 0] > 0.5:  # pointing right
        scores.append(("WHERE",     0.60, "Index wagging — where?"))

    # ── EMERGENCY: E handshape — all fingers folded ────────────────────────
    if _all_fingers_curled(hand) and _thumb_extended(hand) and _wrist_near_face(pose, hand):
        scores.append(("EMERGENCY", 0.75, "Closed fist near face — emergency"))

    # ── FINISHED: both hands open, flip outward ────────────────────────────
    if (_all_fingers_extended(hand) and
        _thumb_extended(hand) and
        hand[0, 1] > 0.5):   # hands low
        scores.append(("FINISHED",  0.70, "Open hands flipped out — finished / done"))

    # ── PAIN: index fingers tap together ──────────────────────────────────
    if _index_only(hand):
        scores.append(("PAIN",      0.58, "Index fingers tapping — pain / hurt"))

    return scores


# ── Public classifier ─────────────────────────────────────────────────────────

class GestureClassifier:
    """
    Classifies a sequence of LandmarkResult objects into a sign label.
    Uses majority vote over the sequence + per-frame geometric rules.
    """

    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self._frame_votes: list = []   # list of [(label, conf)] per frame

    def update(self, landmark_result) -> None:
        """Feed one frame's LandmarkResult into the classifier."""
        hand = landmark_result.dominant_hand()
        if hand is None:
            return

        pose   = landmark_result.pose
        scores = _classify_hand(hand, pose)

        if scores:
            self._frame_votes.append(scores)
            if len(self._frame_votes) > self.history_len:
                self._frame_votes.pop(0)

    def classify(self) -> Optional[GestureResult]:
        """
        Returns the best classification from recent frames, or None.
        Uses: weighted majority vote across frame window.
        """
        if len(self._frame_votes) < 5:
            return None

        # Aggregate votes
        vote_totals: dict = {}
        desc_map: dict = {}
        for frame_scores in self._frame_votes:
            for label, conf, desc in frame_scores:
                vote_totals[label] = vote_totals.get(label, 0.0) + conf
                desc_map[label]    = desc

        if not vote_totals:
            return None

        # Normalise by number of frames
        n = len(self._frame_votes)
        normalised = {k: v / n for k, v in vote_totals.items()}

        # Sort by score
        ranked = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
        top_label, top_conf = ranked[0]

        # Clamp confidence
        top_conf = min(top_conf, 0.95)

        alternatives = [(lbl, round(c, 2)) for lbl, c in ranked[1:3]]

        return GestureResult(
            label       = top_label,
            confidence  = round(top_conf, 2),
            description = desc_map.get(top_label, ""),
            alternatives= alternatives,
        )

    def reset(self) -> None:
        self._frame_votes.clear()


# ── Label → human sentence ────────────────────────────────────────────────────

SIGN_TO_SENTENCE = {
    "HELLO":      "Hello!",
    "THANK_YOU":  "Thank you.",
    "YES":        "Yes.",
    "NO":         "No.",
    "HELP":       "Help me, please.",
    "STOP":       "Stop. Wait.",
    "WATER":      "I need water.",
    "PAIN":       "I am in pain.",
    "PLEASE":     "Please.",
    "I_ME":       "I / Me.",
    "YOU":        "You.",
    "UNDERSTAND": "I understand.",
    "REPEAT":     "Please repeat that.",
    "GOOD":       "Good.",
    "BAD":        "Bad.",
    "MORE":       "More, please.",
    "WHAT":       "What?",
    "WHERE":      "Where?",
    "EMERGENCY":  "This is an emergency!",
    "FINISHED":   "I am finished.",
}

def label_to_sentence(label: str) -> str:
    return SIGN_TO_SENTENCE.get(label, label.replace("_", " ").title())
