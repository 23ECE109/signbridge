"""
Confidence Estimator — 3-Tier System
======================================
Computes a confidence score from decoder output and correction memory.

Formula:
  score = w1 × mean_token_prob + w2 × (1 - entropy_norm) + w3 × knn_match_score

Tiers:
  HIGH  (≥ 0.75) → auto-translate and speak
  MED   (0.50–0.74) → show disambiguation dialog
  LOW   (< 0.50) → request re-sign

This is a core differentiator: no deployed sign-language system
exposes uncertainty as a user interaction loop.
"""

import logging
from enum import Enum

import numpy as np

log = logging.getLogger("signbridge.models.confidence")

# ── Weights ───────────────────────────────────────────────────────────────────
W_TOKEN_PROB  = 0.50
W_ENTROPY     = 0.30
W_KNN         = 0.20

# ── Tier thresholds ───────────────────────────────────────────────────────────
THRESHOLD_HIGH = 0.75
THRESHOLD_MED  = 0.50


class ConfidenceTier(Enum):
    HIGH = "high"    # Auto-translate and speak
    MED  = "med"     # Disambiguation dialog
    LOW  = "low"     # Request re-sign


class ConfidenceEstimator:
    """
    Estimates translation confidence and returns the appropriate tier.
    """

    def score(
        self,
        top_k_results: list[tuple[str, float]],
        correction_override: str | None = None,
    ) -> tuple[float, ConfidenceTier]:
        """
        Compute confidence score and tier.

        Args:
            top_k_results: list of (text, probability) from decoder, sorted desc
            correction_override: text from correction memory if similarity > threshold

        Returns:
            (score: float in [0, 1], tier: ConfidenceTier)
        """
        if not top_k_results:
            return 0.0, ConfidenceTier.LOW

        probs = np.array([p for _, p in top_k_results], dtype=np.float32)
        probs = probs / (probs.sum() + 1e-9)  # Re-normalize

        # ── Component 1: mean token probability ──────────────────────────────
        mean_prob = float(probs[0])  # Top-1 probability

        # ── Component 2: entropy (lower entropy = higher confidence) ─────────
        entropy      = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy  = np.log(len(probs) + 1e-9)
        entropy_norm = float(entropy / (max_entropy + 1e-9))  # [0, 1]

        # ── Component 3: correction memory match score ────────────────────────
        knn_score = 1.0 if correction_override is not None else 0.0

        # ── Weighted combination ──────────────────────────────────────────────
        score = (
            W_TOKEN_PROB * mean_prob
            + W_ENTROPY  * (1.0 - entropy_norm)
            + W_KNN      * knn_score
        )
        score = float(np.clip(score, 0.0, 1.0))

        # ── If correction memory override exists, boost to at least MED ──────
        if correction_override is not None:
            score = max(score, THRESHOLD_MED + 0.05)

        # ── Tier decision ─────────────────────────────────────────────────────
        if score >= THRESHOLD_HIGH:
            tier = ConfidenceTier.HIGH
        elif score >= THRESHOLD_MED:
            tier = ConfidenceTier.MED
        else:
            tier = ConfidenceTier.LOW

        log.debug(
            f"Confidence: score={score:.3f} tier={tier.name} "
            f"(prob={mean_prob:.3f}, entropy_norm={entropy_norm:.3f}, knn={knn_score:.1f})"
        )
        return score, tier

    @staticmethod
    def describe_tier(tier: ConfidenceTier) -> str:
        """Human-readable description of what the system will do."""
        return {
            ConfidenceTier.HIGH: "Translating automatically",
            ConfidenceTier.MED:  "Checking with you — ambiguous sign detected",
            ConfidenceTier.LOW:  "Please sign that again — confidence too low",
        }[tier]
