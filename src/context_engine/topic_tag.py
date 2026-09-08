"""
Topic Tag — Session-Level Vocabulary Bias
==========================================
The user selects a conversation topic at session start.
The topic tag biases the decoder's token probability distribution
toward domain-appropriate vocabulary.

Topics:
  - GENERAL    (default)
  - MEDICAL    (medicine, doctor, pain, allergy → up-weighted)
  - EMERGENCY  (help, fire, police → maximally up-weighted)
  - WORK       (meeting, schedule, deadline → up-weighted)
  - EDUCATION  (class, teacher, homework → up-weighted)
"""

import logging
from enum import Enum

import numpy as np

log = logging.getLogger("signbridge.context_engine.topic_tag")

TOPIC_EMB_DIM = 16  # Embedding dimension for topic conditioning


class Topic(Enum):
    GENERAL   = "general"
    MEDICAL   = "medical"
    EMERGENCY = "emergency"
    WORK      = "work"
    EDUCATION = "education"


# Learned topic embeddings (placeholder — trained as part of decoder)
# These would be loaded from the model file in production
_TOPIC_EMBEDDINGS: dict[Topic, np.ndarray] = {
    Topic.GENERAL:   np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    Topic.MEDICAL:   np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    Topic.EMERGENCY: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    Topic.WORK:      np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    Topic.EDUCATION: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}


class TopicTag:
    """Manages the active session topic and provides its embedding."""

    def __init__(self):
        self._topic = Topic.GENERAL

    def set(self, topic_str: str) -> None:
        """Set topic from string name (case-insensitive)."""
        try:
            self._topic = Topic(topic_str.lower())
            log.info(f"Topic set: {self._topic.name}")
        except ValueError:
            log.warning(f"Unknown topic '{topic_str}' — using GENERAL")
            self._topic = Topic.GENERAL

    def get(self) -> Topic:
        return self._topic

    def get_embedding(self) -> np.ndarray:
        """Return the topic embedding [16] for decoder conditioning."""
        return _TOPIC_EMBEDDINGS[self._topic].copy()

    @staticmethod
    def available_topics() -> list[str]:
        return [t.value for t in Topic]
