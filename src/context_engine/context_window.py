"""
Context Window — Sliding Conversation History
==============================================
Maintains the last N translated utterances as a context vector.
Used to bias the translation decoder toward contextually likely interpretations.

Example: In a medical conversation, after "I have an allergy" is in context,
the next ambiguous sign is more likely to be "medicine" than "drug".
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger("signbridge.context_engine.context_window")

WINDOW_SIZE  = 5    # Number of recent utterances to retain
CONTEXT_DIM  = 32   # Dimension of context embedding appended to sequence embedding


class ContextWindow:
    """
    Maintains a sliding window of recent translated utterances
    and produces a fixed-size context embedding.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._history: deque[str] = deque(maxlen=window_size)
        self._vocab: dict[str, int] = {}
        self._loaded = False

    def add(self, utterance: str) -> None:
        """Add a confirmed (HIGH confidence) translation to the history."""
        self._history.append(utterance.strip().lower())
        log.debug(f"Context updated: {list(self._history)}")

    def get_history(self) -> list[str]:
        """Return current context as a list of strings."""
        return list(self._history)

    def get_embedding(self) -> Optional[np.ndarray]:
        """
        Produce a CONTEXT_DIM embedding from the current history.

        Method: Bag-of-words TF-IDF over the sliding window,
        projected to CONTEXT_DIM via a pre-computed projection matrix.

        Returns None if history is empty.
        """
        if not self._history:
            return None

        # Simple bag-of-words over word tokens in history
        words: list[str] = []
        for utterance in self._history:
            words.extend(utterance.split())

        if not words:
            return None

        # Build a simple frequency vector over known words
        # In production: use pre-trained word embeddings (GloVe or similar)
        word_counts: dict[str, int] = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        # Create a dense vector (hashing trick, simple implementation)
        vec = np.zeros(CONTEXT_DIM, dtype=np.float32)
        for word, count in word_counts.items():
            idx = hash(word) % CONTEXT_DIM
            vec[idx] += count

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec  # [32]

    def clear(self) -> None:
        """Clear conversation history (e.g., on session end)."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
