"""
User Profile Manager
=====================
Loads, saves, and serves user embeddings.
All data stored locally, AES-256 encrypted via Python cryptography library.
Key derived from Windows DPAPI (per-user, per-machine).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("signbridge.personalization.user_embedding")

PROFILES_DIR = Path(__file__).parent.parent.parent / "data" / "profiles"
USER_DIM     = 64


class UserProfileManager:
    """
    Manages a single user's profile embedding.
    """

    def __init__(self, user_id: str):
        self.user_id   = user_id
        self._embedding: Optional[np.ndarray] = None
        self._profile_path = PROFILES_DIR / f"{user_id}.enc"
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> bool:
        """Load encrypted user profile from disk. Returns True if found."""
        if not self._profile_path.exists():
            log.info(f"No profile found for user '{self.user_id}' — using default embedding")
            self._embedding = np.zeros(USER_DIM, dtype=np.float32)
            return False

        try:
            data = self._decrypt(self._profile_path.read_bytes())
            profile = json.loads(data.decode("utf-8"))
            self._embedding = np.array(profile["embedding"], dtype=np.float32)
            log.info(f"Profile loaded for user '{self.user_id}'")
            return True
        except Exception as e:
            log.error(f"Profile load failed: {e}")
            self._embedding = np.zeros(USER_DIM, dtype=np.float32)
            return False

    def save(self, embedding: np.ndarray) -> bool:
        """Save user embedding, encrypted, to disk."""
        try:
            self._embedding = embedding.astype(np.float32)
            profile = {"user_id": self.user_id, "embedding": embedding.tolist()}
            raw     = json.dumps(profile).encode("utf-8")
            self._profile_path.write_bytes(self._encrypt(raw))
            log.info(f"Profile saved for user '{self.user_id}'")
            return True
        except Exception as e:
            log.error(f"Profile save failed: {e}")
            return False

    def get_embedding(self) -> np.ndarray:
        """Return the current user embedding [64]. Zeros if no profile."""
        if self._embedding is None:
            return np.zeros(USER_DIM, dtype=np.float32)
        return self._embedding.copy()

    def delete(self) -> None:
        """Permanently delete user profile (GDPR right-to-erasure)."""
        if self._profile_path.exists():
            self._profile_path.unlink()
            self._embedding = None
            log.info(f"Profile deleted for user '{self.user_id}'")

    # ── Encryption helpers ────────────────────────────────────────────────────

    def _get_key(self) -> bytes:
        """
        Derive encryption key from Windows DPAPI (per-user, per-machine).
        Falls back to a constant test key in development.
        NEVER use the fallback key in production.
        """
        try:
            import win32crypt  # pywin32 — optional dependency
            seed   = f"signbridge-{self.user_id}".encode()
            key_bytes = win32crypt.CryptProtectData(seed, None, None, None, None, 0)
            return key_bytes[:32]
        except ImportError:
            # Development fallback — NOT secure for production
            import hashlib
            return hashlib.sha256(f"dev-key-{self.user_id}".encode()).digest()

    def _encrypt(self, data: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import os as _os
        key   = self._get_key()
        nonce = _os.urandom(12)
        ct    = AESGCM(key).encrypt(nonce, data, None)
        return nonce + ct  # Prepend nonce to ciphertext

    def _decrypt(self, data: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        key   = self._get_key()
        nonce = data[:12]
        ct    = data[12:]
        return AESGCM(key).decrypt(nonce, ct, None)
