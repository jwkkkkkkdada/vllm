from __future__ import annotations

import secrets
from typing import Protocol


class KeyProvider(Protocol):
    """Supplies symmetric encryption keys to the disk store."""

    def get_key(self) -> bytes: ...


class EphemeralKeyProvider:
    """Keeps a randomly generated AES-256 key in process memory."""

    _KEY_SIZE_BYTES = 32

    def __init__(self) -> None:
        self._key = secrets.token_bytes(self._KEY_SIZE_BYTES)

    def get_key(self) -> bytes:
        return self._key


class StaticKeyProvider:
    """Supplies a caller-provided key, primarily for dependency injection."""

    def __init__(self, key: bytes) -> None:
        self._key = bytes(key)

    def get_key(self) -> bytes:
        return self._key
