"""Encryption helpers for NiceGUI chat persistence."""

from __future__ import annotations

import base64
import json
import secrets
from contextlib import suppress
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ENVELOPE_VERSION = 1
_ALGORITHM = "AES-256-GCM"


class NiceGUICryptoError(RuntimeError):
    """Raised when NiceGUI encrypted persistence cannot decrypt data."""


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


def ensure_text_key_file(path: Path, *, token_bytes: int = 32) -> str:
    """Return a persistent text key, creating it with restrictive permissions."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = secrets.token_urlsafe(token_bytes)
    path.write_text(value, encoding="utf-8")
    with suppress(PermissionError):
        path.chmod(0o600)
    return value


def ensure_binary_key_file(path: Path) -> bytes:
    """Return a persistent 256-bit key, creating it with restrictive permissions."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        encoded = path.read_text(encoding="utf-8").strip()
        key = _b64decode(encoded)
        if len(key) != 32:
            raise NiceGUICryptoError(f"Invalid 256-bit key file: {path}")
        return key
    key = AESGCM.generate_key(bit_length=256)
    path.write_text(_b64encode(key), encoding="utf-8")
    with suppress(PermissionError):
        path.chmod(0o600)
    return key


class CryptoManager:
    """Authenticated envelope encryption for per-user persisted fields."""

    def __init__(self, user_key_file: Path | str) -> None:
        self.key_path = Path(user_key_file).expanduser()
        self._kek = ensure_binary_key_file(self.key_path)

    def new_data_key(self) -> bytes:
        """Return a new per-user data encryption key."""
        return AESGCM.generate_key(bit_length=256)

    def wrap_user_key(self, *, owner_key_id: str, data_key: bytes) -> str:
        """Encrypt one per-user DEK with the server KEK."""
        return self._encrypt(
            key=self._kek,
            plaintext=data_key,
            aad=f"user-key:{owner_key_id}:v1",
        )

    def unwrap_user_key(self, *, owner_key_id: str, envelope: str) -> bytes:
        """Decrypt one per-user DEK with the server KEK."""
        return self._decrypt(
            key=self._kek,
            envelope=envelope,
            aad=f"user-key:{owner_key_id}:v1",
        )

    def encrypt_text(self, *, data_key: bytes, plaintext: str, aad: str) -> str:
        """Encrypt one UTF-8 text value."""
        return self._encrypt(key=data_key, plaintext=plaintext.encode("utf-8"), aad=aad)

    def decrypt_text(self, *, data_key: bytes, envelope: str, aad: str) -> str:
        """Decrypt one UTF-8 text value."""
        return self._decrypt(key=data_key, envelope=envelope, aad=aad).decode("utf-8")

    def _encrypt(self, *, key: bytes, plaintext: bytes, aad: str) -> str:
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad.encode("utf-8"))
        return json.dumps(
            {
                "v": _ENVELOPE_VERSION,
                "alg": _ALGORITHM,
                "nonce": _b64encode(nonce),
                "ciphertext": _b64encode(ciphertext),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _decrypt(self, *, key: bytes, envelope: str, aad: str) -> bytes:
        try:
            payload = json.loads(envelope)
            if (
                payload.get("v") != _ENVELOPE_VERSION
                or payload.get("alg") != _ALGORITHM
            ):
                raise ValueError("unsupported encrypted envelope")
            nonce = _b64decode(str(payload["nonce"]))
            ciphertext = _b64decode(str(payload["ciphertext"]))
            return AESGCM(key).decrypt(nonce, ciphertext, aad.encode("utf-8"))
        except (KeyError, TypeError, ValueError, InvalidTag) as exc:
            raise NiceGUICryptoError(
                "Could not decrypt NiceGUI persisted data."
            ) from exc


__all__ = [
    "CryptoManager",
    "NiceGUICryptoError",
    "ensure_binary_key_file",
    "ensure_text_key_file",
]
