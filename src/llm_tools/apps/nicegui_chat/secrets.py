"""Encrypted local secret storage for hosted NiceGUI chat sessions."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from llm_tools.apps.nicegui_chat.store import SQLiteNiceGUIChatStore


class NiceGUISecretStoreError(RuntimeError):
    """Raised for encrypted secret store failures."""


def ensure_master_key(path: Path | str) -> bytes:
    """Return a persistent Fernet master key, creating it when missing."""
    key_path = Path(path).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        key = key_path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        with suppress(PermissionError):
            key_path.chmod(0o600)
    try:
        Fernet(key)
    except ValueError as exc:
        raise NiceGUISecretStoreError(
            f"Invalid NiceGUI secret master key at {key_path}."
        ) from exc
    return key


class SQLiteSecretStore:
    """Encrypted SQLite-backed secret store for hosted mode."""

    def __init__(self, store: SQLiteNiceGUIChatStore, *, master_key_path: Path | str):
        self.store = store
        self.master_key_path = Path(master_key_path).expanduser()
        self._fernet = Fernet(ensure_master_key(self.master_key_path))

    def set_secret(
        self, *, owner_user_id: str, session_id: str, name: str, value: str
    ) -> None:
        """Encrypt and store one secret value."""
        cleaned_name = name.strip()
        cleaned_value = value.strip()
        if not cleaned_name:
            raise ValueError("secret name must not be empty")
        if not cleaned_value:
            return
        ciphertext = self._fernet.encrypt(cleaned_value.encode("utf-8")).decode("ascii")
        self.store.upsert_secret_record(
            owner_user_id=owner_user_id,
            session_id=session_id,
            name=cleaned_name,
            ciphertext=ciphertext,
        )

    def get_secret(
        self, *, owner_user_id: str, session_id: str, name: str
    ) -> str | None:
        """Decrypt one secret value."""
        record = self.store.get_secret_record(
            owner_user_id=owner_user_id,
            session_id=session_id,
            name=name,
        )
        if record is None:
            return None
        try:
            return self._fernet.decrypt(record.ciphertext.encode("ascii")).decode(
                "utf-8"
            )
        except InvalidToken as exc:
            raise NiceGUISecretStoreError(
                f"Could not decrypt NiceGUI secret {name!r}."
            ) from exc

    def has_secret(self, *, owner_user_id: str, session_id: str, name: str) -> bool:
        """Return whether a secret is configured without decrypting it."""
        return (
            self.store.get_secret_record(
                owner_user_id=owner_user_id,
                session_id=session_id,
                name=name,
            )
            is not None
        )

    def delete_secret(self, *, owner_user_id: str, session_id: str, name: str) -> None:
        """Delete one secret value."""
        self.store.delete_secret_record(
            owner_user_id=owner_user_id,
            session_id=session_id,
            name=name,
        )

    def secret_names(self, *, owner_user_id: str, session_id: str) -> list[str]:
        """Return configured secret names."""
        return self.store.list_secret_names(
            owner_user_id=owner_user_id,
            session_id=session_id,
        )

    def tool_env(self, *, owner_user_id: str, session_id: str) -> dict[str, str]:
        """Return all configured non-provider tool secrets for execution."""
        env: dict[str, str] = {}
        for name in self.secret_names(
            owner_user_id=owner_user_id, session_id=session_id
        ):
            if name == "__provider_api_key__":
                continue
            value = self.get_secret(
                owner_user_id=owner_user_id,
                session_id=session_id,
                name=name,
            )
            if value:
                env[name] = value
        return env


__all__ = [
    "NiceGUISecretStoreError",
    "SQLiteSecretStore",
    "ensure_master_key",
]
