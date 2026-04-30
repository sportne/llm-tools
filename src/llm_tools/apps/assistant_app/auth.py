"""Hosted-mode authentication helpers for the assistant app."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from llm_tools.apps.assistant_app.models import NiceGUIHostedConfig, NiceGUIUser
from llm_tools.apps.assistant_app.paths import expand_app_path
from llm_tools.apps.assistant_app.store import SQLiteNiceGUIChatStore

_DEFAULT_HOSTED_DIR = Path.home() / ".llm-tools" / "assistant" / "nicegui" / "hosted"
_DEFAULT_SESSION_DAYS = 7
_PASSWORD_HASHER = PasswordHasher()


@dataclass(frozen=True)
class HostedStartupValidation:
    """Resolved hosted-mode startup safety state."""

    config: NiceGUIHostedConfig
    is_loopback: bool
    tls_enabled: bool


def is_loopback_host(host: str | None) -> bool:
    """Return whether a host binding is loopback-only."""
    normalized = (host or "127.0.0.1").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def ensure_secret_file(path: Path, *, token_bytes: int = 32) -> str:
    """Return a persistent local secret, creating it when missing."""
    path = expand_app_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    value = secrets.token_urlsafe(token_bytes)
    path.write_text(value, encoding="utf-8")
    with suppress(PermissionError):
        path.chmod(0o600)
    return value


def validate_hosted_startup(
    *,
    auth_mode: str,
    host: str,
    public_base_url: str | None,
    tls_certfile: str | None,
    tls_keyfile: str | None,
    allow_insecure_hosted_secrets: bool,
    secret_key_path: Path | None = None,
) -> HostedStartupValidation:
    """Resolve hosted-mode startup policy and fail closed for unsafe bindings."""
    mode = auth_mode.strip().lower()
    if mode not in {"none", "local"}:
        raise ValueError("auth_mode must be 'none' or 'local'.")
    if bool(tls_certfile) != bool(tls_keyfile):
        raise ValueError("--tls-certfile and --tls-keyfile must be provided together.")
    loopback = is_loopback_host(host)
    tls_enabled = bool(tls_certfile and tls_keyfile) or bool(
        public_base_url and public_base_url.lower().startswith("https://")
    )
    if not loopback and mode == "none":
        raise ValueError("Non-loopback NiceGUI hosting requires --auth-mode local.")
    secret_entry_enabled = True
    warning = None
    if not loopback and not tls_enabled and not allow_insecure_hosted_secrets:
        secret_entry_enabled = False
        warning = (
            "Secret entry is disabled because hosted mode is reachable without HTTPS."
        )
    resolved_secret_key_path = expand_app_path(
        secret_key_path or _DEFAULT_HOSTED_DIR / "app.secret"
    )
    return HostedStartupValidation(
        config=NiceGUIHostedConfig(
            auth_mode=mode,  # type: ignore[arg-type]
            public_base_url=public_base_url,
            tls_certfile=tls_certfile,
            tls_keyfile=tls_keyfile,
            allow_insecure_hosted_secrets=allow_insecure_hosted_secrets,
            secret_key_path=str(resolved_secret_key_path),
            secret_entry_enabled=secret_entry_enabled,
            insecure_hosted_warning=warning,
        ),
        is_loopback=loopback,
        tls_enabled=tls_enabled,
    )


def hash_password(password: str) -> str:
    """Return an Argon2 password hash."""
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Return whether a plaintext password matches an Argon2 hash."""
    try:
        return bool(_PASSWORD_HASHER.verify(password_hash, password))
    except (InvalidHashError, VerifyMismatchError, VerificationError):
        return False


def new_session_token() -> str:
    """Return a random browser session token."""
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """Return a stable hash for a browser session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LocalAuthProvider:
    """Local username/password auth provider backed by the NiceGUI store."""

    def __init__(
        self,
        store: SQLiteNiceGUIChatStore,
        *,
        session_days: int = _DEFAULT_SESSION_DAYS,
    ) -> None:
        self.store = store
        self.session_days = session_days

    def has_users(self) -> bool:
        """Return whether local auth has any users."""
        return bool(self.store.list_users())

    def create_user(
        self, *, username: str, password: str, role: str = "user"
    ) -> NiceGUIUser:
        """Create a local user with an Argon2 password hash."""
        cleaned_username = username.strip()
        if not cleaned_username:
            raise ValueError("username must not be empty")
        if not password:
            raise ValueError("password must not be empty")
        if role not in {"admin", "user"}:
            raise ValueError("role must be 'admin' or 'user'")
        user = self.store.create_user(
            username=cleaned_username,
            password_hash=hash_password(password),
            role=role,  # type: ignore[arg-type]
        )
        self.store.record_auth_event(
            user_id=user.user_id, event_type="user_created", detail={"role": role}
        )
        return user

    def reset_password(self, *, user_id: str, password: str) -> NiceGUIUser | None:
        """Replace one local user's password hash."""
        if not password:
            raise ValueError("password must not be empty")
        self.store.update_user_password_hash(user_id, hash_password(password))
        self.store.record_auth_event(user_id=user_id, event_type="password_reset")
        return self.store.get_user(user_id)

    def authenticate(self, *, username: str, password: str) -> NiceGUIUser | None:
        """Return the authenticated user or None."""
        user = self.store.get_user_by_username(username)
        if user is None or user.disabled:
            self.store.record_auth_event(
                user_id=user.user_id if user else None,
                event_type="login_failed",
                detail={"username": username},
            )
            return None
        if not verify_password(user.password_hash, password):
            self.store.record_auth_event(
                user_id=user.user_id,
                event_type="login_failed",
                detail={"username": username},
            )
            return None
        self.store.update_user_login(user.user_id)
        self.store.record_auth_event(user_id=user.user_id, event_type="login")
        return user

    def create_session(self, user_id: str) -> tuple[str, str]:
        """Create a browser session and return (session_id, token)."""
        token = new_session_token()
        expires_at = (datetime.now(UTC) + timedelta(days=self.session_days)).isoformat()
        session = self.store.create_user_session(
            user_id=user_id,
            token_hash=hash_session_token(token),
            expires_at=expires_at,
        )
        return session.session_id, token

    def user_for_session(self, session_id: str, token: str) -> NiceGUIUser | None:
        """Return the user for a valid browser session."""
        session = self.store.get_user_session(session_id)
        if session is None or session.revoked_at is not None:
            return None
        if datetime.fromisoformat(session.expires_at) <= datetime.now(UTC):
            return None
        if not hmac.compare_digest(session.token_hash, hash_session_token(token)):
            return None
        user = self.store.get_user(session.user_id)
        if user is None or user.disabled:
            return None
        return user

    def revoke_session(self, session_id: str) -> None:
        """Revoke one browser session."""
        self.store.revoke_user_session(session_id)


__all__ = [
    "HostedStartupValidation",
    "LocalAuthProvider",
    "ensure_secret_file",
    "hash_password",
    "hash_session_token",
    "is_loopback_host",
    "new_session_token",
    "validate_hosted_startup",
    "verify_password",
]
