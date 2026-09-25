"""Filesystem, redaction and reproducibility helpers (standard library only)."""

from __future__ import annotations

import html
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_KEYS = {"password", "token", "session_id", "sessionid", "account", "authorization", "cookie", "secret", "credential"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if any(part in str(k).lower().replace("-", "_")
                                                   for part in SECRET_KEYS) else redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        # Guard against accidental free-text token inclusion in logs/reports.
        import re
        value = re.sub(r"gh[pousr]_[A-Za-z0-9]{16,}", "[REDACTED_TOKEN]", value)
        return re.sub(r"(?i)\b(password|token|session[_-]?id)\s*[:=]\s*\S+",
                      r"\1=[REDACTED]", value)
    return value


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(redact(value), ensure_ascii=False, indent=2,
                                  allow_nan=False, default=str).encode("utf-8"))


def escaped(value: Any) -> str:
    return html.escape(str(value), quote=True)
