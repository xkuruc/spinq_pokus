"""Read-first audit of SpinQLabLink. No network work occurs at import time."""

from .adapter import audit_existing_client

__all__ = ["audit_existing_client"]
