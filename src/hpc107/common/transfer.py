"""Shared safety checks for command-generation-only data transfers."""

from __future__ import annotations

import re

from .errors import ConfigurationError

_UNSAFE_REMOTE = re.compile(r"[;|&$`\\\r\n]")


def validate_pan_remote(remote: str) -> None:
    """Reject empty or shell-sensitive Pan remote bases."""
    if not remote:
        raise ConfigurationError("A Pan remote is required")
    if _UNSAFE_REMOTE.search(remote):
        raise ConfigurationError("The Pan remote contains unsupported shell metacharacters")
