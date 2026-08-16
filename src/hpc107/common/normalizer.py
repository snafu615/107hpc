"""Future extension boundary for project normalizers, including optional LLMs.

The deterministic package does not provide an implementation. A future adapter
may propose a manifest, but the proposal must pass the same deterministic
configuration and project validators before it can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import ProjectReport


@dataclass(frozen=True, slots=True)
class ManifestProposal:
    yaml_text: str
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    unresolved: tuple[str, ...]


class ProjectNormalizer(Protocol):
    def propose_manifest(self, report: ProjectReport) -> ManifestProposal:
        """Propose a manifest without performing execution or remote actions."""
