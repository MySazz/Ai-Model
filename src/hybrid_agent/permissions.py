"""Graduated-autonomy policy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class RiskLevel(IntEnum):
    OBSERVE = 0
    SAFE = 1
    APPROVAL = 2
    EXPLICIT = 3
    PROHIBITED = 4


@dataclass(frozen=True)
class Decision:
    allowed: bool
    requires_approval: bool
    reason: str


class PermissionPolicy:
    """Evaluate tool risk without allowing the agent to promote its own access."""

    def decide(self, risk: RiskLevel, approved: bool = False) -> Decision:
        if risk is RiskLevel.PROHIBITED:
            return Decision(False, False, "This action is prohibited by policy.")
        if risk >= RiskLevel.APPROVAL and not approved:
            return Decision(False, True, "Human approval is required.")
        return Decision(True, False, "Allowed by the current autonomy policy.")
