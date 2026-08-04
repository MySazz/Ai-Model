"""Deterministic privacy classification before remote model disclosure."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum


class PrivacyLevel(IntEnum):
    PUBLIC = 0
    CLOUD_ALLOWED = 1
    ASK_BEFORE_CLOUD = 2
    LOCAL_ONLY = 3


@dataclass(frozen=True)
class PrivacyAssessment:
    level: PrivacyLevel
    reasons: tuple[str, ...]


_LOCAL_ONLY_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("cloud access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("generic secret assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"']?[^\s\"']{8,}"
    )),
)

_SENSITIVE_HINTS = re.compile(
    r"(?i)\b(private|confidential|personal|credential|production|customer data|medical|financial)\b"
)


class PrivacyClassifier:
    """Conservative local scanner; model output is never trusted for this decision."""

    def classify(self, text: str) -> PrivacyAssessment:
        reasons = tuple(label for label, pattern in _LOCAL_ONLY_PATTERNS if pattern.search(text))
        if reasons:
            return PrivacyAssessment(PrivacyLevel.LOCAL_ONLY, reasons)
        if _SENSITIVE_HINTS.search(text):
            return PrivacyAssessment(
                PrivacyLevel.ASK_BEFORE_CLOUD,
                ("sensitive-language heuristic",),
            )
        return PrivacyAssessment(PrivacyLevel.CLOUD_ALLOWED, ())
