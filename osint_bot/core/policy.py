from __future__ import annotations

from dataclasses import dataclass

from osint_bot.core.constants import OWNED_ASSET_TARGET_TYPES, SAFE_USAGE_POLICY

BLOCKED_PATTERNS = (
    "password",
    "credential",
    "login",
    "exploit",
    "dropper",
    "payload",
    "phish",
    "steal",
    "private email",
    "non-public",
)


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    warnings: list[str]
    blocked_reason: str = ""


def evaluate_request(target_type: str, target_value: str, authorization: bool) -> PolicyDecision:
    text = f"{target_type} {target_value}".lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in text:
            return PolicyDecision(
                allowed=False,
                warnings=[],
                blocked_reason=f"Request blocked by safety policy. {SAFE_USAGE_POLICY}",
            )

    warnings: list[str] = []
    if target_type in OWNED_ASSET_TARGET_TYPES and not authorization:
        warnings.append(
            "Active infrastructure checks are limited to assets you own or are authorized to assess."
        )

    return PolicyDecision(allowed=True, warnings=warnings)
