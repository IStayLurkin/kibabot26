from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OSINTRequest:
    target_type: str
    target_value: str
    mode: str
    authorization: bool = False
    requester_name: str = ""
    requester_id: int | None = None
    options: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OSINTResult:
    summary: str
    findings: list[str]
    sources: list[str]
    warnings: list[str]
    blocked_reason: str = ""
    raw_sections: dict[str, str] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return not self.blocked_reason
