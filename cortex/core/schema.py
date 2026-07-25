from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class Verdict(str, Enum):
    VULNERABLE = "vulnerable"
    LIKELY_SAFE = "likely_safe"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # first-class, not an error path


class Finding(BaseModel):
    verdict: Verdict
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None  # §3.2 grounding: strip when ungrounded
    severity: Optional[Literal["Critical", "High", "Medium", "Low"]] = None
    line_hint: Optional[str] = None
    explanation: str
    attack_path: Optional[str] = None  # "how it's abused" — NOT working exploit code
    fix: Optional[str] = None
    grounded_on: list[str] = Field(default_factory=list)  # retrieved doc ids; empty => must be INSUFFICIENT_EVIDENCE


class Report(BaseModel):
    findings: list[Finding]
    overall_risk: int = Field(ge=0, le=100)
    abstained: bool = False
