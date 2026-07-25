"""Forced-JSON harness: validate, retry, grounding rule, abstain."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError

from core.engine import Engine
from core.schema import Finding, Report, Verdict

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
KEV_RE = re.compile(r"\bKEV-[A-Z0-9-]+\b", re.IGNORECASE)

REPORT_SCHEMA_HINT = """
You MUST return a single JSON object matching this schema (no prose outside JSON):
{
  "findings": [
    {
      "verdict": "vulnerable" | "likely_safe" | "insufficient_evidence",
      "cwe_id": string | null,
      "cve_id": string | null,
      "severity": "Critical" | "High" | "Medium" | "Low" | null,
      "line_hint": string | null,
      "explanation": string,
      "attack_path": string | null,
      "fix": string | null,
      "grounded_on": [string]   // ONLY ids from the retrieved evidence list provided
    }
  ],
  "overall_risk": integer 0-100,
  "abstained": boolean
}
Rules:
- If you cannot support a claim with retrieved evidence, set verdict to insufficient_evidence.
- Do NOT invent CVE/KEV identifiers that are not in the retrieved evidence.
- attack_path explains how abuse works defensively; never emit working exploit code.
- Return JSON only.
""".strip()


def extract_json(text: str) -> Any:
    """Extract JSON object; tolerate ```json fences."""
    if text is None:
        raise ValueError("empty model output")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Find outermost {...}
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start : end + 1])
        raise


def _names_specific_cve_or_kev(finding: Finding) -> bool:
    if finding.cve_id:
        return True
    blob = " ".join(
        x
        for x in [
            finding.explanation or "",
            finding.attack_path or "",
            finding.fix or "",
            finding.line_hint or "",
            finding.cwe_id or "",
        ]
        if x
    )
    return bool(CVE_RE.search(blob) or KEV_RE.search(blob))


def apply_grounding_rule(
    report: Report,
    allowed_doc_ids: Optional[set[str]] = None,
) -> Report:
    """
    §3.2 Grounding rule (post-hoc validator, not a prompt suggestion):

    For any finding that names a specific CVE/KEV entry:
      if finding.grounded_on is empty (no retrieved doc supports it):
          force finding.verdict = INSUFFICIENT_EVIDENCE
          strip the unsupported cve_id
    Also: grounded_on entries must be subset of allowed retrieved doc ids when provided;
    fabricated citation strings do not count.
    Empty grounded_on on a CVE/KEV claim => insufficient_evidence.
    """
    allowed = allowed_doc_ids  # None means do not filter ids, still enforce empty check

    new_findings: list[Finding] = []
    for f in report.findings:
        grounded = list(f.grounded_on or [])
        if allowed is not None:
            grounded = [g for g in grounded if g in allowed]

        names_cve = _names_specific_cve_or_kev(f)
        # §3.2: CVE/KEV claim with empty grounded_on => insufficient_evidence + strip cve_id
        if names_cve and not grounded:
            f = f.model_copy(
                update={
                    "verdict": Verdict.INSUFFICIENT_EVIDENCE,
                    "cve_id": None,
                    "grounded_on": [],
                }
            )
        else:
            # Drop fabricated citation ids not in retrieved set
            f = f.model_copy(update={"grounded_on": grounded})
        new_findings.append(f)

    abstained = report.abstained or (
        bool(new_findings)
        and all(x.verdict == Verdict.INSUFFICIENT_EVIDENCE for x in new_findings)
    )
    return report.model_copy(update={"findings": new_findings, "abstained": abstained})


def abstain_report(reason: str) -> Report:
    return Report(
        findings=[
            Finding(
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                explanation=reason,
                grounded_on=[],
            )
        ],
        overall_risk=0,
        abstained=True,
    )


def report_json_schema() -> dict[str, Any]:
    """JSON schema handed to Ollama `format` for grammar-constrained decoding.

    Constraining the decoder is a latency lever, not just a correctness one: the first
    response is already schema-valid, so the retry round-trips almost never fire.
    """
    return Report.model_json_schema()


class Harness:
    """Strict JSON schema loop with retry-on-invalid and grounding."""

    def __init__(self, engine: Engine, config: Optional[dict[str, Any]] = None):
        self.engine = engine
        self.config = config or engine.config
        harness_cfg = self.config.get("harness", {})
        self.max_retries = int(harness_cfg.get("max_retries", 2))
        self.constrain_json = bool(harness_cfg.get("constrain_json", True))
        samp = self.config.get("sampling", {}).get("extraction", {})
        self.temperature = float(samp.get("temperature", 0.2))
        self.top_p = float(samp.get("top_p", 0.95))
        self.top_k = int(samp.get("top_k", 64))

    def run(
        self,
        user_prompt: str,
        *,
        system_extra: str = "",
        retrieved_doc_ids: Optional[list[str]] = None,
        role: str = "deep",
        model: Optional[str] = None,
        label: str = "harness",
    ) -> Report:
        """
        1. Prompt with schema in system (Gemma 4 supports system role) + grammar-constrained format.
        2. Extract JSON, Report.model_validate.
        3. On ValidationError: retry up to max_retries with the error appended.
        4. After the retries: return Report with abstained=True.
        5. Run grounding rule (§3.2) over every finding.
        """
        allowed = set(retrieved_doc_ids or [])
        evidence_block = ""
        if retrieved_doc_ids is not None:
            evidence_block = (
                "\nRetrieved evidence doc ids (grounded_on MUST use only these): "
                + json.dumps(list(retrieved_doc_ids))
            )

        system = REPORT_SCHEMA_HINT + evidence_block
        if system_extra:
            system = system_extra.strip() + "\n\n" + system

        prompt = user_prompt
        last_err: Optional[str] = None
        fmt: Optional[dict[str, Any]] = report_json_schema() if self.constrain_json else None

        # max_retries = 2 means: initial + 2 retries = 3 attempts total
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.engine.generate(
                    prompt,
                    system=system,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    role=role,
                    model=model,
                    format=fmt,
                    label=label,
                )
                data = extract_json(raw)
                report = Report.model_validate(data)
                return apply_grounding_rule(
                    report,
                    allowed_doc_ids=allowed if retrieved_doc_ids is not None else None,
                )
            except (ValidationError, ValueError, json.JSONDecodeError, TypeError) as e:
                last_err = str(e)
                if attempt < self.max_retries:
                    prompt = (
                        user_prompt
                        + f"\n\nYour last output failed because: {last_err}; "
                        "return valid JSON matching the schema"
                    )
                    continue
                return abstain_report(
                    f"Model failed schema validation after retries: {last_err}"
                )
            except Exception as e:
                last_err = str(e)
                # A backend that rejects the schema grammar shouldn't cost us the analysis.
                if fmt is not None:
                    fmt = None
                    continue
                return abstain_report(f"Generation failed: {last_err}")

        return abstain_report(f"Model failed schema validation after retries: {last_err}")
