"""Multimodal architecture review — reason over an uploaded infrastructure diagram.

Division of labour, chosen to keep the honesty guarantees intact:

  * the **model** reads the picture. Topology is visible evidence, so mapping it to
    MITRE ATT&CK techniques and NIST SP 800-53 control families is something Gemma 4
    can legitimately do from the image plus its own training.
  * the **local index** supplies CVEs. A diagram naming "Tomcat 9" does not prove which
    2025/2026 advisories apply, and Gemma's cutoff is Jan 2025 — so CVEs are attached by
    structured lookup over the retrieved KEV/OSV docs, never recalled by the model.

Any CVE the model names anyway is stripped by `ground_diagram_report`, exactly like the
text path. Vision requires a stock gemma4 tag; the SecOps GGUF fine-tune has no vision.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from core.engine import Engine
from core.harness import extract_json
from core.rag import RAG

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)

Exposure = Literal["internet-facing", "dmz", "internal", "unknown"]
Severity = Literal["Critical", "High", "Medium", "Low"]


class DiagramFinding(BaseModel):
    component: str
    exposure: Exposure = "unknown"
    severity: Severity = "Medium"
    mitre_technique: str = "unknown"  # e.g. T1190
    mitre_tactic: str = "unknown"  # e.g. Initial Access
    nist_control: str = "unknown"  # e.g. SC-7
    nist_rationale: str = ""
    explanation: str
    attack_path: Optional[str] = None
    fix: Optional[str] = None
    cve_id: Optional[str] = None  # stripped unless a retrieved doc backs it
    grounded_on: list[str] = Field(default_factory=list)


MAX_FINDINGS = 5


class DiagramReport(BaseModel):
    components: list[str] = Field(default_factory=list)
    # Bounded so constrained decoding stops before the token budget runs out.
    findings: list[DiagramFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    overall_risk: int = Field(default=0, ge=0, le=100)
    abstained: bool = False
    notes: str = ""


@dataclass
class AdvisoryHit:
    """A CVE attached to a diagram component by local lookup, not by recall."""

    component: str
    doc_id: str
    cve_id: str
    text: str


@dataclass
class VisionResult:
    report: DiagramReport
    advisories: list[AdvisoryHit] = field(default_factory=list)
    model: str = ""
    ms: float = 0.0
    stripped_cves: list[str] = field(default_factory=list)


DIAGRAM_SCHEMA_HINT = """
Return ONE JSON object only, no prose:
{
  "components": ["short name of each system you can see"],
  "findings": [
    {
      "component": "which component this is about",
      "exposure": "internet-facing" | "dmz" | "internal" | "unknown",
      "severity": "Critical" | "High" | "Medium" | "Low",
      "mitre_technique": "MITRE ATT&CK id like T1190",
      "mitre_tactic": "tactic name like Initial Access",
      "nist_control": "NIST SP 800-53 control id like SC-7, AC-4, IA-2, SI-4, AU-6",
      "nist_rationale": "one sentence: which control objective this design fails",
      "explanation": "one sentence: the architectural weakness you can SEE",
      "attack_path": "one sentence: how an attacker would move through it",
      "fix": "one sentence: the defensive change"
    }
  ],
  "overall_risk": 0-100,
  "notes": "anything unreadable in the diagram"
}
Rules:
- At most 5 findings. Most severe first. One short sentence per field.
- Judge ONLY what is visible: trust boundaries, exposed services, flat networks,
  missing segmentation, unauthenticated paths, single points of failure, plaintext links.
- Do NOT output CVE identifiers. You have no evidence for them; they are attached
  separately from a local advisory index.
- NIST mappings are advisory control references, not a compliance score.
- attack_path is a defensive explanation, never working exploit code.
- If the image is not an infrastructure diagram, return empty findings and say so in notes.
""".strip()

DIAGRAM_SYSTEM = (
    "You are CORTEX, an offline blue-team architecture reviewer looking at an "
    "infrastructure diagram. Identify what you can actually see and map each weakness "
    "to a MITRE ATT&CK technique and a NIST SP 800-53 control.\n\n" + DIAGRAM_SCHEMA_HINT
)


def diagram_json_schema() -> dict[str, Any]:
    return DiagramReport.model_json_schema()


def ground_diagram_report(
    report: DiagramReport, allowed_doc_ids: Optional[set[str]] = None
) -> tuple[DiagramReport, list[str]]:
    """Strip any CVE the model asserted without a retrieved doc behind it.

    Unlike the text path this keeps the finding: the MITRE/NIST reasoning came from the
    picture and stands on its own. Only the unsupported CVE claim is removed.
    """
    allowed = allowed_doc_ids if allowed_doc_ids is not None else set()
    stripped: list[str] = []
    findings: list[DiagramFinding] = []
    for f in report.findings:
        grounded = [g for g in (f.grounded_on or []) if g in allowed]
        cve = f.cve_id
        if cve and not grounded:
            stripped.append(cve)
            cve = None
        # The model sometimes smuggles a CVE into prose instead of the field.
        blob = " ".join(x for x in [f.explanation, f.attack_path or "", f.fix or ""] if x)
        if not grounded:
            for hidden in CVE_RE.findall(blob):
                stripped.append(hidden)
            if stripped:
                blob_clean = CVE_RE.sub("[unverified CVE removed]", f.explanation)
                f = f.model_copy(update={"explanation": blob_clean})
        findings.append(f.model_copy(update={"cve_id": cve, "grounded_on": grounded}))
    return report.model_copy(update={"findings": findings}), stripped


class VisionAnalyzer:
    """Diagram → visible-weakness findings, then locally-grounded advisories."""

    # Advisory lookup is only meaningful for product names. Generic infrastructure
    # nouns ("backup", "remote", "controller") match unrelated CVEs and make the
    # evidence panel look like keyword soup, so they are excluded.
    STOPWORDS = {
        "the", "and", "for", "with", "server", "service", "services", "cluster",
        "internal", "external", "public", "private", "network", "user", "users",
        "client", "clients", "data", "database", "unknown", "internet", "cloud",
        "zone", "subnet", "gateway", "instance", "node", "app", "application",
        "remote", "admin", "admins", "administrator", "primary", "secondary",
        "host", "hosts", "jump", "bastion", "backup", "backups", "collector",
        "proxy", "reverse", "controller", "domain", "partner", "partners",
        "tier", "edge", "firewall", "load", "balancer", "storage", "share",
        "shared", "account", "accounts", "access", "port", "ports", "vlan",
        "primary", "replica", "queue", "cache", "worker", "workers", "api",
        "web", "mail", "file", "print", "log", "logs", "logging", "monitor",
        "monitoring", "console", "portal", "dashboard", "endpoint", "endpoints",
    }

    def __init__(
        self,
        engine: Engine,
        rag: Optional[RAG] = None,
        *,
        config: Optional[dict[str, Any]] = None,
    ):
        self.engine = engine
        self.config = config or engine.config
        self.rag = rag
        vision_cfg = self.config.get("vision", {})
        wanted = self.config.get("models", {}).get("multimodal", "gemma4:12b")
        self.model = self.engine._prefix_match(wanted) or self._first_stock_gemma()
        self.num_predict = int(vision_cfg.get("num_predict", 1100))
        self.max_advisories = int(vision_cfg.get("max_advisories", 6))
        # Gemma's vision tower works on 896px tiles; anything larger triggers
        # pan-and-scan multi-crop, which multiplies image tokens and latency.
        self.max_edge = int(vision_cfg.get("max_edge", 896))
        self.constrain_json = bool(
            self.config.get("harness", {}).get("constrain_json", True)
        )

    def _first_stock_gemma(self) -> Optional[str]:
        """Only stock gemma4 tags have vision — GGUF repacks and the SecOps fine-tune
        drop the vision tower, so never fall back onto an org-prefixed tag."""
        for name in self.engine.available:
            low = name.lower()
            if low.startswith("gemma4") and "gguf" not in low and "/" not in low:
                return name
        return None

    @property
    def available(self) -> bool:
        return bool(self.model)

    def _prepare_image(self, path: Path) -> Path:
        """Fit the diagram inside one vision tile. Cuts latency several-fold."""
        try:
            from PIL import Image
        except ImportError:
            return path
        try:
            with Image.open(path) as im:
                if max(im.size) <= self.max_edge:
                    return path
                im = im.convert("RGB")
                im.thumbnail((self.max_edge, self.max_edge), Image.LANCZOS)
                out_dir = Path(self.config.get("cache", {}).get("dir", "data/cache")) / "vision"
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{path.stem}-{im.width}x{im.height}.png"
                im.save(out)
                return out
        except Exception:
            return path

    def analyze(self, image_path: str) -> VisionResult:
        import time

        if not self.available:
            return VisionResult(
                report=DiagramReport(
                    abstained=True,
                    notes="No vision-capable Gemma 4 tag found. Pull gemma4:12b or gemma4:e4b.",
                ),
                model="",
            )

        path = Path(image_path)
        if not path.is_file():
            return VisionResult(
                report=DiagramReport(abstained=True, notes=f"Image not readable: {image_path}"),
                model=self.model,
            )

        t0 = time.perf_counter()
        try:
            raw = self.engine.generate(
                "Review this infrastructure diagram for security weaknesses. "
                "Return the JSON object described in the system prompt.",
                system=DIAGRAM_SYSTEM,
                images=[str(self._prepare_image(path))],
                model=self.model,
                role="deep",
                temperature=0.2,
                num_predict=self.num_predict,
                format=diagram_json_schema() if self.constrain_json else None,
                label="vision",
                max_retries=1,
            )
            report = DiagramReport.model_validate(extract_json(raw))
        except (ValidationError, ValueError, json.JSONDecodeError, TypeError) as e:
            return VisionResult(
                report=DiagramReport(
                    abstained=True, notes=f"Model did not return a valid diagram report: {e}"
                ),
                model=self.model,
                ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:
            return VisionResult(
                report=DiagramReport(abstained=True, notes=f"Vision analysis failed: {e}"),
                model=self.model,
                ms=(time.perf_counter() - t0) * 1000.0,
            )

        advisories = self._lookup_advisories(report)
        allowed = {a.doc_id for a in advisories}
        report, stripped = ground_diagram_report(report, allowed_doc_ids=allowed)
        if not report.findings:
            report = report.model_copy(update={"abstained": True})

        return VisionResult(
            report=report,
            advisories=advisories,
            model=self.model,
            ms=(time.perf_counter() - t0) * 1000.0,
            stripped_cves=sorted(set(stripped)),
        )

    def _terms(self, report: DiagramReport) -> list[str]:
        seen: list[str] = []
        for raw in list(report.components) + [f.component for f in report.findings]:
            for token in re.split(r"[^A-Za-z0-9.+-]+", raw or ""):
                t = token.strip(".-+").lower()
                if len(t) < 3 or t in self.STOPWORDS or t.isdigit():
                    continue
                if t not in seen:
                    seen.append(t)
        return seen

    def _lookup_advisories(self, report: DiagramReport) -> list[AdvisoryHit]:
        """Structured component → advisory lookup against the local post-cutoff index."""
        if self.rag is None:
            return []
        hits: list[AdvisoryHit] = []
        seen_docs: set[str] = set()
        for term in self._terms(report):
            docs = self.rag.lookup_package(term)
            if not docs:
                # KEV entries are keyed by CVE, with vendor/product only in the text, so
                # fall back to semantic search — then keep only docs that actually name
                # the component. A near-miss neighbour is not evidence.
                docs = [
                    d
                    for d in self.rag.retrieve_for_query(f"{term} vulnerability", top_k=4)
                    if term in (d.text or "").lower()
                ]
            if not docs:
                continue
            for d in docs:
                if d.id in seen_docs:
                    continue
                cve = (d.metadata or {}).get("cve_id") or ""
                if not cve:
                    found = CVE_RE.search(d.text or "")
                    cve = found.group(0) if found else ""
                seen_docs.add(d.id)
                hits.append(
                    AdvisoryHit(
                        component=term,
                        doc_id=d.id,
                        cve_id=cve,
                        text=(d.text or "")[:300],
                    )
                )
                if len(hits) >= self.max_advisories:
                    return hits
        return hits
