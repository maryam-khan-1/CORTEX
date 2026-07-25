"""Two-stage triage: E2B classify → SecOps deep analyze on flagged subset."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from core.engine import Engine
from core.harness import Harness
from core.rag import RAG
from core.schema import Report

STAGE1_SCHEMA = """
Return JSON only:
{"label":"benign"|"suspicious"|"critical","mitre":"Txxxx or unknown","rationale":"short"}
""".strip()


@dataclass
class Stage1Result:
    item: str
    label: str
    mitre: str
    rationale: str
    votes: dict[str, int] = field(default_factory=dict)
    consensus: str = ""  # e.g. "Critical (3/3 agree)"
    agreement: float = 0.0


@dataclass
class TriageResult:
    stage1: list[Stage1Result]
    reports: dict[int, Report]  # index into stage1 -> Report
    flagged_indices: list[int]


def _extract_obj(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start : end + 1])
        raise


class Triage:
    def __init__(
        self,
        engine: Engine,
        harness: Optional[Harness] = None,
        rag: Optional[RAG] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        self.engine = engine
        self.harness = harness or Harness(engine)
        self.rag = rag
        self.config = config or engine.config
        self.consensus_n = int(self.config.get("triage", {}).get("consensus_n", 3))
        cons = self.config.get("sampling", {}).get("consensus", {})
        self.cons_temp = float(cons.get("temperature", 0.3))
        self.cons_top_p = float(cons.get("top_p", 0.95))
        self.cons_top_k = int(cons.get("top_k", 64))
        self.rag_top_k = int(self.config.get("rag", {}).get("top_k", 5))

    def classify_one(self, item: str, n: Optional[int] = None) -> Stage1Result:
        """Stage 1: run classification n times at temp 0.3; report agreement % as consensus."""
        n = n or self.consensus_n
        labels: list[str] = []
        mitres: list[str] = []
        rationales: list[str] = []
        for _ in range(n):
            try:
                raw = self.engine.generate(
                    f"Classify this security log/finding line:\n{item}",
                    system=STAGE1_SCHEMA,
                    temperature=self.cons_temp,
                    top_p=self.cons_top_p,
                    top_k=self.cons_top_k,
                    role="fast",
                )
                obj = _extract_obj(raw)
                label = str(obj.get("label", "benign")).lower().strip()
                if label not in {"benign", "suspicious", "critical"}:
                    label = "benign"
                labels.append(label)
                mitres.append(str(obj.get("mitre", "unknown")))
                rationales.append(str(obj.get("rationale", "")))
            except Exception as e:
                labels.append("benign")
                mitres.append("unknown")
                rationales.append(f"classify error: {e}")

        counts = Counter(labels)
        winner, win_n = counts.most_common(1)[0]
        agreement = win_n / max(len(labels), 1)
        title = winner.capitalize()
        consensus = f"{title} ({win_n}/{len(labels)} agree)"
        # pick first matching winner's mitre/rationale
        idx = labels.index(winner)
        return Stage1Result(
            item=item,
            label=winner,
            mitre=mitres[idx],
            rationale=rationales[idx],
            votes=dict(counts),
            consensus=consensus,
            agreement=agreement,
        )

    def stage1(self, items: list[str]) -> list[Stage1Result]:
        return [self.classify_one(x) for x in items]

    def stage2_analyze(self, item: str, prompt_extra: str = "") -> Report:
        """Deep SecOps + RAG + harness on a single flagged item."""
        docs = []
        retrieved_ids: list[str] = []
        evidence = ""
        if self.rag is not None:
            docs = self.rag.retrieve_for_query(item, top_k=self.rag_top_k)
            retrieved_ids = [d.id for d in docs]
            evidence = self.rag.format_evidence(docs)

        user = (
            "Perform a grounded security analysis of the following.\n"
            f"ITEM:\n{item}\n\n"
        )
        if evidence:
            user += f"RETRIEVED EVIDENCE (cite only these doc ids in grounded_on):\n{evidence}\n\n"
        else:
            user += (
                "No retrieved evidence was available. If you cannot ground a CVE/KEV claim, "
                "use verdict insufficient_evidence.\n\n"
            )
        if prompt_extra:
            user += prompt_extra + "\n"
        return self.harness.run(
            user,
            retrieved_doc_ids=retrieved_ids if self.rag is not None else None,
            role="deep",
        )

    def run(self, items: list[str], prompt_extra: str = "") -> TriageResult:
        s1 = self.stage1(items)
        flagged = [
            i for i, r in enumerate(s1) if r.label in {"suspicious", "critical"}
        ]
        reports: dict[int, Report] = {}
        for i in flagged:
            reports[i] = self.stage2_analyze(s1[i].item, prompt_extra=prompt_extra)
        return TriageResult(stage1=s1, reports=reports, flagged_indices=flagged)
