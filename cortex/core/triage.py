"""Two-stage triage: E2B classify → SecOps deep analyze on flagged subset."""

from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from core.engine import Engine
from core.harness import Harness
from core.rag import RAG
from core.schema import Report

STAGE1_SCHEMA = """
You are a SOC stage-1 classifier. Return JSON only, no prose:
{"label":"benign"|"suspicious"|"critical","mitre":"Txxxx or unknown","rationale":"max 12 words"}
Be decisive and brief — this is a triage pass, not a report.
""".strip()

STAGE1_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["benign", "suspicious", "critical"]},
        "mitre": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "mitre", "rationale"],
}

STAGE1_BATCH_SCHEMA = """
You are a SOC stage-1 classifier. You will get numbered log lines.
Return JSON only, no prose: {"results":[{"i":<line number>,"label":"benign"|"suspicious"|"critical","mitre":"Txxxx or unknown","rationale":"max 12 words"}]}
Return exactly one entry per input line, in order. Be decisive and brief.
""".strip()

STAGE1_BATCH_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "label": {
                        "type": "string",
                        "enum": ["benign", "suspicious", "critical"],
                    },
                    "mitre": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["i", "label", "mitre", "rationale"],
            },
        }
    },
    "required": ["results"],
}

LABELS = ("benign", "suspicious", "critical")

# Fine-tuned security models answer with their own vocabulary ("malicious", "high").
# Mapping those to the nearest severity is safer than dropping the vote.
LABEL_SYNONYMS = {
    "malicious": "critical",
    "high": "critical",
    "severe": "critical",
    "attack": "critical",
    "compromised": "critical",
    "medium": "suspicious",
    "low": "suspicious",
    "anomalous": "suspicious",
    "unusual": "suspicious",
    "warning": "suspicious",
    "normal": "benign",
    "safe": "benign",
    "informational": "benign",
    "info": "benign",
}

# A parse failure must never masquerade as "benign" — that is a missed detection.
UNKNOWN_LABEL = "unknown"


def normalize_label(raw: str) -> str:
    label = (raw or "").strip().lower().strip(".\"'")
    if label in LABELS:
        return label
    if label in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[label]
    for key, mapped in LABEL_SYNONYMS.items():
        if key in label:
            return mapped
    for known in LABELS:
        if known in label:
            return known
    return UNKNOWN_LABEL


@dataclass
class Stage1Result:
    item: str
    label: str
    mitre: str
    rationale: str
    votes: dict[str, int] = field(default_factory=dict)
    consensus: str = ""  # e.g. "Critical (3/3 agree)"
    agreement: float = 0.0
    votes_spent: int = 0  # actual model calls used (early exit can beat consensus_n)


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
        triage_cfg = self.config.get("triage", {})
        self.consensus_n = int(triage_cfg.get("consensus_n", 3))
        self.early_exit = bool(triage_cfg.get("early_exit", True))
        self.max_workers = max(1, int(triage_cfg.get("max_workers", 3)))
        self.batch = bool(triage_cfg.get("batch", True))
        self.batch_min = max(2, int(triage_cfg.get("batch_min", 3)))
        self.batch_size = max(2, int(triage_cfg.get("batch_size", 8)))
        cons = self.config.get("sampling", {}).get("consensus", {})
        self.cons_temp = float(cons.get("temperature", 0.3))
        self.cons_top_p = float(cons.get("top_p", 0.95))
        self.cons_top_k = int(cons.get("top_k", 64))
        self.rag_top_k = int(self.config.get("rag", {}).get("top_k", 5))

    def _vote_once(self, item: str, seed_hint: int) -> tuple[str, str, str]:
        """One stage-1 vote. seed_hint keeps repeated votes from colliding in the cache."""
        try:
            raw = self.engine.generate(
                f"Classify this security log/finding line (vote {seed_hint}):\n{item}",
                system=STAGE1_SCHEMA,
                temperature=self.cons_temp,
                top_p=self.cons_top_p,
                top_k=self.cons_top_k,
                role="fast",
                format=STAGE1_FORMAT,
                label="stage1",
            )
            obj = _extract_obj(raw)
            label = normalize_label(str(obj.get("label", "")))
            return label, str(obj.get("mitre", "unknown")), str(obj.get("rationale", ""))
        except Exception as e:
            return UNKNOWN_LABEL, "unknown", f"classify error: {e}"

    def classify_one(self, item: str, n: Optional[int] = None) -> Stage1Result:
        """Stage 1 with early-exit consensus.

        Agreement counts, not a calibrated confidence. When the first two votes already
        agree the remaining votes cannot change the winner, so we stop paying for them.
        """
        n = n or self.consensus_n
        votes: list[tuple[str, str, str]] = []
        for i in range(n):
            votes.append(self._vote_once(item, i))
            if self.early_exit and self._decided([v[0] for v in votes], n):
                break
        return self._tally(item, votes)

    @staticmethod
    def _tally(item: str, votes: list[tuple[str, str, str]]) -> Stage1Result:
        labels = [v[0] for v in votes]
        counts = Counter(labels)
        winner, win_n = counts.most_common(1)[0]
        idx = labels.index(winner)  # keep the winning vote's MITRE/rationale
        return Stage1Result(
            item=item,
            label=winner,
            mitre=votes[idx][1],
            rationale=votes[idx][2],
            votes=dict(counts),
            consensus=f"{winner.capitalize()} ({win_n}/{len(labels)} agree)",
            agreement=win_n / max(len(labels), 1),
            votes_spent=len(labels),
        )

    @staticmethod
    def _decided(labels: list[str], n: int) -> bool:
        """True when the remaining votes cannot overturn the current leader."""
        if len(labels) >= n:
            return True
        counts = Counter(labels)
        leader = counts.most_common(1)[0][1]
        runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        return leader > runner_up + (n - len(labels))

    def _vote_batch(self, items: list[str], seed_hint: int) -> list[tuple[str, str, str]]:
        """One vote for every line in a single model call.

        Ollama serializes requests by default, so N separate calls cost N prompt evals
        and N generation ramps. Folding a whole round into one call is the difference
        between minutes and seconds on bulk triage. Falls back per-item if the batch
        response doesn't line up.
        """
        numbered = "\n".join(f"{i}. {it}" for i, it in enumerate(items))
        try:
            raw = self.engine.generate(
                f"Classify these {len(items)} security log lines (vote {seed_hint}):\n{numbered}",
                system=STAGE1_BATCH_SCHEMA,
                temperature=self.cons_temp,
                top_p=self.cons_top_p,
                top_k=self.cons_top_k,
                role="fast",
                format=STAGE1_BATCH_FORMAT,
                num_predict=max(64, 42 * len(items)),
                label="stage1-batch",
            )
            rows = _extract_obj(raw).get("results") or []
            by_index: dict[int, tuple[str, str, str]] = {}
            for row in rows:
                try:
                    idx = int(row.get("i"))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(items):
                    by_index[idx] = (
                        normalize_label(str(row.get("label", ""))),
                        str(row.get("mitre", "unknown")),
                        str(row.get("rationale", "")),
                    )
            if len(by_index) == len(items):
                return [by_index[i] for i in range(len(items))]
        except Exception:
            pass

        # Batch was unusable — fall back to one call per line rather than guessing.
        return [self._vote_once(it, seed_hint) for it in items]

    def _vote_round(self, pending: list[str], round_idx: int) -> list[tuple[str, str, str]]:
        if self.batch and len(pending) >= self.batch_min:
            out: list[tuple[str, str, str]] = []
            for start in range(0, len(pending), self.batch_size):
                out.extend(self._vote_batch(pending[start : start + self.batch_size], round_idx))
            return out
        if self.max_workers == 1 or len(pending) == 1:
            return [self._vote_once(x, round_idx) for x in pending]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pending))) as pool:
            return list(pool.map(lambda x: self._vote_once(x, round_idx), pending))

    def stage1(self, items: list[str], n: Optional[int] = None) -> list[Stage1Result]:
        """Classify items one vote round at a time, deduplicating repeated lines.

        Voting round-by-round across all lines (rather than all votes for line 1, then
        line 2, ...) lets each round go out as a single batched call while still letting
        a line drop out of later rounds as soon as its winner is locked.
        """
        n = n or self.consensus_n
        unique: list[str] = list(dict.fromkeys(items))
        if not unique:
            return []

        votes: dict[str, list[tuple[str, str, str]]] = {x: [] for x in unique}
        pending = list(unique)

        for round_idx in range(n):
            if not pending:
                break
            for item, vote in zip(pending, self._vote_round(pending, round_idx)):
                votes[item].append(vote)
            if self.early_exit:
                pending = [
                    x for x in pending if not self._decided([v[0] for v in votes[x]], n)
                ]

        tallied = {x: self._tally(x, votes[x]) for x in unique}
        return [tallied[it] for it in items]

    def stage2_analyze(self, item: str, prompt_extra: str = "") -> Report:
        """Deep SecOps + RAG + harness on a single flagged item."""
        docs = []
        retrieved_ids: list[str] = []
        evidence = ""
        if self.rag is not None:
            docs = self.rag.retrieve_for_query(item, top_k=self.rag_top_k)
            retrieved_ids = [d.id for d in docs]
            # Keep evidence short — long RAG dumps dominate prompt eval on M1.
            trimmed = []
            for d in docs:
                text = d.text if len(d.text) <= 420 else d.text[:420] + "…"
                trimmed.append(f"[{d.id}] {text}")
            evidence = "\n".join(trimmed)

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

    def run(
        self,
        items: list[str],
        prompt_extra: str = "",
        *,
        consensus_n: Optional[int] = None,
        max_deep: Optional[int] = None,
    ) -> TriageResult:
        s1 = self.stage1(items, n=consensus_n)
        # `unknown` means stage-1 could not be parsed; fail toward review, not toward benign.
        flagged = [
            i
            for i, r in enumerate(s1)
            if r.label in {"suspicious", "critical", UNKNOWN_LABEL}
        ]
        # Deep analysis is the expensive half: critical first, and cap how many run.
        budget = max_deep if max_deep is not None else int(
            self.config.get("triage", {}).get("max_deep", 4)
        )
        ordered = sorted(flagged, key=lambda i: 0 if s1[i].label == "critical" else 1)
        selected = ordered[:budget] if budget > 0 else ordered

        # Identical flagged lines share one deep report instead of re-running the 4B.
        reports: dict[int, Report] = {}
        by_item: dict[str, Report] = {}
        for i in selected:
            item = s1[i].item
            if item not in by_item:
                by_item[item] = self.stage2_analyze(item, prompt_extra=prompt_extra)
            reports[i] = by_item[item]
        return TriageResult(stage1=s1, reports=reports, flagged_indices=selected)
