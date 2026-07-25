"""Continuous autonomous defense loop — CORTEX's always-on OODA cycle.

One background worker runs, without an operator prompt:

    observe  → pull the next event off the live feed
    orient   → stage-1 Gemma classify with early-exit consensus (fast model)
    decide   → escalate only critical/suspicious; suppress repeats via memory
    act      → agent investigation with native tools (deep model)
    reflect  → the agent's own critique pass can reopen the investigation
    learn    → remember the signature so the same alert is never paid for twice

The loop owns its own thread and never blocks Gradio: the UI reads a snapshot.
Everything is local — no network at run time.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Optional

from core.agent import Agent, AgentResult
from core.live_feed import FeedEvent, LiveFeedState
from core.schema import Report, Verdict
from core.telemetry import Telemetry
from core.triage import Stage1Result, Triage

PHASES = ("observe", "orient", "decide", "act", "reflect", "learn")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def signature(text: str) -> str:
    """Collapse volatile tokens (IPs, ports, hex, counts) so near-duplicates dedupe."""
    s = text.lower()
    s = re.sub(r"\d+\.\d+\.\d+\.\d+", "<ip>", s)
    s = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", s)
    s = re.sub(r"\d+", "<n>", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()[:160]


@dataclass
class LoopEntry:
    """One line in the agent activity stream."""

    ts: str
    phase: str
    detail: str


@dataclass
class Incident:
    ts: str
    source: str
    alert: str
    label: str
    mitre: str
    consensus: str
    risk: int
    verdict: str
    summary: str
    fix: str = ""
    attack_path: str = ""
    cwe: str = ""
    cve: str = ""
    grounded_on: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    revisions: int = 0
    steps: int = 0
    ms: float = 0.0
    suppressed: int = 0  # later duplicates folded into this incident


@dataclass
class LoopStats:
    cycles: int = 0
    observed: int = 0
    escalated: int = 0
    suppressed: int = 0
    revisions: int = 0
    abstentions: int = 0
    triage_ms_total: float = 0.0
    triage_count: int = 0
    investigate_ms_total: float = 0.0
    investigate_count: int = 0

    @property
    def avg_triage_ms(self) -> float:
        return self.triage_ms_total / self.triage_count if self.triage_count else 0.0

    @property
    def avg_investigate_ms(self) -> float:
        return (
            self.investigate_ms_total / self.investigate_count
            if self.investigate_count
            else 0.0
        )


class AutonomyLoop:
    """Always-on defender. Start once; it keeps triaging and investigating on its own."""

    def __init__(
        self,
        triage: Triage,
        agent: Agent,
        feed: LiveFeedState,
        *,
        config: Optional[dict[str, Any]] = None,
        telemetry: Optional[Telemetry] = None,
    ):
        self.triage = triage
        self.agent = agent
        self.feed = feed
        cfg = (config or {}).get("autonomy", {})
        self.interval = float(cfg.get("interval_s", 2.0))
        self.consensus_n = int(cfg.get("consensus_n", 2))
        self.escalate_labels = set(cfg.get("escalate_labels", ["critical", "suspicious"]))
        self.investigate_labels = set(cfg.get("investigate_labels", ["critical"]))
        self.max_incidents = int(cfg.get("max_incidents", 30))
        self.telemetry = telemetry

        self.phase: str = "idle"
        self.current: str = ""
        self.stats = LoopStats()
        self.incidents: Deque[Incident] = deque(maxlen=self.max_incidents)
        self.activity: Deque[LoopEntry] = deque(maxlen=120)
        self.memory: dict[str, int] = {}  # signature -> incident age counter
        self._by_signature: dict[str, Incident] = {}

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_error: str = ""

    # ---------- lifecycle ----------

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def start(self) -> str:
        with self._lock:
            if self.running:
                return "Autonomy loop already running."
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="cortex-autonomy", daemon=True
            )
            self._thread.start()
            self._log("observe", "loop started — autonomous defense engaged")
            return "Autonomy loop started."

    def stop(self) -> str:
        with self._lock:
            if not self.running:
                return "Autonomy loop is not running."
            self._stop.set()
            self._log("learn", "loop stopped by operator")
            self.phase = "idle"
            return "Autonomy loop stopped."

    def reset(self) -> str:
        with self._lock:
            self.incidents.clear()
            self.activity.clear()
            self.memory.clear()
            self._by_signature.clear()
            self.stats = LoopStats()
            self.phase = "idle" if not self.running else self.phase
            self.current = ""
            return "Autonomy state cleared."

    # ---------- the loop ----------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.cycle_once()
            except Exception as e:  # a bad cycle must never kill the defender
                self._last_error = str(e)
                self._log("learn", f"cycle error handled: {e}")
            self._stop.wait(self.interval)

    def cycle_once(self) -> Optional[Incident]:
        """One full OODA pass. Public so tests and the UI can single-step the loop."""
        event = self._observe()
        if event is None:
            return None

        sig = signature(event.message)
        if self._suppress(sig):
            return None

        s1, triage_ms = self._orient(event)
        incident = self._decide_and_act(event, s1, sig, triage_ms)
        with self._lock:
            self.stats.cycles += 1
            self.phase = "learn"
            self.current = ""
        return incident

    def _observe(self) -> Optional[FeedEvent]:
        with self._lock:
            self.phase = "observe"
        event = self.feed.push_random()
        with self._lock:
            self.stats.observed += 1
            self.current = event.message[:110]
        self._log("observe", f"{event.source}: {event.message[:88]}")
        return event

    def _suppress(self, sig: str) -> bool:
        """Memory check — an alert class already investigated is folded, not re-paid for."""
        with self._lock:
            known = self._by_signature.get(sig)
            if known is None:
                return False
            known.suppressed += 1
            self.memory[sig] = self.memory.get(sig, 1) + 1
            self.stats.suppressed += 1
            self.stats.cycles += 1
            self.phase = "learn"
        self._log("learn", f"known pattern — folded into existing incident (x{known.suppressed + 1})")
        return True

    def _orient(self, event: FeedEvent) -> tuple[Stage1Result, float]:
        with self._lock:
            self.phase = "orient"
        t0 = time.perf_counter()
        s1 = self.triage.classify_one(event.message, n=self.consensus_n)
        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.stats.triage_ms_total += ms
            self.stats.triage_count += 1
        self._log(
            "orient",
            f"stage-1 {s1.consensus} · {s1.mitre} · {s1.votes_spent} vote(s) · {ms:.0f}ms",
        )
        return s1, ms

    def _decide_and_act(
        self, event: FeedEvent, s1: Stage1Result, sig: str, triage_ms: float
    ) -> Optional[Incident]:
        with self._lock:
            self.phase = "decide"
        if s1.label not in self.escalate_labels:
            self._log("decide", f"benign — no deep spend ({s1.label})")
            with self._lock:
                self.memory[sig] = self.memory.get(sig, 0) + 1
            return None

        if s1.label not in self.investigate_labels:
            # Escalated but not worth the 4B: record a triage-only incident.
            incident = Incident(
                ts=_now(),
                source=event.source,
                alert=event.message,
                label=s1.label,
                mitre=s1.mitre,
                consensus=s1.consensus,
                risk=45,
                verdict="triaged",
                summary=s1.rationale or "Stage-1 flagged; queued below critical threshold.",
                ms=triage_ms,
            )
            self._record(incident, sig)
            self._log("decide", "suspicious — logged without deep analysis")
            return incident

        self._log("act", "critical — escalating to agent investigation")
        with self._lock:
            self.phase = "act"
            self.stats.escalated += 1

        t0 = time.perf_counter()
        result = self.agent.investigate(
            event.message,
            context=f"source={event.source} stage1={s1.consensus} mitre={s1.mitre}",
            on_event=lambda kind, text: self._log(
                "reflect" if kind == "critique" else "act", f"{kind}: {text}"
            ),
        )
        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.phase = "reflect"
            self.stats.investigate_ms_total += ms
            self.stats.investigate_count += 1
            self.stats.revisions += result.revisions
            if result.report.abstained:
                self.stats.abstentions += 1

        incident = self._to_incident(event, s1, result, ms)
        self._record(incident, sig)
        self._log(
            "reflect",
            f"report risk {incident.risk}/100 · {incident.verdict} · "
            f"{len(result.trace)} tool call(s) · {result.revisions} revision(s) · {ms:.0f}ms",
        )
        return incident

    def _to_incident(
        self, event: FeedEvent, s1: Stage1Result, result: AgentResult, ms: float
    ) -> Incident:
        report: Report = result.report
        primary = None
        for f in report.findings:
            if f.verdict == Verdict.VULNERABLE:
                primary = f
                break
        if primary is None and report.findings:
            primary = report.findings[0]

        return Incident(
            ts=_now(),
            source=event.source,
            alert=event.message,
            label=s1.label,
            mitre=s1.mitre,
            consensus=s1.consensus,
            risk=report.overall_risk,
            verdict=(primary.verdict.value if primary else "insufficient_evidence"),
            summary=(primary.explanation if primary else "No finding produced."),
            fix=(primary.fix or "" if primary else ""),
            attack_path=(primary.attack_path or "" if primary else ""),
            cwe=(primary.cwe_id or "" if primary else ""),
            cve=(primary.cve_id or "" if primary else ""),
            grounded_on=list(primary.grounded_on) if primary else [],
            tools=result.tool_names,
            critiques=list(result.critiques),
            revisions=result.revisions,
            steps=result.steps,
            ms=ms,
        )

    def _record(self, incident: Incident, sig: str) -> None:
        with self._lock:
            self.incidents.appendleft(incident)
            self._by_signature[sig] = incident
            self.memory[sig] = self.memory.get(sig, 0) + 1

    def _log(self, phase: str, detail: str) -> None:
        with self._lock:
            self.activity.appendleft(LoopEntry(ts=_now(), phase=phase, detail=detail))

    # ---------- read side ----------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "phase": self.phase,
                "current": self.current,
                "cycles": self.stats.cycles,
                "observed": self.stats.observed,
                "escalated": self.stats.escalated,
                "suppressed": self.stats.suppressed,
                "revisions": self.stats.revisions,
                "abstentions": self.stats.abstentions,
                "avg_triage_ms": self.stats.avg_triage_ms,
                "avg_investigate_ms": self.stats.avg_investigate_ms,
                "known_patterns": len(self.memory),
                "incidents": list(self.incidents),
                "activity": list(self.activity)[:40],
                "last_error": self._last_error,
            }

    def incident_rows(self) -> list[list[Any]]:
        with self._lock:
            return [
                [
                    i.ts,
                    i.source,
                    i.label,
                    i.mitre,
                    f"{i.risk}/100",
                    i.verdict,
                    i.cwe or i.cve or "—",
                    f"{i.ms:.0f}ms",
                    i.alert[:90],
                ]
                for i in self.incidents
            ]
