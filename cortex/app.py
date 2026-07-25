"""CORTEX Gradio UI — autonomous SOC console + on-demand analysis modes."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any, Optional

# Force embedding loads offline before any HF/chroma imports.
from core.offline import enable_hf_offline, resolve_embedding_model

enable_hf_offline()
resolve_embedding_model("data/models/all-MiniLM-L6-v2")

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.agent import Agent, format_trace
from core.autonomy import PHASES, AutonomyLoop, Incident, LoopEntry
from core.engine import Engine, load_config
from core.harness import Harness, abstain_report, apply_grounding_rule
from core.live_feed import FEED, LiveFeedState
from core.rag import RAG
from core.schema import Finding, Report, Verdict
from core.telemetry import TELEMETRY
from core.triage import Triage
from core.vision import VisionAnalyzer, VisionResult

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
SAMPLE_LOGS = ROOT / "data" / "sample_logs.csv"
VULN_CODE = ROOT / "data" / "vulnerable_code.py"
DEMO_DIAGRAM = ROOT / "data" / "demo_diagram.png"
THEME_CSS = (ROOT / "assets" / "theme.css").read_text()


def _read_prompt(name: str) -> str:
    path = PROMPTS / name
    return path.read_text() if path.exists() else ""


def _report_to_md(report: Report) -> str:
    lines = [
        f"**Overall risk:** {report.overall_risk}/100",
        f"**Abstained:** {report.abstained}",
        "",
    ]
    for i, f in enumerate(report.findings, 1):
        lines.append(f"### Finding {i}: `{f.verdict.value}`")
        if f.cve_id:
            lines.append(f"- CVE: `{f.cve_id}`")
        if f.cwe_id:
            lines.append(f"- CWE: `{f.cwe_id}`")
        if f.severity:
            lines.append(f"- Severity: {f.severity}")
        if f.line_hint:
            lines.append(f"- Line hint: {f.line_hint}")
        lines.append(f"- Explanation: {f.explanation}")
        if f.attack_path:
            lines.append(f"- Attack path: {f.attack_path}")
        if f.fix:
            lines.append(f"- Fix: {f.fix}")
        cites = ", ".join(f"`{c}`" for c in f.grounded_on) or "_none_"
        lines.append(f"- Grounded on: {cites}")
        lines.append("")
    return "\n".join(lines)


class AppState:
    def __init__(self) -> None:
        self.config = load_config()
        self.engine = Engine(self.config)
        self.harness = Harness(self.engine, self.config)
        try:
            self.rag: Optional[RAG] = RAG(
                persist_dir=ROOT / self.config.get("rag", {}).get("persist_dir", "data/chroma"),
                collection=self.config.get("rag", {}).get("collection", "cortex_cve"),
                embedding_model=self.config.get("rag", {}).get(
                    "embedding_model", "data/models/all-MiniLM-L6-v2"
                ),
            )
        except Exception as e:
            print(f"RAG unavailable: {e}")
            self.rag = None
        self.triage = Triage(self.engine, self.harness, self.rag, self.config)
        self.agent = Agent(self.engine, self.rag, code_root=ROOT, config=self.config)
        self.vision = VisionAnalyzer(self.engine, self.rag, config=self.config)
        self.feed: LiveFeedState = FEED
        self.autonomy = AutonomyLoop(
            self.triage, self.agent, self.feed, config=self.config, telemetry=TELEMETRY
        )
        if self.config.get("performance", {}).get("prewarm_on_start", False):
            self.engine.prewarm()


STATE: Optional[AppState] = None


def get_state() -> AppState:
    global STATE
    if STATE is None:
        STATE = AppState()
    return STATE


def load_sample_logs() -> str:
    if not SAMPLE_LOGS.exists():
        return ""
    rows = []
    with open(SAMPLE_LOGS, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row["message"])
    return "\n".join(rows)


# ---------- shared renderers ----------

def render_hero(autonomous: bool = False) -> str:
    live = (
        '<span class="cortex-pill live"><span class="cortex-live-dot"></span>'
        "Autonomous defense engaged</span>"
        if autonomous
        else '<span class="cortex-pill">Standby — operator driven</span>'
    )
    return f"""
    <div class="cortex-hero">
      <div class="cortex-hero-top">
        <div>
          <h1 class="cortex-brand">CORTEX</h1>
          <p class="cortex-tagline">
            Grounded, offline-first blue-team assistant on Gemma 4. It triages continuously
            on its own, investigates with native tool calling, critiques its own reports —
            and abstains instead of inventing a CVE it never retrieved.
          </p>
        </div>
      </div>
      <div class="cortex-pills">
        {live}
        <span class="cortex-pill">E2B triage → SecOps deep</span>
        <span class="cortex-pill">Post-cutoff CVE/KEV grounding</span>
        <span class="cortex-pill warn">No network at run time</span>
      </div>
    </div>
    """


def render_telemetry() -> str:
    t = TELEMETRY.snapshot()
    return f"""
    <div class="cortex-stats">
      <div class="cortex-stat total">
        <div class="k">Model calls</div><div class="v">{t['total_calls']}</div>
        <div class="sub">this session</div>
      </div>
      <div class="cortex-stat benign">
        <div class="k">Cache hits</div><div class="v">{t['cache_hit_rate']:.0%}</div>
        <div class="sub">{t['cache_hits']} reused · ~{t['saved_s']:.0f}s saved</div>
      </div>
      <div class="cortex-stat accent">
        <div class="k">Fast stage</div><div class="v">{t['fast_avg_ms']/1000:.1f}s</div>
        <div class="sub">E2B avg per call</div>
      </div>
      <div class="cortex-stat suspicious">
        <div class="k">Deep stage</div><div class="v">{t['deep_avg_ms']/1000:.1f}s</div>
        <div class="sub">SecOps avg · p95 {t['p95_ms']/1000:.1f}s</div>
      </div>
    </div>
    """


def render_phase_rail(snap: dict[str, Any]) -> str:
    active = snap.get("phase", "idle")
    try:
        active_idx = PHASES.index(active)
    except ValueError:
        active_idx = -1
    chips = []
    for i, phase in enumerate(PHASES):
        if i == active_idx:
            cls = "active"
        elif active_idx >= 0 and i < active_idx:
            cls = "done"
        else:
            cls = ""
        chips.append(f'<span class="cortex-phase {cls}">{phase}</span>')
        if i < len(PHASES) - 1:
            chips.append('<span class="cortex-phase-arrow">›</span>')
    current = html.escape(snap.get("current") or ("idle — press Engage" if not snap["running"] else "…"))
    return (
        '<div class="cortex-loop">'
        + "".join(chips)
        + f'<span class="cortex-loop-current">{current}</span>'
        + "</div>"
    )


def render_loop_stats(snap: dict[str, Any]) -> str:
    return f"""
    <div class="cortex-stats">
      <div class="cortex-stat total">
        <div class="k">Cycles</div><div class="v">{snap['cycles']}</div>
        <div class="sub">{snap['observed']} observed</div>
      </div>
      <div class="cortex-stat critical">
        <div class="k">Escalated</div><div class="v">{snap['escalated']}</div>
        <div class="sub">agent investigations</div>
      </div>
      <div class="cortex-stat benign">
        <div class="k">Suppressed</div><div class="v">{snap['suppressed']}</div>
        <div class="sub">{snap['known_patterns']} known patterns</div>
      </div>
      <div class="cortex-stat accent">
        <div class="k">Self-revisions</div><div class="v">{snap['revisions']}</div>
        <div class="sub">{snap['abstentions']} honest abstentions</div>
      </div>
      <div class="cortex-stat suspicious">
        <div class="k">Triage</div><div class="v">{snap['avg_triage_ms']/1000:.1f}s</div>
        <div class="sub">deep {snap['avg_investigate_ms']/1000:.1f}s</div>
      </div>
    </div>
    """


def render_incidents(incidents: list[Incident]) -> str:
    if not incidents:
        return (
            '<div class="cortex-empty">No incidents yet — engage the loop and CORTEX will '
            "triage the stream on its own.</div>"
        )
    cards = []
    # Cards are the scannable view; the full history stays in the incident table.
    for inc in incidents[:12]:
        chips = []
        if inc.cwe:
            chips.append(f'<span class="cortex-chip">{html.escape(inc.cwe)}</span>')
        if inc.cve:
            chips.append(f'<span class="cortex-chip">{html.escape(inc.cve)}</span>')
        for tool in dict.fromkeys(inc.tools):
            chips.append(f'<span class="cortex-chip">{html.escape(tool)}()</span>')
        for doc in inc.grounded_on[:3]:
            chips.append(f'<span class="cortex-chip cite">{html.escape(doc)}</span>')
        if inc.revisions:
            chips.append(
                f'<span class="cortex-chip warn">self-revised ×{inc.revisions}</span>'
            )
        if inc.suppressed:
            chips.append(
                f'<span class="cortex-chip">+{inc.suppressed} repeat(s) folded</span>'
            )

        detail = []
        if inc.attack_path:
            detail.append(
                f'<div class="cortex-card-line"><b>Attack path</b> {html.escape(inc.attack_path)}</div>'
            )
        if inc.fix:
            detail.append(
                f'<div class="cortex-card-line"><b>Fix</b> {html.escape(inc.fix)}</div>'
            )
        if inc.critiques:
            detail.append(
                '<div class="cortex-card-line"><b>Review</b> '
                + html.escape(inc.critiques[0])
                + "</div>"
            )

        cards.append(
            f"""
        <div class="cortex-card {inc.label}">
          <div class="cortex-card-head">
            <span class="cortex-badge {inc.label}">{inc.label}</span>
            <span class="cortex-badge {inc.verdict}">{inc.verdict.replace('_', ' ')}</span>
            <span class="cortex-chip">{html.escape(inc.mitre)}</span>
            <span class="cortex-card-ts">{inc.ts} · {inc.ms/1000:.1f}s</span>
          </div>
          <div class="cortex-card-alert">{html.escape(inc.alert[:190])}</div>
          <div class="cortex-risk"><span style="width:{max(min(inc.risk, 100), 2)}%"></span></div>
          <div class="cortex-card-line"><b>Risk {inc.risk}/100</b> · {html.escape(inc.consensus)}</div>
          <div class="cortex-card-line">{html.escape(inc.summary[:260])}</div>
          {''.join(detail)}
          <div class="cortex-chips">{''.join(chips)}</div>
        </div>
        """
        )
    return '<div class="cortex-incidents">' + "".join(cards) + "</div>"


def render_activity(entries: list[LoopEntry]) -> str:
    def row(ts: str, phase: str, detail: str) -> str:
        return (
            f'<div class="cortex-act-row {phase}">'
            f'<div class="t">{html.escape(ts)}</div>'
            f'<div class="p">{html.escape(phase)}</div>'
            f"<div>{html.escape(detail)}</div>"
            f"</div>"
        )

    if not entries:
        rows = row("--:--:--", "learn", "waiting for engage")
    else:
        rows = "".join(row(e.ts, e.phase, e.detail) for e in entries)
    return f"""
    <div class="cortex-activity">
      <div class="cortex-activity-head">Agent activity — observe · orient · decide · act · reflect · learn</div>
      <div class="cortex-activity-body">{rows}</div>
    </div>
    """


def render_stats(feed: LiveFeedState) -> str:
    window = feed.counts()
    total = feed.critical_total + feed.suspicious_total + feed.benign_total
    return f"""
    <div class="cortex-stats">
      <div class="cortex-stat total">
        <div class="k">Events seen</div><div class="v">{total}</div>
      </div>
      <div class="cortex-stat critical">
        <div class="k">Critical</div><div class="v">{feed.critical_total}</div>
      </div>
      <div class="cortex-stat suspicious">
        <div class="k">Suspicious</div><div class="v">{feed.suspicious_total}</div>
      </div>
      <div class="cortex-stat benign">
        <div class="k">Window mix</div>
        <div class="v" style="font-size:1.1rem;margin-top:0.35rem">
          {window.get('critical',0)}C · {window.get('suspicious',0)}S · {window.get('benign',0)}B
        </div>
      </div>
    </div>
    """


def render_feed_html(feed: LiveFeedState) -> str:
    rows = []
    for ev in list(feed.events)[:28]:
        tip = (
            f"<div class='cortex-tip'>"
            f"<strong>Consensus</strong> {html.escape(ev.consensus)}<br/>"
            f"<strong>MITRE</strong> {html.escape(ev.mitre)} · "
            f"<strong>Source</strong> {html.escape(ev.source)}<br/>"
            f"<strong>Heuristic score</strong> {ev.score:.0%} "
            f"(stream classifier — not a confidence interval)<br/>"
            f"<span style='opacity:.85'>{html.escape(ev.message)}</span>"
            f"</div>"
        )
        rows.append(
            f"<div class='cortex-row'>"
            f"<div class='ts'>{html.escape(ev.ts)}</div>"
            f"<div class='src'>{html.escape(ev.source)}</div>"
            f"<div><span class='cortex-badge {ev.label}'>{ev.label}</span></div>"
            f"<div class='cortex-msg'>{html.escape(ev.message[:160])}</div>"
            f"{tip}"
            f"</div>"
        )
    body = "\n".join(rows) if rows else "<div class='cortex-row'>Waiting for events…</div>"
    return f"""
    <div class="cortex-feed">
      <div class="cortex-feed-head">
        <span>Threat feed · real-time simulation</span>
        <span>tick {feed.tick} · hover a row for detail</span>
      </div>
      <div class="cortex-feed-body">{body}</div>
    </div>
    """


def build_charts(feed: LiveFeedState):
    counts = feed.counts()
    labels = ["critical", "suspicious", "benign"]
    values = [counts.get(l, 0) for l in labels]
    colors = ["#b42318", "#c45c26", "#2f6b4f"]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=("Window severity", "Cumulative detections"),
    )
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values if sum(values) else [1, 1, 1],
            marker=dict(colors=colors, line=dict(color="#f3f6f4", width=2)),
            hole=0.55,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    hist = feed.history or [{"t": 0, "critical": 0, "suspicious": 0, "benign": 0}]
    xs = [h["t"] for h in hist]
    for name, color, dash, width in [
        ("critical", "#b42318", None, 2.5),
        ("suspicious", "#c45c26", None, 2.2),
        ("benign", "#2f6b4f", "dot", 1.8),
    ]:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[h[name] for h in hist],
                name=name,
                line=dict(color=color, width=width, dash=dash),
                fill="tozeroy" if name == "critical" else None,
                fillcolor="rgba(180,35,24,0.12)" if name == "critical" else None,
                hovertemplate="t=%{x}<br>" + name + "=%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=42, b=10),
        height=290,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(size=12, color="#14212b"),
        legend=dict(orientation="h", y=1.18, x=0.55),
    )
    fig.update_xaxes(title_text="tick", gridcolor="rgba(20,33,43,0.08)")
    fig.update_yaxes(title_text="count", gridcolor="rgba(20,33,43,0.08)")
    return fig


# ---------- autonomy handlers ----------

def autonomy_bundle():
    """Everything the autonomy tab shows, from one snapshot."""
    st = get_state()
    snap = st.autonomy.snapshot()
    return (
        render_hero(snap["running"]),
        render_phase_rail(snap),
        render_loop_stats(snap),
        render_incidents(snap["incidents"]),
        render_activity(snap["activity"]),
        render_telemetry(),
        st.autonomy.incident_rows(),
    )


def autonomy_start():
    get_state().autonomy.start()
    return autonomy_bundle()


def autonomy_stop():
    get_state().autonomy.stop()
    return autonomy_bundle()


def autonomy_reset():
    get_state().autonomy.reset()
    return autonomy_bundle()


def autonomy_step():
    """Single OODA pass — useful when demoing without leaving the loop running."""
    get_state().autonomy.cycle_once()
    return autonomy_bundle()


# ---------- analysis modes ----------

def run_log_triage(
    log_text: str, consensus_n: Optional[int] = None
) -> tuple[list[list[Any]], str]:
    st = get_state()
    items = [ln.strip() for ln in (log_text or "").splitlines() if ln.strip()]
    if not items:
        return [], "No log lines provided."
    extra = _read_prompt("log_triage.md")
    result = st.triage.run(items, prompt_extra=extra, consensus_n=consensus_n)
    table = []
    for i, s1 in enumerate(result.stage1):
        table.append(
            [
                s1.item[:120],
                s1.label,
                s1.mitre,
                s1.consensus,
                f"{s1.agreement:.0%}",
                "yes" if i in result.flagged_indices else "no",
            ]
        )
    reports_md = []
    for i in result.flagged_indices:
        reports_md.append(
            f"## Deep analysis — line {i+1}\n" + _report_to_md(result.reports[i])
        )
    return table, "\n\n".join(reports_md) if reports_md else "_No suspicious/critical lines flagged._"


def run_code_audit(file_obj, code_text: str) -> str:
    st = get_state()
    source = (code_text or "").strip()
    if file_obj:
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
        if path:
            with open(path) as f:
                source = f.read()
    if not source:
        if VULN_CODE.exists():
            source = VULN_CODE.read_text()
        else:
            return "No code provided."
    extra = _read_prompt("code_audit.md")
    ids: list[str] = []
    evidence = ""
    if st.rag is not None:
        docs = st.rag.retrieve_for_query(
            source[:2000], top_k=st.config.get("rag", {}).get("top_k", 5)
        )
        ids = [d.id for d in docs]
        evidence = st.rag.format_evidence(docs)
    user = f"{extra}\n\nSOURCE:\n```\n{source}\n```\n"
    if evidence:
        user += f"\nRETRIEVED EVIDENCE:\n{evidence}\n"
    report = st.harness.run(
        user, retrieved_doc_ids=ids if st.rag is not None else None, role="deep"
    )
    return _report_to_md(report)


def run_cve_query(query: str) -> str:
    st = get_state()
    q = (query or "").strip()
    if not q:
        return "Enter a CVE id or question."
    extra = _read_prompt("cve_triage.md")
    ids: list[str] = []
    evidence = ""
    if st.rag is not None:
        docs = st.rag.retrieve_for_query(q, top_k=st.config.get("rag", {}).get("top_k", 5))
        ids = [d.id for d in docs]
        evidence = st.rag.format_evidence(docs)
    user = f"{extra}\n\nQUERY:\n{q}\n"
    if evidence:
        user += f"\nRETRIEVED EVIDENCE:\n{evidence}\n"
    else:
        user += "\nNo evidence retrieved. Abstain if you cannot ground claims.\n"
    report = st.harness.run(
        user, retrieved_doc_ids=ids if st.rag is not None else None, role="deep"
    )
    return _report_to_md(report)


def _format_agent_result(result) -> tuple[str, str]:
    lines = [
        f"**Steps:** {result.steps}  ",
        f"**Docs retrieved:** {', '.join(f'`{i}`' for i in result.retrieved_doc_ids) or '_none_'}  ",
        f"**Self-revisions:** {result.revisions}",
    ]
    if result.critiques:
        lines.append("")
        lines.append("**Review that reopened the investigation:**")
        for c in result.critiques:
            lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines) + _report_to_md(result.report), format_trace(result.trace)


def run_agent(goal: str = "") -> tuple[str, str]:
    """Run agent; empty goal falls back to configured autonomous defend goal."""
    st = get_state()
    result = st.agent.run(goal)
    return _format_agent_result(result)


def run_autonomous_defend(target: str = "") -> tuple[str, str]:
    """One-click autonomous defense — no free-text goal required."""
    st = get_state()
    path = (target or "").strip() or None
    result = st.agent.defend(target=path)
    header = f"_Autonomous defend · target `{path or st.agent.default_target}`_\n\n"
    md, trace = _format_agent_result(result)
    return header + md, trace


SEVERITY_CLASS = {
    "Critical": "critical",
    "High": "critical",
    "Medium": "suspicious",
    "Low": "benign",
}


def render_diagram_result(result: VisionResult) -> str:
    report = result.report
    head = (
        '<div class="cortex-vision-head">'
        f'<span class="cortex-pill live"><span class="cortex-live-dot"></span>'
        f"vision · {html.escape(result.model or 'no model')}</span>"
        f'<span class="cortex-pill">{result.ms/1000:.1f}s</span>'
        f'<span class="cortex-pill">risk {report.overall_risk}/100</span>'
        f'<span class="cortex-pill">{len(report.findings)} finding(s)</span>'
        "</div>"
    )

    if not report.findings:
        return (
            head
            + '<div class="cortex-empty">'
            + html.escape(report.notes or "No architectural weaknesses returned.")
            + "</div>"
        )

    cards = []
    for f in report.findings:
        cls = SEVERITY_CLASS.get(f.severity, "suspicious")
        chips = [
            f'<span class="cortex-chip mitre">{html.escape(f.mitre_technique)} · {html.escape(f.mitre_tactic)}</span>',
            f'<span class="cortex-chip nist">NIST {html.escape(f.nist_control)}</span>',
            f'<span class="cortex-chip">{html.escape(f.exposure)}</span>',
        ]
        if f.cve_id:
            chips.append(f'<span class="cortex-chip cve">{html.escape(f.cve_id)}</span>')
        for doc in f.grounded_on[:2]:
            chips.append(f'<span class="cortex-chip cite">{html.escape(doc)}</span>')

        detail = []
        if f.nist_rationale:
            detail.append(
                f'<div class="cortex-card-line"><b>Control gap</b> {html.escape(f.nist_rationale)}</div>'
            )
        if f.attack_path:
            detail.append(
                f'<div class="cortex-card-line"><b>Attack path</b> {html.escape(f.attack_path)}</div>'
            )
        if f.fix:
            detail.append(
                f'<div class="cortex-card-line"><b>Fix</b> {html.escape(f.fix)}</div>'
            )

        cards.append(
            f"""
        <div class="cortex-card {cls} cortex-pop">
          <div class="cortex-card-head">
            <span class="cortex-badge {cls}">{html.escape(f.severity)}</span>
            <span class="cortex-card-title">{html.escape(f.component)}</span>
          </div>
          <div class="cortex-card-line">{html.escape(f.explanation)}</div>
          {''.join(detail)}
          <div class="cortex-chips">{''.join(chips)}</div>
        </div>
        """
        )

    blocks = [head, '<div class="cortex-incidents">' + "".join(cards) + "</div>"]

    if result.advisories:
        rows = "".join(
            f'<div class="cortex-adv-row">'
            f'<span class="cortex-chip cve">{html.escape(a.cve_id or "advisory")}</span>'
            f'<span class="cortex-chip cite">{html.escape(a.doc_id)}</span>'
            f'<span class="cortex-adv-comp">{html.escape(a.component)}</span>'
            f'<span class="cortex-adv-text">{html.escape(a.text[:180])}</span>'
            f"</div>"
            for a in result.advisories
        )
        blocks.append(
            '<div class="cortex-advisories"><div class="cortex-activity-head">'
            "Advisories matched from the local post-cutoff index (not model recall)"
            f"</div>{rows}</div>"
        )

    if result.stripped_cves:
        blocks.append(
            '<div class="cortex-strip-note">Stripped '
            + ", ".join(f"<code>{html.escape(c)}</code>" for c in result.stripped_cves)
            + " — named without retrieved evidence.</div>"
        )

    if report.notes:
        blocks.append(
            f'<div class="cortex-card-line" style="margin-top:.6rem"><b>Notes</b> {html.escape(report.notes)}</div>'
        )

    return "".join(blocks)


def run_diagram_analysis(image_path) -> str:
    """Multimodal architecture review — model reads topology, index supplies CVEs."""
    st = get_state()
    path = image_path if isinstance(image_path, str) else getattr(image_path, "name", None)
    if not path:
        return '<div class="cortex-empty">Upload an infrastructure diagram to analyze.</div>'
    return render_diagram_result(st.vision.analyze(path))


def run_full_demo() -> tuple[str, list[list[Any]], str, str, str, str, str]:
    st = get_state()
    demo_cfg = st.config.get("demo", {})
    narration: list[str] = [
        "# CORTEX live demo",
        f"- Fast model: `{st.engine.fast_model}`",
        f"- Deep model: `{st.engine.deep_model}`",
        "- Pipeline: E2B triage → SecOps deep analysis; agent uses **native function calling**.",
        "- Reasoning tokens disabled, schema-constrained decoding, cached repeats.",
        "",
    ]
    all_logs = [ln for ln in load_sample_logs().splitlines() if ln.strip()]
    limit = int(demo_cfg.get("log_limit", 4))
    demo_logs = all_logs[:limit]
    demo_consensus = int(demo_cfg.get("consensus_n", 1))
    narration.append(
        f"## 1. Two-stage triage ({len(demo_logs)} log lines, consensus N={demo_consensus})"
    )
    table, deep_md = run_log_triage("\n".join(demo_logs), consensus_n=demo_consensus)
    flagged = sum(1 for row in table if row[5] == "yes")
    narration.append(
        f"Stage-1 consensus (agreement counts, **not** a CI). Flagged: **{flagged}**."
    )
    cve_q = demo_cfg.get("cve_query", "CVE-2025-24813")
    narration.append(f"\n## 2. Grounded CVE — `{cve_q}`")
    cve_md = run_cve_query(cve_q)
    fake_q = demo_cfg.get("fake_cve_query", "CVE-2099-00001")
    narration.append(f"\n## 3. Abstention — `{fake_q}`")
    fake_report = apply_grounding_rule(
        Report(
            findings=[
                Finding(
                    verdict=Verdict.VULNERABLE,
                    explanation=f"Claims {fake_q} without evidence",
                    cve_id="CVE-2099-00001",
                    grounded_on=["invented-doc"],
                )
            ],
            overall_risk=99,
        ),
        allowed_doc_ids=set(),
    )
    if not fake_report.findings or fake_report.findings[0].verdict != Verdict.INSUFFICIENT_EVIDENCE:
        fake_report = abstain_report("No local evidence for fabricated CVE.")
    abstain_md = _report_to_md(fake_report)
    narration.append("Post-hoc validator forced `insufficient_evidence` and stripped `cve_id`.")
    narration.append("\n## 4. Autonomous agent defend (native tools + self-critique)")
    agent_md, agent_trace = run_autonomous_defend()
    narration.append("Model chose tool order — not a fixed pipeline.")
    narration.append("\n---\n**Demo complete.** The Autonomy tab runs this without prompting.")
    return "\n".join(narration), table, deep_md, cve_md, abstain_md, agent_md, agent_trace


def build_ui() -> gr.Blocks:
    feed = FEED
    cfg = load_config()
    agent_cfg = cfg.get("agent") or {}
    default_target = str(agent_cfg.get("default_target", "data/vulnerable_code.py"))
    auto_cfg = cfg.get("autonomy") or {}
    interval = float(auto_cfg.get("interval_s", 2.0))

    with gr.Blocks(title="CORTEX", analytics_enabled=False) as demo:
        hero = gr.HTML(render_hero(False))

        with gr.Tabs():
            # ---------- Autonomy (headline) ----------
            with gr.Tab("Autonomy"):
                gr.Markdown(
                    "**Continuous autonomous defense.** No prompt, no goal box. Press engage and "
                    "CORTEX observes the stream, triages with the fast model, escalates only what "
                    "earns it, investigates with native tools, then critiques its own report and "
                    "revises it. Repeats are folded into known patterns instead of re-analyzed."
                )
                with gr.Row():
                    engage_btn = gr.Button("Engage autonomous defense", variant="primary")
                    stop_btn = gr.Button("Stand down", variant="stop")
                    step_btn = gr.Button("Single cycle")
                    reset_btn = gr.Button("Clear incidents")

                idle_snap: dict[str, Any] = {
                    "phase": "idle", "current": "", "running": False,
                    "cycles": 0, "observed": 0, "escalated": 0, "suppressed": 0,
                    "revisions": 0, "abstentions": 0, "known_patterns": 0,
                    "avg_triage_ms": 0.0, "avg_investigate_ms": 0.0,
                }
                phase_rail = gr.HTML(render_phase_rail(idle_snap))
                loop_stats = gr.HTML(render_loop_stats(idle_snap))
                telemetry_html = gr.HTML(render_telemetry())

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Incidents CORTEX raised on its own")
                        incidents_html = gr.HTML(render_incidents([]))
                    with gr.Column(scale=2):
                        gr.Markdown("### Reasoning trace")
                        activity_html = gr.HTML(render_activity([]))

                with gr.Accordion("Incident table", open=False):
                    incident_table = gr.Dataframe(
                        headers=[
                            "time", "source", "label", "mitre", "risk",
                            "verdict", "cwe/cve", "latency", "alert",
                        ],
                        interactive=False,
                        wrap=True,
                    )
                auto_timer = gr.Timer(max(interval, 1.5), active=True)

            # ---------- Live SOC ----------
            with gr.Tab("Live SOC"):
                with gr.Row():
                    live_toggle = gr.Checkbox(value=True, label="Stream live threat feed")
                    gr.Markdown(
                        "_Feed uses a fast on-host heuristic for real-time feel; the Autonomy "
                        "tab runs full Gemma routing on the same stream._"
                    )
                stats = gr.HTML(render_stats(feed))
                feed_html = gr.HTML(render_feed_html(feed))
                charts = gr.Plot(value=build_charts(feed), label="Detection graphics")
                live_table = gr.Dataframe(
                    headers=["time", "source", "label", "mitre", "consensus", "message"],
                    value=feed.table_rows(),
                    label="Triage table",
                    interactive=False,
                    wrap=True,
                )
                live_timer = gr.Timer(1.6, active=False)

            # ---------- Scripted demo ----------
            with gr.Tab("Guided demo"):
                gr.Markdown(
                    "One-click walkthrough: triage → grounded CVE → abstention → autonomous agent."
                )
                demo_btn = gr.Button("Run full demo", variant="primary")
                demo_walk = gr.Markdown()
                demo_table = gr.Dataframe(
                    headers=["line", "label", "mitre", "consensus", "agreement", "deep?"],
                    interactive=False,
                    wrap=True,
                )
                with gr.Accordion("Deep triage reports", open=False):
                    demo_deep = gr.Markdown()
                with gr.Row():
                    demo_cve = gr.Markdown(label="Grounded CVE")
                    demo_abstain = gr.Markdown(label="Abstention")
                demo_agent = gr.Markdown()
                with gr.Accordion("Agent tool trace", open=False):
                    demo_trace = gr.Markdown()

            # ---------- Log triage ----------
            with gr.Tab("Log triage"):
                gr.Markdown(
                    "Stage-1 consensus votes stop early once the winner is mathematically "
                    "locked, and identical lines are analyzed once."
                )
                logs = gr.Textbox(lines=10, label="Log lines", value=load_sample_logs())
                log_btn = gr.Button("Run two-stage triage", variant="primary")
                log_table = gr.Dataframe(
                    headers=["line", "label", "mitre", "consensus", "agreement", "deep?"],
                    interactive=False,
                    wrap=True,
                )
                log_report = gr.Markdown()

            # ---------- Multimodal diagram review ----------
            with gr.Tab("Diagram review"):
                gr.Markdown(
                    "**Multimodal architecture review.** Upload an infrastructure diagram. "
                    "Gemma 4 vision reads the topology it can actually see and maps each "
                    "weakness to a **MITRE ATT&CK** technique and a **NIST SP 800-53** "
                    "control. CVEs are attached separately by local index lookup on the "
                    "components it named — never recalled from training, and stripped if "
                    "unsupported. NIST mappings are advisory control references, not a "
                    "compliance score."
                )
                with gr.Row():
                    diagram_img = gr.Image(
                        label="Infrastructure diagram", type="filepath", height=320
                    )
                    with gr.Column():
                        diagram_btn = gr.Button("Review architecture", variant="primary")
                        gr.Markdown(
                            "_Vision runs on a stock `gemma4` tag (the SecOps fine-tune has "
                            "no image input). A new diagram takes a couple of minutes on the "
                            "12B vision model; results are content-addressed, so a diagram "
                            "you have reviewed before returns instantly._"
                        )
                        if DEMO_DIAGRAM.is_file():
                            gr.Examples(
                                examples=[[str(DEMO_DIAGRAM)]],
                                inputs=[diagram_img],
                                label="Sample as-built network",
                            )
                diagram_out = gr.HTML()

            # ---------- Code analyzer ----------
            with gr.Tab("Code analyzer"):
                code_file = gr.File(
                    label="Upload source",
                    file_types=[".py", ".js", ".ts", ".go", ".java", ".txt"],
                )
                code_box = gr.Textbox(
                    lines=12,
                    label="Or paste code",
                    value=VULN_CODE.read_text() if VULN_CODE.exists() else "",
                )
                code_btn = gr.Button("Analyze code", variant="primary")
                code_out = gr.Markdown()

            # ---------- CVE ----------
            with gr.Tab("Grounded CVE"):
                gr.Markdown(
                    "Answers come only from the local post-cutoff index. Unsupported CVE "
                    "claims are stripped and forced to `insufficient_evidence`."
                )
                cve_q = gr.Textbox(label="CVE / question", placeholder="CVE-2025-24813")
                cve_btn = gr.Button("Grounded report", variant="primary")
                cve_out = gr.Markdown()

            # ---------- Agent ----------
            with gr.Tab("Agent"):
                gr.Markdown(
                    f"**Autonomous defense on demand** — no prompt required. CORTEX audits "
                    f"`{default_target}` with native tool calling, then critiques its own report."
                )
                agent_target = gr.Textbox(
                    label="Defend target (path)", value=default_target, lines=1
                )
                defend_btn = gr.Button("Defend now", variant="primary")
                with gr.Accordion("Advanced: custom goal override", open=False):
                    agent_goal = gr.Textbox(
                        lines=3,
                        label="Optional goal (leave blank for autonomous default)",
                        value="",
                        placeholder="Leave empty — Defend now uses the autonomous goal.",
                    )
                    agent_btn = gr.Button("Run with custom goal")
                agent_out = gr.Markdown()
                with gr.Accordion("Agent tool trace", open=False):
                    agent_trace = gr.Markdown()

        gr.Markdown(
            "_Consensus labels are agreement counts, not confidence intervals. "
            "Attack paths are defensive explanations, never working exploit code. "
            "No cloud calls at run time._"
        )

        autonomy_outputs = [
            hero,
            phase_rail,
            loop_stats,
            incidents_html,
            activity_html,
            telemetry_html,
            incident_table,
        ]
        engage_btn.click(autonomy_start, outputs=autonomy_outputs)
        stop_btn.click(autonomy_stop, outputs=autonomy_outputs)
        reset_btn.click(autonomy_reset, outputs=autonomy_outputs)
        step_btn.click(autonomy_step, outputs=autonomy_outputs)
        auto_timer.tick(autonomy_bundle, outputs=autonomy_outputs)

        live_toggle.change(
            lambda live: (gr.update(active=bool(live)), *live_bundle(bool(live))),
            inputs=[live_toggle],
            outputs=[live_timer, stats, feed_html, charts, live_table],
        )
        live_timer.tick(
            live_bundle, inputs=[live_toggle], outputs=[stats, feed_html, charts, live_table]
        )

        demo_btn.click(
            run_full_demo,
            outputs=[demo_walk, demo_table, demo_deep, demo_cve, demo_abstain, demo_agent, demo_trace],
        )
        log_btn.click(run_log_triage, inputs=[logs], outputs=[log_table, log_report])
        diagram_btn.click(run_diagram_analysis, inputs=[diagram_img], outputs=[diagram_out])
        code_btn.click(run_code_audit, inputs=[code_file, code_box], outputs=[code_out])
        cve_btn.click(run_cve_query, inputs=[cve_q], outputs=[cve_out])
        defend_btn.click(
            run_autonomous_defend, inputs=[agent_target], outputs=[agent_out, agent_trace]
        )
        agent_btn.click(run_agent, inputs=[agent_goal], outputs=[agent_out, agent_trace])

        demo.load(autonomy_bundle, outputs=autonomy_outputs)

    return demo


def live_bundle(live: bool = True):
    """Advance the heuristic stream when live, then re-render its widgets."""
    feed = FEED
    if live:
        feed.running = True
        feed.push_random()
    else:
        feed.running = False
    return render_stats(feed), render_feed_html(feed), build_charts(feed), feed.table_rows()


def main() -> None:
    ui = build_ui()
    ui.launch(css=THEME_CSS)


if __name__ == "__main__":
    main()
