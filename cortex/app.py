"""CORTEX Gradio UI — animated live SOC dashboard + analysis modes."""

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
from core.engine import Engine, load_config
from core.harness import Harness, apply_grounding_rule, abstain_report
from core.live_feed import FEED, LiveFeedState
from core.rag import RAG
from core.schema import Finding, Report, Verdict
from core.triage import Triage

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
SAMPLE_LOGS = ROOT / "data" / "sample_logs.csv"
VULN_CODE = ROOT / "data" / "vulnerable_code.py"
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
        self.feed: LiveFeedState = FEED


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


# ---------- Live dashboard renderers ----------

def render_hero(live: bool) -> str:
    status = "Live stream" if live else "Paused"
    return f"""
    <div class="cortex-hero">
      <h1 class="cortex-brand">CORTEX</h1>
      <p class="cortex-tagline">
        Grounded, offline-first blue-team assistant on Gemma 4 —
        honest abstention when evidence is missing, native tool-calling when it isn’t.
      </p>
      <div class="cortex-live-pill">
        <span class="cortex-live-dot" style="{'opacity:0.25;animation:none' if not live else ''}"></span>
        {status} · post-cutoff CVE/KEV grounding
      </div>
    </div>
    """


def render_stats(feed: LiveFeedState) -> str:
    window = feed.counts()
    total = feed.critical_total + feed.suspicious_total + feed.benign_total
    return f"""
    <div class="cortex-stats">
      <div class="cortex-stat total" style="animation-delay:0ms">
        <div class="k">Events seen</div><div class="v">{total}</div>
      </div>
      <div class="cortex-stat critical" style="animation-delay:60ms">
        <div class="k">Critical</div><div class="v">{feed.critical_total}</div>
      </div>
      <div class="cortex-stat suspicious" style="animation-delay:120ms">
        <div class="k">Suspicious</div><div class="v">{feed.suspicious_total}</div>
      </div>
      <div class="cortex-stat benign" style="animation-delay:180ms">
        <div class="k">Window mix</div>
        <div class="v" style="font-size:1.15rem;margin-top:0.4rem">
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
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[h["critical"] for h in hist],
            name="critical",
            line=dict(color="#b42318", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(180,35,24,0.12)",
            hovertemplate="t=%{x}<br>critical=%{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[h["suspicious"] for h in hist],
            name="suspicious",
            line=dict(color="#c45c26", width=2.2),
            hovertemplate="t=%{x}<br>suspicious=%{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[h["benign"] for h in hist],
            name="benign",
            line=dict(color="#2f6b4f", width=1.8, dash="dot"),
            hovertemplate="t=%{x}<br>benign=%{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=42, b=10),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(family="IBM Plex Sans", size=12, color="#14212b"),
        legend=dict(orientation="h", y=1.18, x=0.55),
    )
    fig.update_xaxes(title_text="tick", gridcolor="rgba(20,33,43,0.08)")
    fig.update_yaxes(title_text="count", gridcolor="rgba(20,33,43,0.08)")
    return fig


def live_dashboard_bundle(live: bool):
    feed = FEED
    if live:
        feed.running = True
        feed.push_random()
    else:
        feed.running = False
    return (
        render_hero(live),
        render_stats(feed),
        render_feed_html(feed),
        build_charts(feed),
        feed.table_rows(),
    )


def tick_feed(live: bool):
    """Timer callback — advance stream when live."""
    return live_dashboard_bundle(bool(live))


# ---------- Analysis modes (unchanged contracts) ----------

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
    docs = []
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
    docs = []
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
    md = (
        f"**Steps:** {result.steps}  \n"
        f"**Docs retrieved:** {', '.join(f'`{i}`' for i in result.retrieved_doc_ids) or '_none_'}\n\n"
        + _report_to_md(result.report)
    )
    return md, format_trace(result.trace)


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
    header = (
        f"_Autonomous defend · target `{path or st.agent.default_target}`_\n\n"
    )
    md, trace = _format_agent_result(result)
    return header + md, trace


def run_full_demo() -> tuple[str, list[list[Any]], str, str, str, str, str]:
    st = get_state()
    demo_cfg = st.config.get("demo", {})
    narration: list[str] = [
        "# CORTEX live demo",
        f"- Fast model: `{st.engine.fast_model}`",
        f"- Deep model: `{st.engine.deep_model}`",
        "- Pipeline: E2B triage → SecOps deep analysis; agent uses **native function calling**.",
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
    narration.append("\n## 4. Autonomous agent defend (native tools)")
    agent_md, agent_trace = run_autonomous_defend()
    narration.append("Model chose tool order — not a fixed pipeline.")
    narration.append("\n---\n**Demo complete.**")
    return "\n".join(narration), table, deep_md, cve_md, abstain_md, agent_md, agent_trace


def build_ui() -> gr.Blocks:
    choices = [
        "Live SOC",
        "Demo",
        "Log triage",
        "Code analyzer",
        "CVE / grounded report",
        "Agent",
    ]
    default_mode = "Live SOC"
    feed = FEED

    with gr.Blocks(title="CORTEX") as demo:
        hero = gr.HTML(render_hero(True))

        mode = gr.Radio(choices=choices, value=default_mode, label="Mode")

        # ---- Live SOC ----
        with gr.Column(visible=True) as live_panel:
            with gr.Row():
                live_toggle = gr.Checkbox(value=True, label="Stream live threat feed")
                gr.Markdown(
                    "_Feed uses a fast on-host heuristic for real-time feel; "
                    "switch to **Log triage** / **Demo** for full Gemma routing._"
                )
            stats = gr.HTML(render_stats(feed))
            feed_html = gr.HTML(render_feed_html(feed))
            charts = gr.Plot(value=build_charts(feed), label="Detection graphics")
            live_table = gr.Dataframe(
                headers=["time", "source", "label", "mitre", "consensus", "message"],
                value=feed.table_rows(),
                label="Triage table (color labels in feed above)",
                interactive=False,
                wrap=True,
            )
            timer = gr.Timer(1.6, active=True)

        # ---- Scripted demo ----
        with gr.Column(visible=False) as demo_panel:
            gr.Markdown("One-click walkthrough: triage → grounded CVE → abstention → agent.")
            demo_btn = gr.Button("Run full demo", variant="primary")
            demo_walk = gr.Markdown()
            demo_table = gr.Dataframe(
                headers=["line", "label", "mitre", "consensus", "agreement", "deep?"],
                interactive=False,
            )
            demo_deep = gr.Markdown()
            demo_cve = gr.Markdown()
            demo_abstain = gr.Markdown()
            demo_agent = gr.Markdown()
            demo_trace = gr.Markdown()

        with gr.Column(visible=False) as log_panel:
            logs = gr.Textbox(lines=12, label="Log lines", value=load_sample_logs())
            log_btn = gr.Button("Run two-stage triage", variant="primary")
            log_table = gr.Dataframe(
                headers=["line", "label", "mitre", "consensus", "agreement", "deep?"],
                interactive=False,
            )
            log_report = gr.Markdown()

        with gr.Column(visible=False) as code_panel:
            code_file = gr.File(label="Upload source", file_types=[".py", ".js", ".ts", ".go", ".java", ".txt"])
            code_box = gr.Textbox(
                lines=14,
                label="Or paste code",
                value=VULN_CODE.read_text() if VULN_CODE.exists() else "",
            )
            code_btn = gr.Button("Analyze code", variant="primary")
            code_out = gr.Markdown()

        with gr.Column(visible=False) as cve_panel:
            cve_q = gr.Textbox(label="CVE / question", placeholder="CVE-2025-24813")
            cve_btn = gr.Button("Grounded report", variant="primary")
            cve_out = gr.Markdown()

        with gr.Column(visible=False) as agent_panel:
            agent_cfg = load_config().get("agent") or {}
            default_target = str(
                agent_cfg.get("default_target", "data/vulnerable_code.py")
            )
            gr.Markdown(
                f"**Autonomous defense** — no prompt required. "
                f"CORTEX audits `{default_target}` with native tool calling "
                f"(`get_code_context` → Report; CVE tools only when citing evidence)."
            )
            agent_target = gr.Textbox(
                label="Defend target (path)",
                value=default_target,
                lines=1,
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
            agent_trace = gr.Markdown()

        def _toggle(m: str):
            return (
                gr.update(visible=m == "Live SOC"),
                gr.update(visible=m == "Demo"),
                gr.update(visible=m == "Log triage"),
                gr.update(visible=m == "Code analyzer"),
                gr.update(visible=m == "CVE / grounded report"),
                gr.update(visible=m == "Agent"),
                gr.update(active=m == "Live SOC"),
            )

        mode.change(
            _toggle,
            inputs=[mode],
            outputs=[live_panel, demo_panel, log_panel, code_panel, cve_panel, agent_panel, timer],
        )

        live_toggle.change(
            live_dashboard_bundle,
            inputs=[live_toggle],
            outputs=[hero, stats, feed_html, charts, live_table],
        )
        timer.tick(
            tick_feed,
            inputs=[live_toggle],
            outputs=[hero, stats, feed_html, charts, live_table],
        )

        demo_btn.click(
            run_full_demo,
            outputs=[demo_walk, demo_table, demo_deep, demo_cve, demo_abstain, demo_agent, demo_trace],
        )
        log_btn.click(run_log_triage, inputs=[logs], outputs=[log_table, log_report])
        code_btn.click(run_code_audit, inputs=[code_file, code_box], outputs=[code_out])
        cve_btn.click(run_cve_query, inputs=[cve_q], outputs=[cve_out])
        defend_btn.click(
            run_autonomous_defend, inputs=[agent_target], outputs=[agent_out, agent_trace]
        )
        agent_btn.click(run_agent, inputs=[agent_goal], outputs=[agent_out, agent_trace])

    return demo


def main() -> None:
    ui = build_ui()
    ui.launch(css=THEME_CSS)


if __name__ == "__main__":
    main()
