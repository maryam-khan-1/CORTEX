"""CORTEX Gradio UI — wiring only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from core.engine import Engine, load_config
from core.harness import Harness
from core.rag import RAG
from core.schema import Report
from core.triage import Triage

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
SAMPLE_LOGS = ROOT / "data" / "sample_logs.csv"
VULN_CODE = ROOT / "data" / "vulnerable_code.py"


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
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
            )
        except Exception as e:
            print(f"RAG unavailable: {e}")
            self.rag = None
        self.triage = Triage(self.engine, self.harness, self.rag, self.config)


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


def run_log_triage(log_text: str) -> tuple[list[list[Any]], str]:
    st = get_state()
    items = [ln.strip() for ln in (log_text or "").splitlines() if ln.strip()]
    if not items:
        return [], "No log lines provided."
    extra = _read_prompt("log_triage.md")
    result = st.triage.run(items, prompt_extra=extra)
    table = []
    for i, s1 in enumerate(result.stage1):
        deep = ""
        if i in result.reports:
            deep = _report_to_md(result.reports[i])
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
        reports_md.append(f"## Deep analysis — line {i+1}\n" + _report_to_md(result.reports[i]))
    return table, "\n\n".join(reports_md) if reports_md else "_No suspicious/critical lines flagged._"


def run_code_audit(file_obj, code_text: str) -> str:
    st = get_state()
    source = (code_text or "").strip()
    # Gradio filepath: type defaults to filepath (a str). Read with open(path).read()
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
        docs = st.rag.retrieve_for_query(source[:2000], top_k=st.config.get("rag", {}).get("top_k", 5))
        ids = [d.id for d in docs]
        evidence = st.rag.format_evidence(docs)
    user = f"{extra}\n\nSOURCE:\n```\n{source}\n```\n"
    if evidence:
        user += f"\nRETRIEVED EVIDENCE:\n{evidence}\n"
    report = st.harness.run(user, retrieved_doc_ids=ids if st.rag is not None else None, role="deep")
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
    report = st.harness.run(user, retrieved_doc_ids=ids if st.rag is not None else None, role="deep")
    return _report_to_md(report)


def build_ui() -> gr.Blocks:
    choices = ["Log triage", "Code analyzer", "CVE / grounded report"]
    default_mode = "Log triage"  # MUST exactly equal one of the choice strings
    with gr.Blocks(title="CORTEX") as demo:
        gr.Markdown(
            "# CORTEX\nGrounded, offline-first blue-team assistant on Gemma 4. "
            "Consensus labels are **agreement counts**, not confidence intervals."
        )
        mode = gr.Radio(choices=choices, value=default_mode, label="Mode")

        with gr.Column(visible=True) as log_panel:
            logs = gr.Textbox(lines=12, label="Log lines (one per line)", value=load_sample_logs())
            log_btn = gr.Button("Run two-stage triage")
            log_table = gr.Dataframe(
                headers=["line", "label", "mitre", "consensus", "agreement", "deep?"],
                label="Stage-1 triage",
                interactive=False,
            )
            log_report = gr.Markdown(label="Grounded deep reports")

        with gr.Column(visible=False) as code_panel:
            code_file = gr.File(label="Upload source (filepath)", file_types=[".py", ".js", ".ts", ".go", ".java", ".txt"])
            code_box = gr.Textbox(lines=16, label="Or paste code", value=VULN_CODE.read_text() if VULN_CODE.exists() else "")
            code_btn = gr.Button("Analyze code")
            code_out = gr.Markdown()

        with gr.Column(visible=False) as cve_panel:
            cve_q = gr.Textbox(label="CVE / question", placeholder="CVE-2025-24813 or describe the vuln class")
            cve_btn = gr.Button("Grounded report")
            cve_out = gr.Markdown()

        def _toggle(m: str):
            return (
                gr.update(visible=m == "Log triage"),
                gr.update(visible=m == "Code analyzer"),
                gr.update(visible=m == "CVE / grounded report"),
            )

        mode.change(_toggle, inputs=[mode], outputs=[log_panel, code_panel, cve_panel])
        log_btn.click(run_log_triage, inputs=[logs], outputs=[log_table, log_report])
        code_btn.click(run_code_audit, inputs=[code_file, code_box], outputs=[code_out])
        cve_btn.click(run_cve_query, inputs=[cve_q], outputs=[cve_out])
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
