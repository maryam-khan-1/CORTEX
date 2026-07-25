"""Agentic loop via Gemma 4 native function calling.

Tools: search_cve, lookup_kev, check_dependency, get_code_context.
The model decides which to call until it can produce a grounded Report.

Two properties matter here beyond "it calls tools":
  * closed loop — a cheap critique pass re-opens the investigation when a finding
    cites a CVE it never retrieved, so the agent corrects itself instead of the
    operator noticing later;
  * bounded cost — the final step drops the tool list so the loop always terminates
    with a schema-valid Report rather than an out-of-steps abstention.
Tool state is per-run, so several investigations can run on different threads.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.engine import Engine
from core.harness import (
    REPORT_SCHEMA_HINT,
    abstain_report,
    apply_grounding_rule,
    extract_json,
    report_json_schema,
)
from core.rag import RAG
from core.schema import Report, Verdict
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    result_preview: str


@dataclass
class AgentResult:
    report: Report
    trace: list[ToolEvent] = field(default_factory=list)
    steps: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    revisions: int = 0

    @property
    def tool_names(self) -> list[str]:
        return [e.name for e in self.trace]


class _RunContext:
    """Per-run tool state so concurrent investigations never share retrieval sets."""

    def __init__(self) -> None:
        self.retrieved: dict[str, str] = {}
        self.lock = threading.Lock()

    def remember(self, doc_id: str, text: str) -> None:
        with self.lock:
            self.retrieved[doc_id] = text

    def allowed(self) -> set[str]:
        with self.lock:
            return set(self.retrieved.keys())


class Agent:
    """Native tool-calling agent. Not a fixed pipeline — model chooses tools."""

    DEFAULT_TARGET = "data/vulnerable_code.py"
    DEFAULT_GOAL = (
        "Autonomous defend: audit data/vulnerable_code.py. Call get_code_context once. "
        "Report CWE findings from the code (SQL injection, command injection, unsafe pickle, etc.) "
        "— code findings do not need CVE ids or grounded_on. Only search_cve/lookup_kev if citing "
        "a specific CVE you retrieved. Then return ONLY Report JSON. Keep explanations short."
    )

    def __init__(
        self,
        engine: Engine,
        rag: Optional[RAG] = None,
        *,
        code_root: Optional[Path] = None,
        max_steps: int = 8,
        config: Optional[dict[str, Any]] = None,
    ):
        self.engine = engine
        self.rag = rag
        self.code_root = Path(code_root or ROOT)
        self.config = config or engine.config
        agent_cfg = self.config.get("agent") or {}
        self.max_steps = int(agent_cfg.get("max_steps", max_steps))
        self.default_target = str(agent_cfg.get("default_target", self.DEFAULT_TARGET))
        self.default_goal = str(agent_cfg.get("default_goal", self.DEFAULT_GOAL))
        self.max_revisions = int(agent_cfg.get("max_revisions", 1))
        self.self_critique = bool(agent_cfg.get("self_critique", True))
        self.rag_top_k = int(agent_cfg.get("rag_top_k", 4))
        # Generation length is the dominant latency term at ~15 tok/s on the deep model,
        # so the token budget is the knob the autonomy loop tightens.
        self.num_predict = int(agent_cfg.get("num_predict", 384))
        self.num_predict_terse = int(agent_cfg.get("num_predict_terse", 256))
        # Live alerts get a tighter step budget: every extra tool round-trip is another
        # ~15s on the deep model, and an alert rarely needs more than one lookup.
        self.alert_max_steps = int(agent_cfg.get("alert_max_steps", 3))
        self.constrain_json = bool(
            self.config.get("harness", {}).get("constrain_json", True)
        )
        samp = self.config.get("sampling", {}).get("extraction", {})
        self.temperature = float(samp.get("temperature", 0.2))
        self.top_p = float(samp.get("top_p", 0.95))
        self.top_k = int(samp.get("top_k", 64))

    # ---------- tools ----------

    def _build_tools(self, ctx: _RunContext) -> list[Callable]:
        agent = self

        def search_cve(query: str) -> str:
            """
            Semantic + structured search over the local post-cutoff CVE/KEV index.

            Args:
              query: Free-text vulnerability question or CVE keywords.

            Returns:
              str: JSON list of {id, text} evidence docs.
            """
            if agent.rag is None:
                return json.dumps({"error": "rag_unavailable", "docs": []})
            docs = agent.rag.retrieve_for_query(query, top_k=agent.rag_top_k)
            return agent._docs_payload(docs, ctx)

        def lookup_kev(cve_id: str) -> str:
            """
            Exact KEV / CVE record lookup by CVE identifier.

            Args:
              cve_id: CVE id like CVE-2025-24813.

            Returns:
              str: JSON with matching KEV/CVE evidence docs, or empty if unknown locally.
            """
            if agent.rag is None:
                return json.dumps({"error": "rag_unavailable", "docs": []})
            docs = agent.rag.lookup_kev(cve_id) or agent.rag.lookup_cve(cve_id)
            payload = json.loads(agent._docs_payload(docs, ctx))
            payload.update({"cve_id": cve_id, "in_kev": bool(docs)})
            return json.dumps(payload)

        def check_dependency(package: str, ecosystem: str = "PyPI") -> str:
            """
            Structured lookup of known-bad advisories for a dependency by name.

            Args:
              package: Package name, e.g. requests or django.
              ecosystem: Package ecosystem, e.g. PyPI or npm.

            Returns:
              str: JSON with advisory evidence docs for that package, empty if none known.
            """
            if agent.rag is None:
                return json.dumps({"error": "rag_unavailable", "docs": []})
            docs = agent.rag.lookup_package(package, ecosystem)
            payload = json.loads(agent._docs_payload(docs, ctx))
            payload.update({"package": package, "ecosystem": ecosystem})
            return json.dumps(payload)

        def get_code_context(path: str) -> str:
            """
            Read a source file (or region) from the local workspace for audit context.

            Args:
              path: Relative path under the project root, e.g. data/vulnerable_code.py

            Returns:
              str: File contents (truncated) or an error message.
            """
            target = (agent.code_root / path).resolve()
            root = agent.code_root.resolve()
            if not str(target).startswith(str(root)) or not target.is_file():
                return json.dumps({"error": f"path_not_allowed_or_missing: {path}"})
            text = target.read_text(errors="replace")
            if len(text) > 12000:
                text = text[:12000] + "\n...[truncated]..."
            return json.dumps({"path": path, "content": text})

        return [search_cve, lookup_kev, check_dependency, get_code_context]

    def _docs_payload(self, docs: list[Any], ctx: _RunContext) -> str:
        out = []
        for d in docs:
            ctx.remember(d.id, d.text)
            text = d.text if len(d.text) <= 420 else d.text[:420] + "…"
            out.append(
                {"id": d.id, "text": text, "cve_id": (d.metadata or {}).get("cve_id", "")}
            )
        return json.dumps({"docs": out})

    # ---------- prompts ----------

    def _system_prompt(self) -> str:
        return (
            "You are CORTEX, an autonomous offline blue-team defender.\n"
            "Do not ask the operator for instructions — start defending immediately.\n"
            "Tools: get_code_context, search_cve, lookup_kev, check_dependency.\n"
            "Preferred procedure:\n"
            "1) Call get_code_context on the target once.\n"
            "2) Emit ONLY a JSON Report. Code-pattern / CWE findings do NOT need CVE ids "
            "or grounded_on entries.\n"
            "3) Call search_cve / lookup_kev / check_dependency only when claiming a "
            "specific CVE id; then grounded_on must list retrieved doc ids you actually saw.\n"
            "Gemma cutoff is Jan 2025 — never invent 2025/2026 CVEs.\n"
            "If CVE evidence is missing, verdict=insufficient_evidence and omit cve_id.\n"
            "Finish quickly: prefer Report JSON over extra tool calls.\n\n"
            + REPORT_SCHEMA_HINT
        )

    def resolve_goal(self, user_goal: Optional[str] = None) -> str:
        """Use operator text if provided; otherwise the configured autonomous defend goal."""
        g = (user_goal or "").strip()
        return g if g else self.default_goal

    # ---------- entry points ----------

    def defend(
        self,
        *,
        target: Optional[str] = None,
        model: Optional[str] = None,
        on_event: Optional[Callable[[str, str], None]] = None,
    ) -> AgentResult:
        """Autonomous defense — no free-text goal required."""
        path = (target or self.default_target).strip() or self.default_target
        goal = (
            f"Autonomous defend: audit {path}. Call get_code_context once on that path. "
            "Report CWE findings from the code — code findings do not need CVE ids or "
            "grounded_on. Only search_cve/lookup_kev if citing a specific CVE you retrieved. "
            "Then return ONLY Report JSON. Keep explanations short."
        )
        return self.run(goal, model=model, on_event=on_event)

    def investigate(
        self,
        alert: str,
        *,
        context: str = "",
        model: Optional[str] = None,
        on_event: Optional[Callable[[str, str], None]] = None,
    ) -> AgentResult:
        """Investigate a single live alert — the autonomy loop's escalation path."""
        goal = (
            "Autonomous defend: a detection fired. Investigate and return ONLY Report JSON.\n"
            f"ALERT: {alert}\n"
            + (f"CONTEXT: {context}\n" if context else "")
            + "Use search_cve or check_dependency only if you need CVE evidence; MITRE/CWE "
            "reasoning from the alert text alone is fine and needs no grounded_on. "
            "Emit exactly one finding with a one-sentence explanation, a one-sentence "
            "attack_path and a one-sentence fix. Be terse."
        )
        return self.run(
            goal,
            model=model,
            on_event=on_event,
            num_predict=self.num_predict_terse,
            max_steps=self.alert_max_steps,
        )

    # ---------- core loop ----------

    def run(
        self,
        user_goal: str,
        *,
        model: Optional[str] = None,
        on_event: Optional[Callable[[str, str], None]] = None,
        num_predict: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> AgentResult:
        """Tool loop, then a critique pass that can re-open the investigation once."""
        goal = self.resolve_goal(user_goal)
        budget = num_predict if num_predict is not None else self.num_predict
        steps = max_steps if max_steps is not None else self.max_steps
        result = self._tool_loop(
            goal, model=model, on_event=on_event, num_predict=budget, max_steps=steps
        )

        if not self.self_critique:
            return result

        critiques: list[str] = []
        revisions = 0
        while revisions < self.max_revisions:
            critique = self._critique(goal, result)
            if not critique:
                break
            critiques.append(critique)
            self._emit(on_event, "critique", critique)
            revised = self._tool_loop(
                goal
                + "\n\nPREVIOUS ATTEMPT WAS REJECTED BY REVIEW: "
                + critique
                + "\nFix exactly that and return corrected Report JSON.",
                model=model,
                on_event=on_event,
                num_predict=budget,
                max_steps=steps,
            )
            revisions += 1
            # Keep the revision only when review actually improved it.
            if self._score(revised.report) >= self._score(result.report):
                revised.trace = result.trace + revised.trace
                revised.steps += result.steps
                result = revised
            break

        result.critiques = critiques
        result.revisions = revisions
        return result

    def _tool_loop(
        self,
        goal: str,
        *,
        model: Optional[str] = None,
        on_event: Optional[Callable[[str, str], None]] = None,
        num_predict: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> AgentResult:
        step_budget = max_steps if max_steps is not None else self.max_steps
        ctx = _RunContext()
        tools = self._build_tools(ctx)
        tools_by_name = {fn.__name__: fn for fn in tools}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": goal},
        ]
        trace: list[ToolEvent] = []
        # Once the model has answered in prose instead of JSON, arguing with it costs
        # another full deep call. Go straight to a constrained closing turn instead.
        force_finalize = False

        for step in range(1, step_budget + 1):
            final_step = force_finalize or step >= step_budget
            if final_step:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "FINAL STEP: Do not call tools. Return ONLY valid Report JSON now. "
                            "Use CWE findings from code you already read; omit cve_id unless "
                            "you retrieved supporting docs."
                        ),
                    }
                )

            resp = self.engine.chat(
                messages,
                model=model,
                tools=None if final_step else tools,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                role="deep",
                # Constrain the closing turn so the last step can't waste itself on prose.
                format=report_json_schema() if (final_step and self.constrain_json) else None,
                num_predict=num_predict,
                label="agent-step",
            )
            msg = resp.message
            # Append assistant turn (preserve tool_calls for the protocol)
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls and not final_step:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)

            if tool_calls and not final_step:
                for tc in tool_calls:
                    name, args = _parse_tool_call(tc)
                    impl = tools_by_name.get(name or "")
                    if impl is None:
                        result_text = json.dumps({"error": f"unknown_tool: {name}"})
                    else:
                        try:
                            result_text = impl(**args)
                        except TypeError as e:
                            result_text = json.dumps({"error": f"bad_args: {e}"})
                        except Exception as e:
                            result_text = json.dumps({"error": str(e)})

                    preview = result_text if len(result_text) < 400 else result_text[:400] + "…"
                    trace.append(
                        ToolEvent(name=name or "?", arguments=args, result_preview=preview)
                    )
                    self._emit(on_event, "tool", f"{name} {json.dumps(args)[:120]}")
                    # Truncate tool payloads in the chat history to keep prompt eval cheap.
                    tool_content = result_text if len(result_text) <= 1800 else result_text[:1800] + "…"
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": tool_content}
                    )
                continue

            # No tool calls (or final step) — try to parse final Report
            content = msg.content or ""
            try:
                data = extract_json(content)
                report = Report.model_validate(data)
                allowed = ctx.allowed()
                report = apply_grounding_rule(report, allowed_doc_ids=allowed)
                self._emit(on_event, "report", f"risk {report.overall_risk}/100")
                return AgentResult(
                    report=report,
                    trace=trace,
                    steps=step,
                    retrieved_doc_ids=sorted(allowed),
                )
            except (ValidationError, ValueError, json.JSONDecodeError, TypeError) as e:
                if final_step:
                    break
                force_finalize = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your last output failed because: {e}; "
                            "return valid JSON matching the schema."
                        ),
                    }
                )
                continue

        report = abstain_report(
            f"Agent reached max_steps={step_budget} without a valid grounded Report."
        )
        self._emit(on_event, "abstain", "out of steps")
        return AgentResult(
            report=report,
            trace=trace,
            steps=step_budget,
            retrieved_doc_ids=sorted(ctx.allowed()),
        )

    # ---------- reflection ----------

    def _critique(self, goal: str, result: AgentResult) -> str:
        """Deterministic checks first; only ask the model when the cheap checks pass.

        Returns a critique string to act on, or "" when the report is acceptable.
        """
        report = result.report
        allowed = set(result.retrieved_doc_ids)

        if report.abstained and not report.findings:
            return "Report abstained with no findings; produce concrete CWE findings from the code."
        if not report.findings:
            return "Report contained zero findings; list at least the clear code weaknesses."
        for f in report.findings:
            if f.cve_id and not set(f.grounded_on or []).issubset(allowed):
                return (
                    f"Finding cites {f.cve_id} without retrieved evidence; "
                    "either retrieve it or drop the CVE and report the CWE."
                )
        if all(f.verdict == Verdict.INSUFFICIENT_EVIDENCE for f in report.findings) and (
            "get_code_context" in result.tool_names
        ):
            return (
                "Every finding abstained even though the source was read; report the "
                "concrete CWE weaknesses visible in that code."
            )
        return ""

    @staticmethod
    def _score(report: Report) -> int:
        """Prefer reports that commit to grounded findings over ones that abstain."""
        if not report.findings:
            return -1
        decided = sum(
            1 for f in report.findings if f.verdict != Verdict.INSUFFICIENT_EVIDENCE
        )
        return decided * 2 + len(report.findings) - (5 if report.abstained else 0)

    @staticmethod
    def _emit(on_event: Optional[Callable[[str, str], None]], kind: str, text: str) -> None:
        if on_event is None:
            return
        try:
            on_event(kind, text)
        except Exception:
            pass


def _parse_tool_call(tc: Any) -> tuple[Optional[str], dict[str, Any]]:
    fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
    name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
    raw_args = (
        getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
    )
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = dict(raw_args or {})
    return name, args


def format_trace(trace: list[ToolEvent]) -> str:
    if not trace:
        return "_No tool calls — model answered without tools._"
    lines = ["### Agent tool trace"]
    for i, ev in enumerate(trace, 1):
        lines.append(f"**{i}. `{ev.name}`** args=`{json.dumps(ev.arguments)}`")
        lines.append(f"```\n{ev.result_preview}\n```")
    return "\n".join(lines)
