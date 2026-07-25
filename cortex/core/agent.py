"""Agentic loop via Gemma 4 native function calling.

Tools: search_cve, lookup_kev, get_code_context.
The model decides which to call until it can produce a grounded Report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.engine import Engine
from core.harness import REPORT_SCHEMA_HINT, apply_grounding_rule, abstain_report, extract_json
from core.rag import RAG
from core.schema import Report
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
        samp = self.config.get("sampling", {}).get("extraction", {})
        self.temperature = float(samp.get("temperature", 0.2))
        self.top_p = float(samp.get("top_p", 0.95))
        self.top_k = int(samp.get("top_k", 64))
        self._retrieved: dict[str, str] = {}  # id -> text
        self._tools = self._build_tools()

    def _build_tools(self) -> list[Callable]:
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
            docs = agent.rag.retrieve_for_query(query, top_k=5)
            out = []
            for d in docs:
                agent._retrieved[d.id] = d.text
                out.append({"id": d.id, "text": d.text, "cve_id": (d.metadata or {}).get("cve_id", "")})
            return json.dumps({"docs": out})

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
            out = []
            for d in docs:
                agent._retrieved[d.id] = d.text
                out.append({"id": d.id, "text": d.text, "cve_id": (d.metadata or {}).get("cve_id", "")})
            return json.dumps({"cve_id": cve_id, "docs": out, "in_kev": bool(docs)})

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

        return [search_cve, lookup_kev, get_code_context]

    def _tool_map(self) -> dict[str, Callable]:
        return {fn.__name__: fn for fn in self._tools}

    def _system_prompt(self) -> str:
        return (
            "You are CORTEX, an autonomous offline blue-team defender.\n"
            "Do not ask the operator for instructions — start defending immediately.\n"
            "Tools: get_code_context, search_cve, lookup_kev.\n"
            "Preferred procedure:\n"
            "1) Call get_code_context on the target once.\n"
            "2) Emit ONLY a JSON Report. Code-pattern / CWE findings do NOT need CVE ids "
            "or grounded_on entries.\n"
            "3) Call search_cve / lookup_kev only when claiming a specific CVE id; then "
            "grounded_on must list retrieved doc ids you actually saw.\n"
            "Gemma cutoff is Jan 2025 — never invent 2025/2026 CVEs.\n"
            "If CVE evidence is missing, verdict=insufficient_evidence and omit cve_id.\n"
            "Finish quickly: prefer Report JSON over extra tool calls.\n\n"
            + REPORT_SCHEMA_HINT
        )

    def resolve_goal(self, user_goal: Optional[str] = None) -> str:
        """Use operator text if provided; otherwise the configured autonomous defend goal."""
        g = (user_goal or "").strip()
        return g if g else self.default_goal

    def defend(self, *, target: Optional[str] = None, model: Optional[str] = None) -> AgentResult:
        """Autonomous defense — no free-text goal required."""
        path = (target or self.default_target).strip() or self.default_target
        goal = (
            f"Autonomous defend: audit {path}. Call get_code_context once on that path. "
            "Report CWE findings from the code — code findings do not need CVE ids or "
            "grounded_on. Only search_cve/lookup_kev if citing a specific CVE you retrieved. "
            "Then return ONLY Report JSON. Keep explanations short."
        )
        return self.run(goal, model=model)

    def run(self, user_goal: str, *, model: Optional[str] = None) -> AgentResult:
        """Loop: model may call tools until it emits a grounded Report or max_steps."""
        goal = self.resolve_goal(user_goal)
        self._retrieved.clear()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": goal},
        ]
        trace: list[ToolEvent] = []
        tools_by_name = self._tool_map()

        for step in range(1, self.max_steps + 1):
            final_step = step >= self.max_steps
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
                tools=None if final_step else self._tools,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                role="deep",
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
                    fn = getattr(tc, "function", None) or tc.get("function")
                    name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
                    raw_args = (
                        getattr(fn, "arguments", None)
                        if not isinstance(fn, dict)
                        else fn.get("arguments")
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

                    impl = tools_by_name.get(name or "")
                    if impl is None:
                        result = json.dumps({"error": f"unknown_tool: {name}"})
                    else:
                        try:
                            result = impl(**args)
                        except TypeError as e:
                            result = json.dumps({"error": f"bad_args: {e}"})
                        except Exception as e:
                            result = json.dumps({"error": str(e)})

                    preview = result if len(result) < 400 else result[:400] + "…"
                    trace.append(ToolEvent(name=name or "?", arguments=args, result_preview=preview))
                    # Truncate tool payloads in the chat history to keep prompt eval cheap.
                    tool_content = result if len(result) <= 1800 else result[:1800] + "…"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "content": tool_content,
                        }
                    )
                continue

            # No tool calls (or final step) — try to parse final Report
            content = msg.content or ""
            try:
                data = extract_json(content)
                report = Report.model_validate(data)
                allowed = set(self._retrieved.keys())
                report = apply_grounding_rule(report, allowed_doc_ids=allowed)
                return AgentResult(
                    report=report,
                    trace=trace,
                    steps=step,
                    retrieved_doc_ids=list(allowed),
                )
            except (ValidationError, ValueError, json.JSONDecodeError, TypeError) as e:
                if final_step:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your last output failed because: {e}; "
                            "either call a tool or return valid JSON matching the schema."
                        ),
                    }
                )
                continue

        # Exhausted steps
        report = abstain_report(
            f"Agent reached max_steps={self.max_steps} without a valid grounded Report."
        )
        return AgentResult(
            report=report,
            trace=trace,
            steps=self.max_steps,
            retrieved_doc_ids=list(self._retrieved.keys()),
        )


def format_trace(trace: list[ToolEvent]) -> str:
    if not trace:
        return "_No tool calls — model answered without tools._"
    lines = ["### Agent tool trace"]
    for i, ev in enumerate(trace, 1):
        lines.append(f"**{i}. `{ev.name}`** args=`{json.dumps(ev.arguments)}`")
        lines.append(f"```\n{ev.result_preview}\n```")
    return "\n".join(lines)
