from core.agent import Agent, ToolEvent, format_trace
src = open("core/agent.py").read()
assert "search_cve" in src and "lookup_kev" in src and "get_code_context" in src
assert "native" in (Agent.__doc__ or "").lower() or "function calling" in src
assert "def defend" in src and "FINAL STEP" in src and "autonomous" in src.lower()
print("agent ok", format_trace([ToolEvent("search_cve", {"query": "tomcat"}, '{"docs":[]}')])[:40])
