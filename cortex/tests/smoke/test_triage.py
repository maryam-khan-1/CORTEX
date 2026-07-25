from core.triage import Stage1Result, TriageResult
from core.schema import Report, Finding, Verdict
s = Stage1Result(item="x", label="critical", mitre="T1003", rationale="dump", votes={"critical":3}, consensus="Critical (3/3 agree)", agreement=1.0)
assert "3/3" in s.consensus and TriageResult(stage1=[s], reports={0: Report(findings=[Finding(verdict=Verdict.INSUFFICIENT_EVIDENCE, explanation="n", grounded_on=[])], overall_risk=0, abstained=True)}, flagged_indices=[0]).flagged_indices == [0]
print("triage ok", s.consensus)
