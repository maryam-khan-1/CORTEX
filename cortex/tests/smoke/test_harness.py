from core.harness import apply_grounding_rule, extract_json, abstain_report
from core.schema import Finding, Report, Verdict
r = Report(findings=[Finding(verdict=Verdict.VULNERABLE, explanation="uses CVE-2026-99999", cve_id="CVE-2026-99999", grounded_on=[])], overall_risk=90)
assert apply_grounding_rule(r, allowed_doc_ids=set()).findings[0].cve_id is None and abstain_report("x").abstained
print("harness grounding ok", extract_json('{"a":1}'))
