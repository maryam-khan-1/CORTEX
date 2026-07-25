from core.schema import Finding, Report, Verdict
f = Finding(verdict=Verdict.VULNERABLE, explanation="sqli", grounded_on=["doc-1"], cve_id="CVE-2025-0001")
r = Report(findings=[f], overall_risk=80)
assert r.findings[0].cve_id == "CVE-2025-0001" and r.overall_risk == 80
print("schema ok", Report.model_json_schema()["properties"]["overall_risk"])
