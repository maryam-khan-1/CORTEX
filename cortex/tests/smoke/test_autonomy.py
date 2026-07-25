"""Autonomy loop wiring — exercised with stubs so it stays offline and instant."""
from core.autonomy import AutonomyLoop, PHASES, signature
from core.live_feed import LiveFeedState
from core.schema import Finding, Report, Verdict
from core.triage import Stage1Result


class FakeTriage:
    def classify_one(self, item, n=None):
        crit = "mimikatz" in item.lower()
        label = "critical" if crit else "benign"
        return Stage1Result(
            item=item, label=label, mitre="T1003", rationale="stub",
            votes={label: 2}, consensus=f"{label.capitalize()} (2/2 agree)",
            agreement=1.0, votes_spent=2,
        )


class FakeAgentResult:
    report = Report(
        findings=[Finding(verdict=Verdict.VULNERABLE, explanation="cred dumping",
                          cwe_id="CWE-522", grounded_on=[])],
        overall_risk=91,
    )
    trace = []
    steps = 2
    retrieved_doc_ids = []
    critiques = ["stub critique"]
    revisions = 1
    tool_names = ["get_code_context"]


class FakeAgent:
    def investigate(self, alert, context="", on_event=None):
        if on_event:
            on_event("tool", "get_code_context")
        return FakeAgentResult()


# Near-duplicate alerts must collapse to one signature so repeats aren't re-analyzed.
a = signature("Failed login for admin from 10.0.0.1 - 12 attempts")
b = signature("Failed login for admin from 203.0.113.9 - 47 attempts")
assert a == b, (a, b)

feed = LiveFeedState()
feed.FEED_ONLY = True
loop = AutonomyLoop(FakeTriage(), FakeAgent(), feed, config={"autonomy": {"interval_s": 0.01}})

# Force a critical event through a full cycle.
feed.push_random = lambda: _crit(feed)


def _crit(f):
    from core.live_feed import FeedEvent
    ev = FeedEvent(ts="00:00:00", source="edr",
                   message="Process created: mimikatz.exe parent=winword.exe",
                   label="critical", mitre="T1003", consensus="Critical (3/3 agree)", score=0.9)
    f.events.appendleft(ev)
    return ev


inc = loop.cycle_once()
assert inc is not None and inc.risk == 91 and inc.revisions == 1, inc
assert inc.cwe == "CWE-522" and inc.verdict == "vulnerable"

# Same signature again -> suppressed, not re-investigated.
assert loop.cycle_once() is None
snap = loop.snapshot()
assert snap["escalated"] == 1 and snap["suppressed"] == 1
assert snap["phase"] in PHASES or snap["phase"] == "idle"
assert loop.incident_rows() and len(loop.incident_rows()[0]) == 9
assert not loop.running
print("autonomy ok", snap["cycles"], "cycles", snap["escalated"], "escalated",
      snap["suppressed"], "suppressed")
