"""Latency machinery: cache round-trip, telemetry accounting, early-exit consensus."""
from pathlib import Path

from core.cache import InferenceCache, make_key
from core.telemetry import Telemetry
from core.triage import Triage, normalize_label, UNKNOWN_LABEL

tmp = Path("data/cache/_smoke.json")
c = InferenceCache(path=tmp, enabled=True)
k = make_key(model="m", prompt="p")
assert c.get(k) is None
c.set(k, "cached-value")
c.flush()
assert InferenceCache(path=tmp).get(k) == "cached-value"
c.clear()
assert not tmp.exists()

t = Telemetry()
t.record("stage1", model="m", role="fast", ms=1000.0)
t.record("stage1", model="m", role="fast", ms=0.4, cached=True)
snap = t.snapshot()
assert snap["total_calls"] == 2 and snap["cache_hits"] == 1
assert 0.49 < snap["cache_hit_rate"] < 0.51
assert snap["fast_avg_ms"] == 1000.0 and snap["saved_s"] > 0

# Early exit: two matching votes out of three cannot be overturned.
assert Triage._decided(["critical", "critical"], 3)
assert not Triage._decided(["critical", "benign"], 3)
assert Triage._decided(["benign", "benign", "critical"], 3)

# A fine-tune saying "malicious" must not be dropped, and a parse failure must never
# be silently downgraded to benign.
assert normalize_label("malicious") == "critical"
assert normalize_label("Medium") == "suspicious"
assert normalize_label("") == UNKNOWN_LABEL
assert UNKNOWN_LABEL != "benign"
print("latency ok", snap["cache_hit_rate"], "hit rate")
