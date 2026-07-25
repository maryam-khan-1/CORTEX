from core.live_feed import LiveFeedState, classify_heuristic
f = LiveFeedState()
f.seed(3)
assert f.tick >= 3 and classify_heuristic("mimikatz.exe")[0] == "critical"
print("live_feed ok", dict(f.counts()), len(f.table_rows()))
