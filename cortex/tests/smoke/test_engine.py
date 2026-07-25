from core.engine import Engine
e = Engine()
assert e.fast_model and e.deep_model
assert any("gemma" in m.lower() or "entrick" in m.lower() or "secops" in m.lower() for m in e.available)
print("engine ok", "fast=", e.fast_model, "deep=", e.deep_model, "n=", len(e.available))
