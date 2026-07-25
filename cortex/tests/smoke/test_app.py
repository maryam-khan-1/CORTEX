from app import build_ui, load_sample_logs
demo = build_ui()
assert demo is not None and hasattr(demo, "launch")
assert "Failed login" in load_sample_logs()
print("app ok", type(demo).__name__)
