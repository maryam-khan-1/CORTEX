from app import build_ui
src = open("app.py").read()
assert "Demo" in src and "run_full_demo" in src and "Agent" in src
assert "run_autonomous_defend" in src and "Defend now" in src
demo = build_ui()
print("demo ui ok", type(demo).__name__, hasattr(demo, "launch"))
