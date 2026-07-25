from app import build_ui
src = open("app.py").read()
assert "run_full_demo" in src and "Guided demo" in src and "Agent" in src
assert "run_autonomous_defend" in src and "Defend now" in src
assert "Engage autonomous defense" in src and "autonomy_bundle" in src
demo = build_ui()
print("demo ui ok", type(demo).__name__, hasattr(demo, "launch"))
