from pathlib import Path
p = Path("scripts/build_index.py")
assert p.exists()
assert "known_exploited_vulnerabilities" in p.read_text()
print("build_index script ok", p.stat().st_size)
