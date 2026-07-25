"""Pre-compute the diagram review so the live demo is instant.

The 12B vision pass costs ~2-3 minutes cold. Results are content-addressed by image
hash, so running this once ahead of time makes the same diagram return in milliseconds
during the demo. Uploading a *different* diagram still runs the full pass — the cache is
a speed optimisation, not a canned answer.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/warm_vision.py [image ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.engine import Engine  # noqa: E402
from core.rag import RAG  # noqa: E402
from core.vision import VisionAnalyzer  # noqa: E402


def main() -> int:
    images = sys.argv[1:] or [str(ROOT / "data" / "demo_diagram.png")]
    analyzer = VisionAnalyzer(Engine(), RAG())
    if not analyzer.available:
        print("No vision-capable gemma4 tag available. Run: ollama pull gemma4:12b")
        return 1

    print(f"vision model: {analyzer.model}")
    for img in images:
        result = analyzer.analyze(img)
        report = result.report
        print(f"\n{img}  ({result.ms/1000:.1f}s)")
        print(f"  components: {len(report.components)}  risk: {report.overall_risk}/100")
        for f in report.findings:
            print(
                f"  - {f.severity:8} {f.component[:28]:28} "
                f"{f.mitre_technique:12} NIST {f.nist_control}"
            )
        if result.advisories:
            print(f"  advisories: {[a.cve_id or a.doc_id for a in result.advisories]}")
        if result.stripped_cves:
            print(f"  stripped ungrounded CVEs: {result.stripped_cves}")
        if report.notes:
            print(f"  notes: {report.notes[:160]}")

    # Second pass proves the cache is populated.
    again = analyzer.analyze(images[0])
    print(f"\ncached repeat: {again.ms/1000:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
