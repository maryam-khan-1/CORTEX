# CORTEX

Grounded, offline-first blue-team assistant on Gemma 4.

## Setup

```bash
cd cortex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# One-time (or after clone): vendor MiniLM weights for fully offline embeddings
python scripts/vendor_embeddings.py
# Build-time only (network): fetch KEV/OSV and persist local Chroma index
# (index is gitignored — required after clone; ~few minutes)
python scripts/build_index.py
```

Embeddings then load from `data/models/all-MiniLM-L6-v2` with `HF_HUB_OFFLINE=1` — no Hub API at demo time.

## Run

```bash
cd cortex
PYTHONPATH=. .venv/bin/python app.py
```

## Smoke tests

```bash
cd cortex
PYTHONPATH=. .venv/bin/python tests/smoke/test_schema.py
PYTHONPATH=. .venv/bin/python tests/smoke/test_engine.py
PYTHONPATH=. .venv/bin/python tests/smoke/test_harness.py
PYTHONPATH=. .venv/bin/python tests/smoke/test_rag.py
PYTHONPATH=. .venv/bin/python tests/smoke/test_triage.py
PYTHONPATH=. .venv/bin/python tests/smoke/test_app.py
```

No cloud calls at demo time. Consensus labels are agreement counts, not confidence intervals.

## Modes

- **Live SOC** — animated real-time threat feed, severity color coding, hover details, charts
- **Demo** — one-click walkthrough (triage → grounded CVE → abstention → agent tools)
- **Log triage** — E2B classify (3× consensus) → SecOps deep on flagged lines
- **Code analyzer** — harness + RAG grounded report
- **CVE / grounded report** — post-cutoff evidence only
- **Agent** — autonomous defend (one click); Gemma 4 native tools (`search_cve`, `lookup_kev`, `get_code_context`)
