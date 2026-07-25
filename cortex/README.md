# CORTEX

Grounded, offline-first blue-team assistant on Gemma 4.

## Setup

```bash
cd cortex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Build-time only (network): fetch KEV/OSV and persist local Chroma index
# (index is gitignored — required after clone; ~few minutes)
python scripts/build_index.py
```

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
