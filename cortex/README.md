# CORTEX

Grounded, offline-first blue-team assistant on Gemma 4.

Gemma 4's knowledge ends **January 2025**, so every threat a SOC cares about today is
out-of-distribution. CORTEX grounds every CVE claim in a local post-cutoff CVE/KEV index,
**abstains** when the evidence isn't there, and defends **continuously without being asked**.

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

Embeddings then load from `data/models/all-MiniLM-L6-v2` with `HF_HUB_OFFLINE=1` — no Hub API
at demo time, and the stylesheet ships no webfonts, so the UI renders with the network unplugged.

## Run

```bash
cd cortex
PYTHONPATH=. .venv/bin/python app.py
```

Open the **Autonomy** tab and press **Engage autonomous defense**. Nothing else is required —
no prompt, no goal box.

## The autonomy loop

One background thread runs an OODA cycle over the live event stream:

| Phase | What happens | Model |
|---|---|---|
| observe | pull the next event off the stream | — |
| orient | stage-1 classify with early-exit consensus | E2B fast |
| decide | escalate only what earns it; fold known patterns | — |
| act | agent investigation via native tool calling | SecOps deep |
| reflect | critique the report; reopen it if a claim is ungrounded | deep |
| learn | remember the signature so repeats are never re-paid | — |

The loop never blocks the UI; Gradio reads a snapshot. A bad cycle is caught and logged
rather than killing the defender.

## Latency levers (all offline)

Measured on an M1 laptop: fast model ~27 tok/s, deep model ~15 tok/s, so **generated
tokens are the cost** and every lever below is about generating fewer of them.

- **`think=False`** (biggest win) — these Gemma 4 builds are reasoning models. Left on, the
  whole `num_predict` budget is spent on `message.thinking` and `content` comes back
  **empty**, which is both the slowest path and the most dangerous: a classifier falling
  back to its default label silently calls a critical alert benign. Disabling reasoning took
  a deep structured report from **~49s to ~8s** and made stage-1 labels correct at all.
- **Content-addressed disk cache** — repeated lines and replayed demos skip inference
  entirely. Re-triaging the sample log set: **121s → 0.09s (~1400x)**.
- **Bounded agent budgets** — live alerts get 3 steps and a terse token cap, and a prose
  reply triggers an immediately constrained closing turn instead of another free-form
  round. Worst-case investigation **44s → 28s**; typical **12–18s**.
- **Early-exit consensus** — stop voting once the winner is mathematically locked.
  Spends 20 votes instead of 30 on the sample set.
- **Batched vote rounds** — one call classifies the whole round instead of one call per
  line, since Ollama serializes requests by default (~20% off bulk triage).
- **Grammar-constrained JSON** via Ollama `format` — the first response is already
  schema-valid, so retry round-trips almost never fire.
- **Hard context caps + `keep_alive`** so the 128K KV is never allocated and weights stay resident.

Honest ceiling: cold bulk triage still costs roughly **10–12s per log line** on this
hardware — that is the floor for a local 2B/4B model, not something caching hides. The
autonomy loop is the fast path because it handles one event at a time (**1–4s** triage,
**12–28s** investigation) and folds repeats instead of re-analyzing them.

Every number in the UI's telemetry strip is measured in-process; nothing leaves the host.
If you want true request parallelism, start the server with `OLLAMA_NUM_PARALLEL=3`.

## Honesty properties

- `insufficient_evidence` is a first-class verdict, not an error path.
- `apply_grounding_rule` is a **post-hoc validator**: a finding naming a CVE with no
  retrieved doc has its `cve_id` stripped and its verdict forced to `insufficient_evidence`.
- Consensus labels are **agreement counts, not confidence intervals**.
- An unparseable stage-1 vote becomes `unknown` and escalates — it never becomes `benign`.
- `attack_path` explains abuse defensively; never working exploit code.

## Modes

- **Autonomy** — continuous self-driving defense: phase rail, incident cards, reasoning trace, telemetry
- **Live SOC** — animated threat feed, severity colors, hover detail, charts
- **Guided demo** — one-click walkthrough (triage → grounded CVE → abstention → autonomous agent)
- **Log triage** — E2B classify (early-exit consensus) → SecOps deep on flagged lines
- **Code analyzer** — harness + RAG grounded report
- **Grounded CVE** — post-cutoff evidence only
- **Agent** — one-click defend; native tools (`search_cve`, `lookup_kev`, `check_dependency`, `get_code_context`)

## Smoke tests

```bash
cd cortex
for t in schema harness offline latency autonomy triage live_feed engine rag agent build_index app demo dashboard; do
  PYTHONPATH=. .venv/bin/python tests/smoke/test_$t.py || echo "FAILED: $t"
done
```

`latency` and `autonomy` run against stubs, so they stay instant and need no model.

## Tuning

All knobs live in `config.json`: `performance.think`, `performance.num_predict_*`,
`harness.constrain_json`, `triage.early_exit` / `max_workers` / `max_deep`,
`agent.alert_max_steps` / `num_predict_terse` / `self_critique`, `autonomy.interval_s`
and the escalation label sets, and `cache.enabled`.
