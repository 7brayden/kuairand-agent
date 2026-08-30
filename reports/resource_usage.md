# Resource usage

> GENERATED from `logs/journal.jsonl` by `scripts/generate_reports.py` — do not edit by hand.


| | |
|---|---|
| iterations attempted | 4 |
| LLM tokens in | 22,491 |
| LLM tokens out | 45,359 |
| **LLM tokens total** | **67,850** |
| **GPU-hours** | **0.0000** |
| wall-clock hours | 0.1535 |

Token counts come from `agent/llm.py`, the single provider call site, and include failed calls, retries, and calls made inside rejected iterations.

GPU-hours are 0 by design: the default inner model is LightGBM lambdarank on CPU.

