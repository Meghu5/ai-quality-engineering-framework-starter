# Phase 10 Promptfoo Evaluation

Promptfoo is optional and isolated from the Python baseline.

Source-of-truth test cases remain in `data/ai/golden_cases.json` and
`data/rag/rag_cases.json`. Use `build_promptfoo_config.py` to generate a
Promptfoo-compatible config from those datasets.

No provider secrets are required for the deterministic-local provider.
