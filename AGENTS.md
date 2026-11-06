# Agent notes

## Environment

- Python 3.14 is required: `essentia` (dev-only, used by the mel parity test) ships macOS arm64 wheels for cp314 only.
- Benchmark scripts need the optional group: `uv run --group bench ...`. A plain `uv sync` removes torch/transformers again.
- `bench/` scripts import `metrics` as a sibling module: run them with `PYTHONPATH=bench`.
