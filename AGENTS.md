# Agent notes

## Environment

- Python 3.14 is required: `essentia` (dev-only, used by the mel parity test) ships macOS arm64 wheels for cp314 only.
- Benchmark scripts need the optional group: `uv run --group bench ...`. A plain `uv sync` removes torch/transformers again.
- `bench/` scripts import `metrics` as a sibling module: run them with `PYTHONPATH=bench`.
- `bench/fetch_youtube.py` needs a residential IP: from datacenter IPs YouTube answers "Sign in to confirm you're not a bot" for nearly every video. Needs only `uv run --no-dev --group fetch` (the `fetch` group bundles deno as yt-dlp's JS runtime).

Landing: merge into local `main`, push `origin/main`, then delete the branch and worktree. No PR.
