#!/usr/bin/env bash
# Entry point for a full agent run.
# Milestone 0 (harness proof): policy.kind=random in configs/agent.yaml exercises the
# whole loop on garbage models. Add --max-iterations N --timeout S to keep it short.
set -euo pipefail
cd "$(dirname "$0")"

# uv is the project's dependency manager but is not always on PATH (pip --user install).
UV="$(command -v uv || echo "$HOME/Library/Python/3.12/bin/uv")"
if [ -x "$UV" ]; then
  "$UV" sync --quiet
  exec "$UV" run python -m agent.loop \
    --agent-config configs/agent.yaml --data-config configs/data.yaml "$@"
fi

echo "uv not found; falling back to system python3" >&2
exec python3 -m agent.loop \
  --agent-config configs/agent.yaml --data-config configs/data.yaml "$@"
