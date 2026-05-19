#!/usr/bin/env sh
set -eu

if [ ! -f config.yaml ] && [ -f config.example.yaml ]; then
  cp config.example.yaml config.yaml
fi

exec .venv/bin/streamlit run web/app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8501}" \
  --browser.gatherUsageStats false
