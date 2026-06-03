#!/usr/bin/env bash
set -euo pipefail

mkdir -p .imperial_rag/qdrant_storage

# Local-only Qdrant for private development. Do not expose this service publicly.
docker run \
  --name imperial-qdrant \
  --rm \
  -p 127.0.0.1:6333:6333 \
  -v "$(pwd)/.imperial_rag/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
