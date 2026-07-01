#!/usr/bin/env bash
set -euo pipefail

# Local-only Elasticsearch for private keyword search. Do not expose this service publicly.
docker compose up elasticsearch
