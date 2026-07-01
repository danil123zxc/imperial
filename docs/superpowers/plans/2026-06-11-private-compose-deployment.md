# Private Compose Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private single-machine Docker Compose deployment for Imperial RAG with Streamlit, Qdrant, Phoenix, and an explicit ingestion profile.

**Architecture:** Add a Docker image for the existing Streamlit app/runtime, then wire it into `compose.yaml` beside Qdrant and Phoenix. Keep private corpus data on the host by bind-mounting `documents/` read-only and `.imperial_rag/` read-write; use service DNS inside Compose for Qdrant and Phoenix while binding all host ports to `127.0.0.1`.

**Tech Stack:** Docker Compose v5, Dockerfile, uv, Python 3.12, Streamlit, Qdrant, Phoenix, pytest.

---

## Scope Check

This plan implements one subsystem: private Compose deployment. It does not add auth, TLS, reverse proxy, Postgres, a backend API split, Kubernetes, or automatic ingestion on app startup.

## File Structure

- Create `tests/test_private_compose_deployment.py`: static regression tests for the deployment files.
- Create `.dockerignore`: keeps private corpus data, local state, secrets, caches, and git metadata out of Docker build context.
- Create `Dockerfile`: builds the reusable app/ingest image from `pyproject.toml`, `uv.lock`, `src/`, and `scripts/`.
- Modify `compose.yaml`: adds `app`, `ingest`, private host bindings, service-DNS env overrides, bind mounts, and health checks.
- Modify `.env.example`: documents host-local defaults and Compose-internal overrides.
- Modify `README.md`: adds private Compose deployment commands and verification.

### Task 1: Add Static Tests For Docker Build Context

**Files:**
- Create: `tests/test_private_compose_deployment.py`

- [ ] **Step 1: Write failing tests for Docker build files**

Create `tests/test_private_compose_deployment.py` with:

```python
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _dockerignore_entries() -> set[str]:
    lines = _read(".dockerignore").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def test_dockerignore_excludes_private_and_generated_data() -> None:
    entries = _dockerignore_entries()

    assert ".env" in entries
    assert ".env.*" in entries
    assert "documents/" in entries
    assert ".imperial_rag/" in entries
    assert ".git/" in entries
    assert "__pycache__/" in entries
    assert ".pytest_cache/" in entries
    assert ".venv/" in entries


def test_dockerfile_builds_uv_streamlit_runtime() -> None:
    dockerfile = _read("Dockerfile")

    assert "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile
    assert '"streamlit", "run", "src/imperial_rag/web_app.py"' in dockerfile
    assert '"--server.address", "0.0.0.0"' in dockerfile
    assert '"--server.port", "8501"' in dockerfile
```

- [ ] **Step 2: Run tests and verify they fail for missing deployment files**

Run:

```bash
uv run python -m pytest tests/test_private_compose_deployment.py -q
```

Expected: FAIL because `.dockerignore` and `Dockerfile` do not exist yet.

### Task 2: Add Docker Build Context And App Image

**Files:**
- Create: `.dockerignore`
- Create: `Dockerfile`
- Test: `tests/test_private_compose_deployment.py`

- [ ] **Step 1: Create `.dockerignore`**

Create `.dockerignore` with:

```dockerignore
.env
.env.*
!.env.example

documents/
.imperial_rag/

.git/
.gitignore

__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/
.venv/
venv/

.DS_Store
.idea/
.vscode/

evals/outputs/
phoenix/
traces/
```

- [ ] **Step 2: Create `Dockerfile`**

Create `Dockerfile` with:

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    STREAMLIT_SERVER_HEADLESS=true

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts

RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["uv", "run", "python", "-m", "streamlit", "run", "src/imperial_rag/web_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
```

- [ ] **Step 3: Run the focused Docker build tests**

Run:

```bash
uv run python -m pytest \
  tests/test_private_compose_deployment.py::test_dockerignore_excludes_private_and_generated_data \
  tests/test_private_compose_deployment.py::test_dockerfile_builds_uv_streamlit_runtime \
  -q
```

Expected: PASS.

- [ ] **Step 4: Build the image directly**

Run:

```bash
docker build -t imperial-rag-app:test .
```

Expected: build completes successfully and the final output includes `naming to docker.io/library/imperial-rag-app:test` or an equivalent local image name.

- [ ] **Step 5: Commit build context, Dockerfile, and their tests**

Run:

```bash
git add tests/test_private_compose_deployment.py .dockerignore Dockerfile
git commit -m "build: add imperial rag app image"
```

### Task 3: Expand Compose To Runtime And Ingest Services

**Files:**
- Modify: `compose.yaml`
- Test: `tests/test_private_compose_deployment.py`

- [ ] **Step 1: Add a failing Compose service test**

Append this test to `tests/test_private_compose_deployment.py`:

```python


def test_compose_defines_private_app_and_ingest_services() -> None:
    compose = _read("compose.yaml")

    required_snippets = [
        "x-imperial-app-base:",
        "app:",
        "ingest:",
        'profiles: ["ingest"]',
        '"127.0.0.1:8501:8501"',
        '"127.0.0.1:6006:6006"',
        '"127.0.0.1:4317:4317"',
        '"127.0.0.1:6333:6333"',
        "QDRANT_URL: http://qdrant:6333",
        "PHOENIX_CLIENT_ENDPOINT: http://phoenix:6006",
        "PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006/v1/traces",
        "./documents:/app/documents:ro",
        "./.imperial_rag:/app/.imperial_rag",
        "scripts/ingest.py",
        "--index-vectors",
    ]

    for snippet in required_snippets:
        assert snippet in compose
```

- [ ] **Step 2: Run the Compose test and verify it fails**

Run:

```bash
uv run python -m pytest tests/test_private_compose_deployment.py::test_compose_defines_private_app_and_ingest_services -q
```

Expected: FAIL because `compose.yaml` does not define `app`, `ingest`, or Compose-internal endpoints yet.

- [ ] **Step 3: Replace `compose.yaml`**

Replace `compose.yaml` with:

```yaml
x-imperial-app-base: &imperial-app-base
  build:
    context: .
    dockerfile: Dockerfile
  image: imperial-rag-app:local
  env_file:
    - .env
  environment:
    IMPERIAL_RAG_WORKSPACE_ROOT: /app
    QDRANT_URL: http://qdrant:6333
    QDRANT_COLLECTION: imperial_chunks_qwen
    PHOENIX_CLIENT_ENDPOINT: http://phoenix:6006
    PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006/v1/traces
  volumes:
    - ./documents:/app/documents:ro
    - ./.imperial_rag:/app/.imperial_rag
  depends_on:
    qdrant:
      condition: service_healthy
    phoenix:
      condition: service_started

services:
  app:
    <<: *imperial-app-base
    ports:
      - "127.0.0.1:8501:8501"
    command:
      - uv
      - run
      - python
      - -m
      - streamlit
      - run
      - src/imperial_rag/web_app.py
      - --server.address
      - 0.0.0.0
      - --server.port
      - "8501"
      - --server.headless
      - "true"
    healthcheck:
      test:
        - CMD
        - /app/.venv/bin/python
        - -c
        - "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read()"
      interval: 10s
      timeout: 3s
      retries: 12
      start_period: 20s
    restart: unless-stopped

  ingest:
    <<: *imperial-app-base
    profiles: ["ingest"]
    command:
      - uv
      - run
      - python
      - scripts/ingest.py
      - --workspace-root
      - /app
      - --index-vectors
    restart: "no"

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "127.0.0.1:6006:6006"
      - "127.0.0.1:4317:4317"
    environment:
      PHOENIX_WORKING_DIR: /mnt/data
    volumes:
      - phoenix_data:/mnt/data
    healthcheck:
      test:
        - CMD
        - /usr/bin/python3.13
        - -c
        - "import urllib.request; urllib.request.urlopen('http://127.0.0.1:6006/', timeout=2).read(1)"
      interval: 10s
      timeout: 3s
      retries: 12
      start_period: 20s

  qdrant:
    image: qdrant/qdrant:latest
    container_name: imperial-qdrant
    ports:
      - "127.0.0.1:6333:6333"
    volumes:
      - ./.imperial_rag/qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "bash -lc ': > /dev/tcp/127.0.0.1/6333'"]
      interval: 10s
      timeout: 3s
      retries: 12
      start_period: 10s

volumes:
  phoenix_data:
    driver: local
```

- [ ] **Step 4: Run focused Compose tests**

Run:

```bash
uv run python -m pytest tests/test_private_compose_deployment.py -q
```

Expected: PASS.

- [ ] **Step 5: Validate Compose config for normal runtime**

Run:

```bash
test -f .env || cp .env.example .env
docker compose config
```

Expected: command exits 0 and the rendered config includes `app`, `qdrant`, and `phoenix`.

- [ ] **Step 6: Validate Compose config for ingestion profile**

Run:

```bash
docker compose --profile ingest config
```

Expected: command exits 0 and the rendered config includes `ingest`.

- [ ] **Step 7: Commit Compose expansion and its test**

Run:

```bash
git add compose.yaml tests/test_private_compose_deployment.py
git commit -m "chore: add private compose app stack"
```

### Task 4: Document Compose Environment And Operator Commands

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_private_compose_deployment.py`

- [ ] **Step 1: Add failing docs and env tests**

Append these tests to `tests/test_private_compose_deployment.py`:

```python


def test_env_example_documents_compose_overrides() -> None:
    env_example = _read(".env.example")

    assert "Compose container overrides" in env_example
    assert "QDRANT_URL=http://qdrant:6333" in env_example
    assert "PHOENIX_CLIENT_ENDPOINT=http://phoenix:6006" in env_example
    assert "PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces" in env_example


def test_readme_documents_private_compose_deployment() -> None:
    readme = _read("README.md")

    assert "## Private Compose Deployment" in readme
    assert "docker compose up -d qdrant phoenix app" in readme
    assert "docker compose --profile ingest up ingest" in readme
    assert "http://127.0.0.1:8501/_stcore/health" in readme
```

- [ ] **Step 2: Run docs and env tests and verify they fail**

Run:

```bash
uv run python -m pytest \
  tests/test_private_compose_deployment.py::test_env_example_documents_compose_overrides \
  tests/test_private_compose_deployment.py::test_readme_documents_private_compose_deployment \
  -q
```

Expected: FAIL because `.env.example` and `README.md` do not document the private Compose deployment yet.

- [ ] **Step 3: Add Compose notes to `.env.example`**

In `.env.example`, keep the existing host-local values unchanged, then add this block immediately after `PHOENIX_TRACING_ENABLED=false` and `IMPERIAL_RAG_TRACING_ENABLED=false`:

```dotenv
# Compose container overrides
# compose.yaml sets these inside app and ingest containers so host-local commands can keep localhost defaults above.
# IMPERIAL_RAG_WORKSPACE_ROOT=/app
# QDRANT_URL=http://qdrant:6333
# PHOENIX_CLIENT_ENDPOINT=http://phoenix:6006
# PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces
```

- [ ] **Step 4: Add README deployment section**

In `README.md`, add this section after the existing local UI instructions and before `## Local Services`:

````markdown
## Private Compose Deployment

The private Compose stack runs the Streamlit app, Qdrant, and Phoenix on one machine with all published ports bound to `127.0.0.1`.

Prepare the server checkout:

```bash
cp .env.example .env
mkdir -p documents .imperial_rag/qdrant_storage
```

Fill `.env` with `DASHSCOPE_API_KEY` and any provider settings needed for the deployed machine. The Compose file overrides service endpoints inside the app containers, so host-local commands can keep the `localhost` defaults from `.env.example`.

Start the runtime stack:

```bash
docker compose up -d qdrant phoenix app
```

Verify the private endpoints from the host:

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:6333/healthz
curl -I --max-time 3 http://127.0.0.1:6006/
```

Open the app through the local machine or an SSH tunnel:

```text
http://127.0.0.1:8501
```

Run ingestion explicitly when documents change:

```bash
docker compose --profile ingest up ingest
```

Inspect logs:

```bash
docker compose logs -f app
docker compose logs -f ingest
```

Stop the stack:

```bash
docker compose down
```
````

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run python -m pytest tests/test_private_compose_deployment.py -q
```

Expected: PASS.

- [ ] **Step 6: Run existing related tests**

Run:

```bash
uv run python -m pytest tests/test_config.py tests/test_web_app.py tests/test_private_compose_deployment.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit docs, env guidance, and their tests**

Run:

```bash
git add .env.example README.md tests/test_private_compose_deployment.py
git commit -m "docs: document private compose deployment"
```

### Task 5: Live Compose Verification

**Files:**
- Modify if verification exposes a concrete issue: `Dockerfile`, `compose.yaml`, `.dockerignore`, `.env.example`, `README.md`, or `tests/test_private_compose_deployment.py`

- [ ] **Step 1: Prepare local files and inspect worktree before live verification**

Run:

```bash
test -f .env || cp .env.example .env
mkdir -p documents .imperial_rag/qdrant_storage
git status --short
```

Expected: no unstaged changes from previous tasks, except pre-existing user changes that are unrelated to this deployment work. The `.env`, `documents/`, and `.imperial_rag/` paths are ignored by git.

- [ ] **Step 2: Build the Compose app image**

Run:

```bash
docker compose build app
```

Expected: command exits 0 and builds `imperial-rag-app:local`.

- [ ] **Step 3: Start the private runtime stack**

Run:

```bash
docker compose up -d qdrant phoenix app
```

Expected: command exits 0 and creates or starts `imperial-qdrant`, `imperial-phoenix-1`, and `imperial-app-1`.

- [ ] **Step 4: Verify containers are healthy or running**

Run:

```bash
docker compose ps
```

Expected: `app` is `healthy`, `qdrant` is `healthy`, and `phoenix` is either `healthy` or running while its HTTP endpoint is reachable in the next step.

- [ ] **Step 5: Verify private host endpoints**

Run:

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:6333/healthz
curl -I --max-time 3 http://127.0.0.1:6006/
```

Expected:

```text
ok
healthz check passed
HTTP/1.1 200 OK
```

The Qdrant line may include JSON or plain text depending on the image version; the command must exit 0.

- [ ] **Step 6: Validate ingestion profile config without running paid/provider work**

Run:

```bash
docker compose --profile ingest config
```

Expected: command exits 0 and includes the `ingest` service with command `scripts/ingest.py --workspace-root /app --index-vectors`.

- [ ] **Step 7: Run the focused tests again**

Run:

```bash
uv run python -m pytest tests/test_config.py tests/test_web_app.py tests/test_private_compose_deployment.py -q
```

Expected: PASS.

- [ ] **Step 8: Fix any verification issue with the narrowest patch**

If a command fails, inspect the relevant logs:

```bash
docker compose logs --tail=120 app
docker compose logs --tail=120 qdrant
docker compose logs --tail=120 phoenix
```

Apply the smallest change to the file named by the failure, then rerun the failing command and the focused tests from Step 7.

- [ ] **Step 9: Commit verification fixes if any were needed**

If Step 8 changed files, run:

```bash
git add Dockerfile compose.yaml .dockerignore .env.example README.md tests/test_private_compose_deployment.py
git commit -m "fix: verify private compose stack"
```

If Step 8 made no changes, skip this commit.

- [ ] **Step 10: Report final status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing user changes remain.

Summarize:

- commits created for the deployment work
- test commands run
- Docker commands run
- final app URL: `http://127.0.0.1:8501`
- whether ingestion was configured only or actually executed
