FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    STREAMLIT_SERVER_HEADLESS=true \
    IMPERIAL_RAG_WORKSPACE_ROOT=/app

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts

RUN uv sync --frozen --no-dev --no-cache

EXPOSE 8501

CMD ["uv", "run", "python", "-m", "streamlit", "run", "src/imperial_rag/web_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
