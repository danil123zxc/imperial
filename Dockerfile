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
