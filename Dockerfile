# Also place Dockerfile at repo root (required by validation)
FROM python:3.11-slim

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

COPY pyproject.toml /app/env/

RUN uv pip install --system --no-cache-dir \
    "openenv-core>=0.2.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0.0" \
    "websockets>=12.0" \
    "openpyxl>=3.1.0"

COPY . /app/env/

ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
