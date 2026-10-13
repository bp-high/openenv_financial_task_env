# Finance Desk OpenEnv server. linux/amd64 only (the arena pins the amd64 image).
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Finance Desk"
LABEL org.opencontainers.image.description="Procedural multi-domain OpenEnv environment (deep finance + all eight arena domains)"
LABEL org.opencontainers.image.source="https://github.com/bp-high/openenv_financial_task_env"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# git is needed to install OpenEnv from its pinned commit; removed afterwards to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir \
        "openenv @ git+https://github.com/meta-pytorch/OpenEnv.git@86a180ede21e044f7929b9a7783ad83aa67d83a3" \
        "fastapi>=0.104.0" "uvicorn>=0.24.0" "pydantic>=2.0.0" "websockets>=12.0" \
        "numpy>=1.26.0" "pandas>=2.0.0" \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY fdesk /app/fdesk

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
EXPOSE 8000

# The server runs as root so the python tool can drop to `nobody` for the agent's code; the agent's
# subprocess cannot read /app (the task generators and answer keys live there).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "fdesk.server.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ws-ping-interval", "60", "--ws-ping-timeout", "60"]
