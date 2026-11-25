FROM python:3.12-slim

WORKDIR /app
ENV UV_NO_DEV=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY . /app
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.10,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-c", "import buglens; print('BugLens container ready')"]
