FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 系统依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# 用 PyPI 安装 uv（避免构建时拉取 ghcr.io/astral-sh/uv）
RUN pip install --no-cache-dir uv

# 先装依赖，利用 Docker 缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 再复制业务代码
COPY . .

RUN mkdir -p /app/data /app/data/upload /app/result

EXPOSE 5557 7860

# 默认启动 API；UI 由 docker-compose 覆盖 command
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5557"]
