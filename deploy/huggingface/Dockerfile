FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
# Spaces dev mode appends `git config --global ...` build steps, which fail with exit 127 without git
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
# the default PyPI wheel carries CUDA and falls back to CPU where no GPU exists
RUN pip install --no-cache-dir torch==2.14.0
COPY pyproject.toml README.md LICENSE ./
COPY plastic/ ./plastic/
RUN pip install --no-cache-dir '.[pretrained]'
COPY deploy/ ./deploy/
ARG PUBLIC_MODEL=qwen
ENV PUBLIC_MODEL=${PUBLIC_MODEL}
RUN python -m deploy.huggingface.pretrained --download /opt/public-model
COPY text/ ./text/
COPY --from=frontend /build/dist ./dashboard/dist
RUN useradd --uid 1000 --create-home demo
USER demo
ENV PYTHONUNBUFFERED=1 PORT=7860 PUBLIC_CHECKPOINT=/opt/public-model
EXPOSE 7860
CMD ["python", "-m", "deploy.huggingface.app"]
