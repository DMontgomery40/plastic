FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
# the default PyPI wheel carries CUDA and falls back to CPU where no GPU exists
RUN pip install --no-cache-dir torch==2.14.0
COPY pyproject.toml README.md LICENSE ./
COPY plastic/ ./plastic/
RUN pip install --no-cache-dir '.[pretrained]'
COPY deploy/ ./deploy/
RUN python -m deploy.huggingface.pretrained --download /opt/qwen
COPY text/ ./text/
COPY --from=frontend /build/dist ./dashboard/dist
RUN useradd --uid 1000 --create-home demo
USER demo
ENV PYTHONUNBUFFERED=1 PORT=7860 QWEN_CHECKPOINT=/opt/qwen
EXPOSE 7860
CMD ["python", "-m", "deploy.huggingface.app"]
