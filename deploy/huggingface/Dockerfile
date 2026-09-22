FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY plastic/ ./plastic/
RUN pip install --no-cache-dir .
COPY deploy/ ./deploy/
COPY text/ ./text/
COPY physics/ ./physics/
COPY --from=frontend /build/dist ./dashboard/dist
RUN useradd --uid 1000 --create-home demo
USER demo
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PORT=7860
EXPOSE 7860
CMD ["python", "-m", "deploy.huggingface.app"]
