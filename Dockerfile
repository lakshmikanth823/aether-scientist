# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY aether_scientist/ aether_scientist/
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME=/models
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["aether"]
CMD ["--help"]
