FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

EXPOSE 8900

ENTRYPOINT ["jev-bridge", "serve", "--host", "0.0.0.0", "--port", "8900"]
