FROM python:3.12-slim

WORKDIR /app

# Install deps first (cache layer)
COPY pyproject.toml README.md ./
COPY tessera/ ./tessera/

RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[railway]"

# Copy rest of source
COPY . .

EXPOSE 8080

CMD uvicorn tessera.api:app --host 0.0.0.0 --port ${PORT:-8080}
