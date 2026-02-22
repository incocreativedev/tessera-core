#!/bin/bash

set -e

HOST=${1:-localhost}
PORT=${2:-8080}
TIMEOUT=${3:-30}

HEALTH_ENDPOINT="http://$HOST:$PORT/health"

echo "🏥 Checking health of tessera-api at $HEALTH_ENDPOINT..."

for i in $(seq 1 $TIMEOUT); do
  if curl -sf "$HEALTH_ENDPOINT" > /dev/null 2>&1; then
    echo "✅ Service is healthy!"
    exit 0
  fi
  echo "  Attempt $i/$TIMEOUT..."
  sleep 1
done

echo "❌ Service failed to become healthy within $TIMEOUT seconds"
exit 1
