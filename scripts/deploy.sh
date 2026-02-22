#!/bin/bash

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "🚀 Starting deployment for $ENVIRONMENT environment (version: $VERSION)"

case $ENVIRONMENT in
  local)
    echo "📦 Building Docker image locally..."
    docker build -t tessera-core:$VERSION .
    echo "🐳 Starting services with docker-compose..."
    docker compose up -d
    echo "✅ Local deployment complete!"
    echo "   API available at http://localhost:8080"
    ;;
  
  staging)
    echo "📦 Pulling image from registry..."
    docker pull ghcr.io/incocreativedev/tessera-core:$VERSION
    echo "🐳 Starting services with docker-compose..."
    docker compose up -d
    echo "✅ Staging deployment complete!"
    ;;
  
  k8s)
    echo "📦 Deploying to Kubernetes cluster..."
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/ingress.yaml
    echo "⏳ Waiting for rollout..."
    kubectl rollout status deployment/tessera-api --timeout=5m
    echo "✅ Kubernetes deployment complete!"
    echo "   Service: tessera-api"
    echo "   Replicas: $(kubectl get deployment tessera-api -o jsonpath='{.spec.replicas}')"
    ;;
  
  *)
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Usage: $0 {local|staging|k8s} [version]"
    exit 1
    ;;
esac
