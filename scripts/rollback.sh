#!/bin/bash

set -e

ENVIRONMENT=${1:-staging}

echo "🔄 Rolling back deployment for $ENVIRONMENT environment..."

case $ENVIRONMENT in
  local)
    echo "⏹️  Stopping containers..."
    docker compose down
    echo "✅ Local rollback complete!"
    ;;
  
  staging)
    echo "⏹️  Restarting previous container..."
    docker compose pull
    docker compose up -d
    echo "✅ Staging rollback complete!"
    ;;
  
  k8s)
    echo "🔙 Rolling back Kubernetes deployment..."
    kubectl rollout undo deployment/tessera-api
    kubectl rollout status deployment/tessera-api --timeout=5m
    echo "✅ Kubernetes rollback complete!"
    echo "   Current replicas: $(kubectl get deployment tessera-api -o jsonpath='{.spec.replicas}')"
    ;;
  
  *)
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Usage: $0 {local|staging|k8s}"
    exit 1
    ;;
esac
