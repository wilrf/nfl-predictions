# NFL Betting Model Deployment Guide

## Overview

This guide covers the deployment of the NFL Betting Model API using Docker and Kubernetes. The system includes:
- Two-stage ensemble ML model with LightGBM and XGBoost
- FastAPI-based REST API
- Redis caching layer
- PostgreSQL database
- Prometheus/Grafana monitoring
- Online learning capabilities
- Drift detection and monitoring

## Prerequisites

- Docker 20.10+
- Kubernetes 1.24+ (for K8s deployment)
- Helm 3.0+ (optional)
- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- 20GB disk space

## Local Development Deployment

### 1. Build and Run with Docker Compose

```bash
# Navigate to project root
cd /path/to/project

# Build and start all services
docker-compose -f deployment/docker-compose.yml up --build

# Or run in detached mode
docker-compose -f deployment/docker-compose.yml up -d --build
```

### 2. Verify Services

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Production Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace production
```

### 2. Deploy Configuration

```bash
# Apply ConfigMap and Secrets
kubectl apply -f deployment/k8s-config.yaml

# Deploy Redis
kubectl apply -f deployment/k8s-redis.yaml

# Deploy main application
kubectl apply -f deployment/k8s-deployment.yaml
```

### 3. Check Deployment Status

```bash
# Check pods
kubectl get pods -n production

# Check services
kubectl get svc -n production

# Check ingress
kubectl get ingress -n production

# View logs
kubectl logs -f deployment/nfl-betting-model -n production
```

### 4. Scale Deployment

```bash
# Manual scaling
kubectl scale deployment nfl-betting-model --replicas=5 -n production

# Check HPA status
kubectl get hpa -n production
```

## Docker Image Management

### Build Image

```bash
# Build production image
docker build -t nfl-betting-model:latest -f deployment/Dockerfile .

# Tag for registry
docker tag nfl-betting-model:latest your-registry/nfl-betting-model:v1.0.0

# Push to registry
docker push your-registry/nfl-betting-model:v1.0.0
```

### Multi-Architecture Build

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry/nfl-betting-model:latest \
  --push \
  -f deployment/Dockerfile .
```

## Model Management

### Update Model

```bash
# Copy new model to container
kubectl cp models/nfl_ensemble model-pod:/app/models/ -n production

# Trigger model reload
curl -X POST http://api.nfl-betting-model.com/update \
  -H "Content-Type: application/json" \
  -d '{"trigger_full_retrain": false}'
```

### Monitor Model Performance

```bash
# Get monitoring status
curl http://api.nfl-betting-model.com/monitor/status

# Generate performance report
curl -X POST http://api.nfl-betting-model.com/monitor/report
```

## Database Management

### Initialize Database

```bash
# Connect to PostgreSQL
kubectl exec -it postgres-pod -n production -- psql -U nfl_user -d nfl_betting

# Run initialization script
kubectl exec -it postgres-pod -n production -- psql -U nfl_user -d nfl_betting -f /init.sql
```

### Backup Database

```bash
# Create backup
kubectl exec postgres-pod -n production -- pg_dump -U nfl_user nfl_betting > backup.sql

# Restore backup
kubectl exec -i postgres-pod -n production -- psql -U nfl_user nfl_betting < backup.sql
```

## Monitoring Setup

### Configure Prometheus

1. Edit `prometheus.yml` with your targets
2. Apply configuration:
```bash
kubectl create configmap prometheus-config --from-file=deployment/prometheus.yml -n production
```

### Setup Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Add Prometheus data source: http://prometheus:9090
3. Import dashboards from `deployment/grafana/dashboards/`

### Key Metrics to Monitor

- **API Metrics**:
  - Request rate: `rate(predictions_total[5m])`
  - Latency: `histogram_quantile(0.95, prediction_duration_seconds)`
  - Error rate: `rate(http_requests_total{status=~"5.."}[5m])`

- **Model Metrics**:
  - Accuracy: `model_accuracy`
  - Drift score: `feature_drift_psi_score`
  - ROI: `betting_roi`

- **System Metrics**:
  - CPU usage: `container_cpu_usage_seconds_total`
  - Memory: `container_memory_usage_bytes`
  - Disk I/O: `container_fs_reads_bytes_total`

## Security Best Practices

### 1. Secrets Management

```bash
# Create secrets
kubectl create secret generic nfl-model-secrets \
  --from-literal=database-url='postgresql://user:pass@db/nfl' \
  --from-literal=api-key='your-api-key' \
  -n production
```

### 2. Network Policies

```yaml
# Apply network policy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nfl-model-netpol
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: nfl-betting-model
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: production
    ports:
    - protocol: TCP
      port: 8000
EOF
```

### 3. RBAC Configuration

```yaml
# Create service account and role
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: nfl-model-sa
  namespace: production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: nfl-model-role
  namespace: production
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
EOF
```

## Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**
```bash
# Check logs
kubectl logs -f pod-name -n production --previous

# Describe pod
kubectl describe pod pod-name -n production
```

2. **Model Loading Failure**
```bash
# Check model file exists
kubectl exec pod-name -n production -- ls -la /app/models

# Check permissions
kubectl exec pod-name -n production -- stat /app/models
```

3. **Redis Connection Issues**
```bash
# Test Redis connection
kubectl exec pod-name -n production -- redis-cli -h redis-service ping
```

4. **High Memory Usage**
```bash
# Check memory limits
kubectl top pods -n production

# Increase limits if needed
kubectl set resources deployment nfl-betting-model --limits=memory=8Gi -n production
```

### Performance Optimization

1. **Enable ONNX Runtime**
```python
# Convert model to ONNX for faster inference
python -m tf2onnx.convert --saved-model models/nfl_ensemble --output models/model.onnx
```

2. **Cache Warming**
```bash
# Preload cache with common predictions
python scripts/warm_cache.py
```

3. **Database Indexing**
```sql
-- Add indexes for better query performance
CREATE INDEX CONCURRENTLY idx_predictions_composite 
ON predictions(game_id, prediction_time DESC);
```

## CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy NFL Model

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'src/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build and Push Docker Image
      env:
        REGISTRY: ${{ secrets.DOCKER_REGISTRY }}
      run: |
        docker build -t $REGISTRY/nfl-model:${{ github.sha }} .
        docker push $REGISTRY/nfl-model:${{ github.sha }}
    
    - name: Deploy to Kubernetes
      env:
        KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
      run: |
        echo "$KUBE_CONFIG" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/nfl-betting-model \
          api=$REGISTRY/nfl-model:${{ github.sha }} \
          -n production
```

## Load Testing

### Using Locust

```python
# locustfile.py
from locust import HttpUser, task, between

class NFLAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        self.client.post("/predict", json={
            "games": [{
                "game_id": "test_game",
                "team_home": "BUF",
                "team_away": "KC",
                # ... other features
            }]
        })
    
    @task
    def health(self):
        self.client.get("/health")
```

Run load test:
```bash
locust -f locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10
```

## Maintenance

### Regular Tasks

- **Daily**:
  - Check monitoring dashboards
  - Review error logs
  - Verify model accuracy metrics

- **Weekly**:
  - Update online learning models
  - Review drift detection reports
  - Database vacuum and analyze

- **Monthly**:
  - Full model retraining
  - Performance optimization review
  - Security patches and updates

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups

# Backup database
pg_dump -h postgres -U nfl_user nfl_betting > $BACKUP_DIR/db_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /app/models

# Backup configurations
kubectl get configmap -n production -o yaml > $BACKUP_DIR/configs_$DATE.yaml

# Upload to S3
aws s3 cp $BACKUP_DIR s3://nfl-backups/$DATE/ --recursive
```

## Support

For issues or questions:
1. Check logs: `kubectl logs -f deployment/nfl-betting-model -n production`
2. Review metrics: http://localhost:3000 (Grafana)
3. API documentation: http://localhost:8000/docs
4. Contact: nfl-model-support@example.com

## License

Copyright 2024 - NFL Betting Model System
