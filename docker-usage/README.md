# NFL Betting System - Docker Setup

## 🏠 House Cleaning & Deployment

This Docker setup provides clean, isolated environments for development, production, and ML training.

## 📁 Files Overview

### **Development & Production**
- `docker-compose.local.yml` - Local development environment
- `docker-compose.prod.yml` - Production deployment
- `Dockerfile.frontend.dev` - Next.js development container
- `Dockerfile.backend.dev` - Python API development container
- `Dockerfile.frontend.prod` - Next.js production container
- `Dockerfile.backend.prod` - Python API production container

### **ML Training**
- `docker-compose.ml.yml` - ML training environment
- `Dockerfile.ml` - ML training container with Jupyter Lab

### **Scripts**
- `start-local.sh` - Start local development
- `start-prod.sh` - Start production deployment
- `start-ml.sh` - Start ML training environment

## 🚀 Quick Start

### **1. Local Development**
```bash
cd docker-usage
./start-local.sh
```
**Access:**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Database: localhost:5432
- Redis: localhost:6379

### **2. Production Deployment**
```bash
cd docker-usage
./start-prod.sh
```
**Features:**
- Optimized builds
- Health checks
- Auto-restart
- Production-ready configuration

### **3. ML Training**
```bash
cd docker-usage
./start-ml.sh
```
**Access:**
- Jupyter Lab: http://localhost:8888
- Data: ../ml_training_data
- Models: ../saved_models

## 🧹 House Cleaning Benefits

### **Development Environment**
- ✅ Consistent across all machines
- ✅ No "works on my machine" issues
- ✅ Easy onboarding for new developers
- ✅ Isolated from system dependencies

### **Production Deployment**
- ✅ Single command deployment
- ✅ Consistent production environment
- ✅ Easy rollbacks
- ✅ Horizontal scaling support

### **ML Training**
- ✅ Isolated ML environment
- ✅ Reproducible experiments
- ✅ Easy model versioning
- ✅ Collaborative Jupyter Lab

## 🔧 Manual Commands

### **Local Development**
```bash
# Start services
docker-compose -f docker-compose.local.yml up -d

# View logs
docker-compose -f docker-compose.local.yml logs -f

# Stop services
docker-compose -f docker-compose.local.yml down

# Restart specific service
docker-compose -f docker-compose.local.yml restart frontend
```

### **Production**
```bash
# Start services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down

# Health check
docker-compose -f docker-compose.prod.yml ps
```

### **ML Training**
```bash
# Start ML services
docker-compose -f docker-compose.ml.yml up -d

# Access Jupyter Lab
open http://localhost:8888

# Run data processing
docker-compose -f docker-compose.ml.yml exec ml-data-processor python -c "from src.data_processor import main; main()"

# Run model training
docker-compose -f docker-compose.ml.yml exec ml-pipeline python -c "from src.model_trainer import main; main()"
```

## 🎯 Use Cases

### **1. House Cleaning**
- **Problem**: Inconsistent development environments
- **Solution**: Docker containers with identical setups
- **Result**: Same behavior everywhere

### **2. Deployment Cleaning**
- **Problem**: Complex deployment process
- **Solution**: Single `docker-compose up` command
- **Result**: Deploy anywhere in minutes

### **3. ML Training**
- **Problem**: ML environment setup complexity
- **Solution**: Pre-configured ML containers
- **Result**: Focus on models, not infrastructure

## 💰 Cost Benefits

### **Current Setup**
- Vercel: $20/month
- Python hosting: $10-20/month
- Database: $10-15/month
- **Total**: $40-55/month

### **Docker Setup**
- Single VPS: $10-20/month
- Managed database: $10-15/month
- **Total**: $20-35/month
- **Savings**: 40-50%

## 🔍 Troubleshooting

### **Port Conflicts**
```bash
# Check what's using ports
lsof -i :3000 -i :8000 -i :5432 -i :6379

# Stop conflicting services
pkill -f "next dev"
pkill -f "simple_api.py"
```

### **Container Issues**
```bash
# View container logs
docker-compose -f docker-compose.local.yml logs frontend

# Restart container
docker-compose -f docker-compose.local.yml restart frontend

# Rebuild container
docker-compose -f docker-compose.local.yml build frontend
```

### **Database Issues**
```bash
# Reset database
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up -d
```

## 🎯 Next Steps

1. **Test Local Development**: Run `./start-local.sh`
2. **Test Production**: Run `./start-prod.sh`
3. **Test ML Training**: Run `./start-ml.sh`
4. **Deploy to Cloud**: Use production config on VPS
5. **Scale Services**: Add load balancers and multiple instances

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Next.js Docker Deployment](https://nextjs.org/docs/deployment#docker-image)
- [Flask Docker Deployment](https://flask.palletsprojects.com/en/2.0.x/deploying/docker/)
- [Jupyter Lab Docker](https://jupyter-docker-stacks.readthedocs.io/)
