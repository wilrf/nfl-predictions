#!/bin/bash

# NFL Betting System - ML Training with Docker
echo "🤖 Starting NFL Betting System - ML Training Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing containers
echo "🧹 Cleaning up existing ML containers..."
docker-compose -f docker-compose.ml.yml down

# Build and start services
echo "📦 Building ML containers..."
docker-compose -f docker-compose.ml.yml build

echo "🚀 Starting ML training environment..."
docker-compose -f docker-compose.ml.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for ML services to start..."
sleep 15

# Check service status
echo "📊 ML Service Status:"
docker-compose -f docker-compose.ml.yml ps

# Show logs
echo "📋 Recent ML logs:"
docker-compose -f docker-compose.ml.yml logs --tail=20

echo ""
echo "✅ NFL Betting System - ML Training Environment is running!"
echo "🧠 Jupyter Lab: http://localhost:8888"
echo "🗄️  Database: localhost:5432"
echo "📊 Data: ../ml_training_data"
echo "🎯 Models: ../saved_models"
echo ""
echo "🛑 To stop: docker-compose -f docker-compose.ml.yml down"
echo "📋 To view logs: docker-compose -f docker-compose.ml.yml logs -f"
echo "🔍 To check status: docker-compose -f docker-compose.ml.yml ps"
echo "🔄 To restart: docker-compose -f docker-compose.ml.yml restart"
echo ""
echo "📝 ML Commands:"
echo "  • Run data processing: docker-compose -f docker-compose.ml.yml exec ml-data-processor python -c 'from src.data_processor import main; main()'"
echo "  • Run model training: docker-compose -f docker-compose.ml.yml exec ml-pipeline python -c 'from src.model_trainer import main; main()'"
echo "  • Access Jupyter: http://localhost:8888"
