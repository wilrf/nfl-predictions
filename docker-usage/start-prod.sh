#!/bin/bash

# NFL Betting System - Production Deployment with Docker
echo "🚀 Starting NFL Betting System - Production Deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.prod.yml down

# Build and start services
echo "📦 Building production containers..."
docker-compose -f docker-compose.prod.yml build

echo "🚀 Starting production environment..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 20

# Check service status
echo "📊 Service Status:"
docker-compose -f docker-compose.prod.yml ps

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.prod.yml logs --tail=20

# Health check
echo "🏥 Health Check:"
curl -f http://localhost:8000/api/health && echo "✅ API Health Check Passed" || echo "❌ API Health Check Failed"
curl -f http://localhost:3000 && echo "✅ Frontend Health Check Passed" || echo "❌ Frontend Health Check Failed"

echo ""
echo "✅ NFL Betting System - Production is running!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo "🗄️  Database: localhost:5432"
echo "⚡ Redis: localhost:6379"
echo ""
echo "🛑 To stop: docker-compose -f docker-compose.prod.yml down"
echo "📋 To view logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "🔍 To check status: docker-compose -f docker-compose.prod.yml ps"
echo "🔄 To restart: docker-compose -f docker-compose.prod.yml restart"
