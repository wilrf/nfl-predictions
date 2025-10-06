#!/bin/bash

# NFL Betting System - Docker Quick Start
echo "🚀 Starting NFL Betting System with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start services
echo "📦 Building containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Service Status:"
docker-compose ps

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=20

echo ""
echo "✅ NFL Betting System is running!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo "🗄️  Database: localhost:5432"
echo "⚡ Redis: localhost:6379"
echo ""
echo "🛑 To stop: docker-compose down"
echo "📋 To view logs: docker-compose logs -f"
echo "🔍 To check status: docker-compose ps"
