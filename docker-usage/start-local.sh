#!/bin/bash

# NFL Betting System - Local Development with Docker
echo "🏠 Starting NFL Betting System - Local Development..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.local.yml down

# Build and start services
echo "📦 Building containers..."
docker-compose -f docker-compose.local.yml build

echo "🚀 Starting local development environment..."
docker-compose -f docker-compose.local.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Check service status
echo "📊 Service Status:"
docker-compose -f docker-compose.local.yml ps

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.local.yml logs --tail=20

echo ""
echo "✅ NFL Betting System - Local Development is running!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo "🗄️  Database: localhost:5432"
echo "⚡ Redis: localhost:6379"
echo ""
echo "🛑 To stop: docker-compose -f docker-compose.local.yml down"
echo "📋 To view logs: docker-compose -f docker-compose.local.yml logs -f"
echo "🔍 To check status: docker-compose -f docker-compose.local.yml ps"
echo "🔄 To restart: docker-compose -f docker-compose.local.yml restart"
