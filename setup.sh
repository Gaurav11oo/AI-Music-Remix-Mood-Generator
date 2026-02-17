#!/bin/bash

# Music Remix App - Quick Start Setup Script
# This script sets up the complete development environment

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  🎵 Music Remix & Mood Generator - Setup Script      ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        exit 1
    else
        echo "✅ $1 found"
    fi
}

check_command node
check_command npm
check_command python3
check_command psql
check_command redis-server

echo ""
echo "✅ All prerequisites met!"
echo ""

# Setup Backend
echo "🔧 Setting up backend..."
cd backend

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file. Please configure it with your settings."
fi

npm install
echo "✅ Backend dependencies installed"

# Setup Python AI Service
echo "🐍 Setting up Python AI service..."
cd python-ai

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Python virtual environment created"
fi

source venv/bin/activate
pip install -r requirements.txt
echo "✅ Python dependencies installed"
deactivate

cd ../..

# Setup Frontend
echo "⚛️  Setting up frontend..."
cd frontend

if [ ! -f ".env.local" ]; then
    echo "NEXT_PUBLIC_API_URL=http://localhost:5000/api" > .env.local
    echo "NEXT_PUBLIC_WS_URL=ws://localhost:5000" >> .env.local
    echo "📝 Created .env.local file"
fi

npm install
echo "✅ Frontend dependencies installed"

cd ..

# Database setup
echo "🗄️  Setting up database..."
echo "Please ensure PostgreSQL is running and create a database:"
echo "  createdb music_remix_db"
echo ""
echo "Then run migrations from the backend directory:"
echo "  cd backend && npm run migrate"
echo ""

# Final instructions
echo "╔═══════════════════════════════════════════════════════╗"
echo "║             🎉 Setup Complete!                        ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "To start the application:"
echo ""
echo "1. Start PostgreSQL and Redis:"
echo "   redis-server"
echo ""
echo "2. Start backend (Terminal 1):"
echo "   cd backend && npm run dev"
echo ""
echo "3. Start Python AI service (Terminal 2):"
echo "   cd backend/python-ai && source venv/bin/activate && python app.py"
echo ""
echo "4. Start frontend (Terminal 3):"
echo "   cd frontend && npm run dev"
echo ""
echo "5. Open your browser:"
echo "   http://localhost:3000"
echo ""
echo "📖 Documentation:"
echo "   - API Docs: http://localhost:5000/api-docs"
echo "   - README: ./README.md"
echo "   - API Reference: ./docs/API.md"
echo "   - Deployment Guide: ./docs/DEPLOYMENT.md"
echo ""
echo "🎵 Happy music remixing!"
