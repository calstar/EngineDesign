#!/bin/bash

# Development script to run both backend and frontend together
# Usage: ./dev.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting EngineDesign development servers...${NC}"

# Cleanup function to kill background processes on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${BLUE}Starting backend on http://localhost:8000${NC}"
cd "$PROJECT_ROOT"
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo -e "${BLUE}Starting frontend on http://localhost:5173${NC}"
cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo -e "${GREEN}Both servers running! Press Ctrl+C to stop.${NC}"

# Wait for both processes
wait

