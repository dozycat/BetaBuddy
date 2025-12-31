#!/bin/bash

# BetaBuddy Shutdown Script
# Stops backend and frontend services

echo "========================================="
echo "BetaBuddy Shutdown Script"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Read PIDs from files
BACKEND_PID=""
FRONTEND_PID=""

if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
fi

if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
fi

# Stop backend
if [ -n "$BACKEND_PID" ]; then
    echo "Stopping backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null && echo -e "${GREEN}Backend stopped${NC}" || echo -e "${RED}Backend process not found${NC}"
    rm -f .backend.pid
else
    echo "No backend PID file found"
fi

# Stop frontend
if [ -n "$FRONTEND_PID" ]; then
    echo "Stopping frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null && echo -e "${GREEN}Frontend stopped${NC}" || echo -e "${RED}Frontend process not found${NC}"
    rm -f .frontend.pid
else
    echo "No frontend PID file found"
fi

# Also try to kill by port (backup method)
echo ""
echo "Checking for processes on ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "Killed process on port 8000"
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "Killed process on port 3000"

echo ""
echo -e "${GREEN}Shutdown complete${NC}"
