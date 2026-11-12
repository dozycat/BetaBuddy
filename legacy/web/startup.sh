#!/bin/bash

# BetaBuddy Startup Script
# Automatically installs dependencies and starts both backend and frontend services

echo "========================================="
echo "BetaBuddy Startup Script"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
fi

printf "${YELLOW}Detected OS: $OS${NC}\n"

# Install prerequisites if missing
echo ""
printf "${YELLOW}Checking and installing prerequisites...${NC}\n"

# Install Homebrew on macOS if needed
if [[ "$OS" == "macos" ]] && ! command_exists brew; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python3
if ! command_exists python3; then
    echo "Installing Python3..."
    if [[ "$OS" == "macos" ]]; then
        brew install python3
    elif [[ "$OS" == "linux" ]]; then
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        elif command_exists yum; then
            sudo yum install -y python3 python3-pip
        fi
    fi
else
    printf "${GREEN}Python3 is already installed${NC}\n"
fi

# Install Node.js and npm
if ! command_exists node || ! command_exists npm; then
    echo "Installing Node.js and npm..."
    if [[ "$OS" == "macos" ]]; then
        brew install node
    elif [[ "$OS" == "linux" ]]; then
        if command_exists apt-get; then
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif command_exists yum; then
            curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
            sudo yum install -y nodejs
        fi
    fi
else
    printf "${GREEN}Node.js and npm are already installed${NC}\n"
fi

# Verify installations
echo ""
printf "${YELLOW}Verifying installations...${NC}\n"
python3 --version
node --version
npm --version

printf "${GREEN}Prerequisites OK${NC}\n"

# Setup Backend
echo ""
printf "${YELLOW}Setting up backend...${NC}\n"
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip || printf "${YELLOW}Warning: Failed to upgrade pip${NC}\n"
pip install -r requirements.txt || printf "${YELLOW}Warning: Some Python dependencies may have failed${NC}\n"

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        printf "${YELLOW}Creating .env from .env.example${NC}\n"
        cp .env.example .env
    else
        printf "${YELLOW}Warning: No .env file found${NC}\n"
    fi
fi

printf "${GREEN}Backend setup complete${NC}\n"

# Setup Frontend
echo ""
printf "${YELLOW}Setting up frontend...${NC}\n"
cd "$SCRIPT_DIR/frontend"

# Install dependencies
echo "Installing Node dependencies..."
npm install || printf "${YELLOW}Warning: Some Node dependencies may have failed${NC}\n"

printf "${GREEN}Frontend setup complete${NC}\n"

# Start services
echo ""
echo "========================================="
printf "${GREEN}Starting services...${NC}\n"
echo "========================================="

# Start backend in background
echo "Starting backend on http://localhost:8000"
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Start frontend in background
echo "Starting frontend on http://localhost:3000"
cd "$SCRIPT_DIR/frontend"
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Save PIDs for cleanup
echo "$BACKEND_PID" > "$SCRIPT_DIR/.backend.pid"
echo "$FRONTEND_PID" > "$SCRIPT_DIR/.frontend.pid"

# Wait for frontend to be ready
echo "Waiting for frontend to be ready..."
sleep 3

# Open browser with frontend URL
echo "Opening browser..."
if [[ "$OS" == "macos" ]]; then
    open "http://localhost:3000"
elif [[ "$OS" == "linux" ]]; then
    if command_exists xdg-open; then
        xdg-open "http://localhost:3000"
    elif command_exists gnome-open; then
        gnome-open "http://localhost:3000"
    fi
fi

echo ""
echo "========================================="
printf "${GREEN}BetaBuddy is running!${NC}\n"
echo "========================================="
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "  Backend:  tail -f backend/backend.log"
echo "  Frontend: tail -f frontend/frontend.log"
echo ""
echo "To stop services, run: ./shutdown.sh"
echo "Or press Ctrl+C and manually kill PIDs: $BACKEND_PID $FRONTEND_PID"
echo ""

# Wait for user interrupt
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait
