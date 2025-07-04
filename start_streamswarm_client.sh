#!/bin/bash

# StreamSwarm Client Startup Script for Linux
# This script is designed to be called from cron @reboot for automatic client startup
# 
# Usage: 
#   1. Edit the configuration variables below
#   2. Make executable: chmod +x start_streamswarm_client.sh
#   3. Add to cron: @reboot /path/to/start_streamswarm_client.sh
#
# Author: StreamSwarm Project
# Version: 1.0

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES FOR YOUR DEPLOYMENT
# =============================================================================

# StreamSwarm server URL (include https:// and port if needed)
SERVER_URL="https://your-server.replit.app"

# API token for client authentication (get from server admin panel)
API_TOKEN="your-api-token-here"

# Client name (will be displayed in server dashboard)
CLIENT_NAME="$(hostname)-client"

# Python virtual environment path
VENV_PATH="/home/$(whoami)/streamswarm-venv"

# Application directory path
APP_DIR="/home/$(whoami)/streamswarm-client"

# Screen session name
SCREEN_SESSION="streamswarm-client"

# Log file location
LOG_FILE="/var/log/streamswarm-client.log"

# =============================================================================
# SCRIPT LOGIC - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start logging
log_message "Starting StreamSwarm client startup script"

# Wait for network to be available (important for @reboot)
log_message "Waiting for network connectivity..."
while ! ping -c 1 google.com &> /dev/null; do
    sleep 5
done
log_message "Network is available"

# Check if screen is installed
if ! command_exists screen; then
    log_message "ERROR: screen is not installed. Please install with: sudo apt-get install screen"
    exit 1
fi

# Check if git is installed
if ! command_exists git; then
    log_message "ERROR: git is not installed. Please install with: sudo apt-get install git"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log_message "ERROR: Virtual environment not found at $VENV_PATH"
    log_message "Please create with: python3 -m venv $VENV_PATH"
    exit 1
fi

# Check if application directory exists
if [ ! -d "$APP_DIR" ]; then
    log_message "ERROR: Application directory not found at $APP_DIR"
    log_message "Please clone the repository to $APP_DIR"
    exit 1
fi

# Check if client.py exists
if [ ! -f "$APP_DIR/client.py" ]; then
    log_message "ERROR: client.py not found in $APP_DIR"
    exit 1
fi

# Kill existing screen session if it exists
if screen -list | grep -q "$SCREEN_SESSION"; then
    log_message "Killing existing screen session: $SCREEN_SESSION"
    screen -S "$SCREEN_SESSION" -X quit
    sleep 2
fi

# Change to application directory
cd "$APP_DIR" || {
    log_message "ERROR: Cannot change to application directory $APP_DIR"
    exit 1
}

# Update from git repository
log_message "Updating application from git repository..."
git pull origin main 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_message "WARNING: git pull failed, continuing with existing code"
fi

# Create new screen session and start client
log_message "Creating screen session: $SCREEN_SESSION"
log_message "Starting StreamSwarm client with server: $SERVER_URL"

# Start screen session with client
screen -dmS "$SCREEN_SESSION" bash -c "
    source '$VENV_PATH/bin/activate'
    cd '$APP_DIR'
    python client.py --server-url '$SERVER_URL' --api-token '$API_TOKEN' --client-name '$CLIENT_NAME' 2>&1 | tee -a '$LOG_FILE'
"

# Verify screen session started
sleep 2
if screen -list | grep -q "$SCREEN_SESSION"; then
    log_message "SUCCESS: StreamSwarm client started in screen session: $SCREEN_SESSION"
    log_message "To view client: screen -r $SCREEN_SESSION"
    log_message "To detach: Ctrl+A then D"
    log_message "To kill: screen -S $SCREEN_SESSION -X quit"
else
    log_message "ERROR: Failed to start screen session"
    exit 1
fi

# Create status script for easy monitoring
cat > "$APP_DIR/client_status.sh" << 'EOF'
#!/bin/bash
# StreamSwarm Client Status Script

SCREEN_SESSION="streamswarm-client"
LOG_FILE="/var/log/streamswarm-client.log"

echo "=== StreamSwarm Client Status ==="
echo "Screen session status:"
if screen -list | grep -q "$SCREEN_SESSION"; then
    echo "  ✓ Client is running in screen session: $SCREEN_SESSION"
    echo "  Command: screen -r $SCREEN_SESSION"
else
    echo "  ✗ Client is not running"
fi

echo ""
echo "Recent log entries:"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "  No log file found"
fi
EOF

chmod +x "$APP_DIR/client_status.sh"

log_message "Client startup complete!"
log_message "Use '$APP_DIR/client_status.sh' to check client status"

exit 0