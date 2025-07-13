#!/bin/bash

# StreamSwarm Server Startup Script for Linux
# This script is designed to be called from cron @reboot for automatic server startup
# 
# Usage: 
#   1. Copy streamswarm_config.sh to parent directory (outside git repo)
#   2. Edit the configuration variables in ../streamswarm_config.sh
#   3. Make executable: chmod +x start_streamswarm_server.sh
#   4. Add to cron: @reboot /path/to/Swarm/scripts/start_streamswarm_server.sh
#
# Author: StreamSwarm Project
# Version: 2.0

# =============================================================================
# LOAD CONFIGURATION FROM PARENT DIRECTORY
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration file should be in parent directory (outside git repo)
CONFIG_FILE="$SCRIPT_DIR/../../streamswarm_config.sh"

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found at $CONFIG_FILE"
    echo "Please copy streamswarm_config.sh to the parent directory and configure it."
    echo "Example: cp $SCRIPT_DIR/streamswarm_config.sh $SCRIPT_DIR/../"
    exit 1
fi

# Source the configuration file
source "$CONFIG_FILE"

# Verify configuration was loaded
if [ "$CONFIG_LOADED" != "true" ]; then
    echo "ERROR: Configuration file did not load properly"
    exit 1
fi

# =============================================================================
# SCRIPT LOGIC - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SERVER_LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start logging
log_message "Starting StreamSwarm server startup script"

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

# Check if gunicorn is installed
if ! command_exists gunicorn; then
    log_message "ERROR: gunicorn is not installed. Please install with: pip install gunicorn"
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

# Check if main.py exists
if [ ! -f "$APP_DIR/main.py" ]; then
    log_message "ERROR: main.py not found in $APP_DIR"
    exit 1
fi

# Kill existing screen session if it exists
if screen -list | grep -q "$SERVER_SCREEN_SESSION"; then
    log_message "Killing existing screen session: $SERVER_SCREEN_SESSION"
    screen -S "$SERVER_SCREEN_SESSION" -X quit
    sleep 2
fi

# Change to application directory
cd "$APP_DIR" || {
    log_message "ERROR: Cannot change to application directory $APP_DIR"
    exit 1
}

# Update from git repository
log_message "Updating application from git repository..."
git pull origin main 2>&1 | tee -a "$SERVER_LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_message "WARNING: git pull failed, continuing with existing code"
fi

# Create new screen session and start server
log_message "Creating screen session: $SERVER_SCREEN_SESSION"
log_message "Starting StreamSwarm server with gunicorn on port 5000"

# Start screen session with server
screen -dmS "$SERVER_SCREEN_SESSION" bash -c "
    source '$VENV_PATH/bin/activate'
    cd '$APP_DIR'
    export SESSION_SECRET='${SESSION_SECRET:-$(openssl rand -hex 32)}'
    export DATABASE_URL='${DATABASE_URL:-sqlite:///streamswarm.db}'
    export PGDATABASE='${PGDATABASE:-}'
    export PGHOST='${PGHOST:-}'
    export PGPORT='${PGPORT:-}'
    export PGUSER='${PGUSER:-}'
    export PGPASSWORD='${PGPASSWORD:-}'
    gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app 2>&1 | tee -a '$SERVER_LOG_FILE'
"

# Verify screen session started
sleep 2
if screen -list | grep -q "$SERVER_SCREEN_SESSION"; then
    log_message "SUCCESS: StreamSwarm server started in screen session: $SERVER_SCREEN_SESSION"
    log_message "Server is accessible at: http://localhost:5000"
    log_message "To view server: screen -r $SERVER_SCREEN_SESSION"
    log_message "To detach: Ctrl+A then D"
    log_message "To kill: screen -S $SERVER_SCREEN_SESSION -X quit"
else
    log_message "ERROR: Failed to start screen session"
    exit 1
fi

# Create status script for easy monitoring
cat > "$APP_DIR/server_status.sh" << EOF
#!/bin/bash
# StreamSwarm Server Status Script

SCREEN_SESSION="$SERVER_SCREEN_SESSION"
LOG_FILE="$SERVER_LOG_FILE"

echo "=== StreamSwarm Server Status ==="
echo "Screen session status:"
if screen -list | grep -q "$SCREEN_SESSION"; then
    echo "  ✓ Server is running in screen session: $SCREEN_SESSION"
    echo "  Command: screen -r $SCREEN_SESSION"
else
    echo "  ✗ Server is not running"
fi

echo ""
echo "Server accessibility:"
if curl -s --connect-timeout 5 http://localhost:5000 > /dev/null; then
    echo "  ✓ Server is responding on port 5000"
else
    echo "  ✗ Server is not responding on port 5000"
fi

echo ""
echo "Recent log entries:"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "  No log file found"
fi
EOF

chmod +x "$APP_DIR/server_status.sh"

log_message "Server startup complete!"
log_message "Use '$APP_DIR/server_status.sh' to check server status"
log_message "Access the web interface at: http://localhost:5000"

exit 0