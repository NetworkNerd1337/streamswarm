#!/bin/bash

# StreamSwarm Client Configuration File
# This file contains all user-editable configuration variables
# Place this file OUTSIDE your git repository (e.g., in parent directory)
# so it won't be overwritten during git updates
#
# Author: StreamSwarm Project
# Version: 1.0

# =============================================================================
# STREAMSWARM CLIENT CONFIGURATION
# EDIT THESE VALUES FOR YOUR DEPLOYMENT
# =============================================================================

# StreamSwarm server URL (include https:// and port if needed)
SERVER_URL="https://swarmstreamserver-example.com"

# API token for client authentication (get from server admin panel)
API_TOKEN="your-api-token-here"

# Client name (will be displayed in server dashboard)
CLIENT_NAME="$(hostname)"

# Python virtual environment path
VENV_PATH="/home/$(whoami)/streamswarm-env"

# Application directory path (where you cloned the StreamSwarm repository)
APP_DIR="/home/$(whoami)/Swarm"

# Screen session name
SCREEN_SESSION="streamswarm-client"

# Log file location
LOG_FILE="/home/$(whoami)/streamswarm-client.log"

# =============================================================================
# CONFIGURATION VALIDATION (DO NOT MODIFY)
# =============================================================================

# Mark configuration as loaded
CONFIG_LOADED=true