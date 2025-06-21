"""
StreamSwarm Configuration
"""

import os

# Server Configuration
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', 5000))
DEBUG_MODE = os.getenv('DEBUG', 'True').lower() == 'true'

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///streamswarm.db')

# Security Configuration
SESSION_SECRET = os.getenv('SESSION_SECRET', 'dev-secret-key-change-in-production')

# Client Configuration
DEFAULT_TEST_DURATION = int(os.getenv('DEFAULT_TEST_DURATION', 300))  # seconds
DEFAULT_TEST_INTERVAL = int(os.getenv('DEFAULT_TEST_INTERVAL', 5))   # seconds
CLIENT_HEARTBEAT_INTERVAL = int(os.getenv('CLIENT_HEARTBEAT_INTERVAL', 30))  # seconds
CLIENT_OFFLINE_TIMEOUT = int(os.getenv('CLIENT_OFFLINE_TIMEOUT', 300))  # seconds

# Network Testing Configuration
PING_COUNT = int(os.getenv('PING_COUNT', 4))
PING_TIMEOUT = int(os.getenv('PING_TIMEOUT', 30))  # seconds
TRACEROUTE_TIMEOUT = int(os.getenv('TRACEROUTE_TIMEOUT', 60))  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Application Settings
APP_NAME = 'StreamSwarm'
APP_VERSION = '1.0.0'
