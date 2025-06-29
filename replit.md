# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that provides real-time network performance testing, AI-powered diagnostics, and secure web-based management. The system consists of a central Flask server with web dashboard and distributed Python clients that perform network monitoring across multiple locations.

## System Architecture

### Architecture Pattern
- **Client-Server Model**: Distributed monitoring from multiple network locations
- **Web-based Dashboard**: Real-time visualization and test management interface
- **Dual Authentication System**: Separate authentication for web GUI (Flask-Login) and client API (token-based)
- **Zero-Trust ML**: Local machine learning processing with no external dependencies

### Technology Stack
- **Backend**: Python Flask with SQLAlchemy ORM
- **Frontend**: Bootstrap 5 with Chart.js for data visualization
- **Database**: SQLite (development) / PostgreSQL (production)
- **ML/AI**: Scikit-learn for local diagnostic processing
- **Authentication**: Flask-Login for web users, API tokens for clients
- **Deployment**: Gunicorn WSGI server

## Key Components

### 1. Flask Web Server (`app.py`, `routes.py`)
- Central server handling web GUI and API endpoints
- SQLAlchemy database integration with automatic migrations
- Flask-Login session management
- Development mode bypass for authentication during testing

### 2. Database Layer (`models.py`)
**Core Tables:**
- `Client`: Stores client information and system details
- `Test`: Defines network monitoring tests with configuration
- `TestResult`: Stores 65+ performance metrics per test run
- `TestClient`: Many-to-many relationship for test assignments
- `User`: Web GUI user accounts with role-based access
- `ApiToken`: Client authentication token management
- `SystemConfig`: Dynamic system configuration settings

### 3. Network Monitoring Client (`client.py`)
- Standalone Python client for distributed monitoring
- Performs comprehensive network tests (ping, traceroute, bandwidth, QoS)
- Collects 65+ system and network metrics
- Automatic server registration and heartbeat mechanism
- Graceful handling of missing dependencies (speedtest-cli, scapy)

### 4. AI/ML Diagnostic Engine (`ml_diagnostics.py`)
- Local Scikit-learn models for anomaly detection
- Network health classification and performance prediction
- Intelligent troubleshooting recommendations
- Model training and management without external cloud dependencies

### 5. PDF Report Generation (`pdf_generator.py`)
- Professional executive reports with charts and analysis
- ReportLab integration for PDF creation
- Matplotlib chart generation and embedding
- Comprehensive performance summaries

### 6. Security & Validation (`validation.py`)
- Input sanitization and XSS prevention
- API parameter validation with Marshmallow schemas
- SQL injection protection through ORM usage
- Secure token generation and management

## Data Flow

1. **Client Registration**: Clients register with server using API tokens
2. **Test Assignment**: Server assigns network tests to specific clients
3. **Metric Collection**: Clients perform tests and collect system metrics
4. **Data Transmission**: Results sent via secure HTTP API with JSON payload
5. **Data Storage**: Server stores results with timestamp and client association
6. **Real-time Visualization**: Web dashboard displays live charts and analytics
7. **AI Analysis**: ML models process data for anomaly detection and health insights

## External Dependencies

### Python Libraries
**Core Flask Stack:**
- flask>=2.3.0, flask-sqlalchemy>=3.0.0, flask-login
- sqlalchemy>=2.0.0, psycopg2-binary>=2.9.0 (PostgreSQL)
- gunicorn>=21.0.0, werkzeug>=2.3.0

**Network Monitoring:**
- psutil>=5.9.0 (system metrics)
- requests>=2.28.0 (HTTP client)
- scapy>=2.5.0 (advanced network analysis)
- speedtest-cli (bandwidth testing)

**AI/ML Processing:**
- scikit-learn, pandas, numpy (machine learning)
- joblib (model persistence)

**Reporting:**
- reportlab, matplotlib (PDF generation)

### System Dependencies
**Ubuntu/Debian:**
```bash
iputils-ping traceroute lm-sensors smartmontools ethtool
libpcap-dev tcpdump iw wireless-tools network-manager
```

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for rapid prototyping
- Debug mode with authentication bypass available
- Hot reloading for code changes

### Production Environment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database for scalability and concurrent access
- Environment variable configuration for secrets
- Process management with systemd or supervisor
- Reverse proxy setup (nginx/Apache) for static assets

### Configuration Management
- Environment variables for sensitive settings
- `SystemConfig` model for dynamic runtime configuration
- Development mode toggle for testing environments
- Flexible database URL configuration

## Changelog

- June 29, 2025. Initial setup
- June 29, 2025. **COMPLETED**: TCP Handshake Timing Diagnostics - Added comprehensive connection establishment analysis with database schema expansion (8 new fields), web interface enhancements, and sub-millisecond precision timing measurements
- June 29, 2025. **FIXED**: TCP Handshake Performance Analysis Display - Resolved template variable name collision that prevented diagnostic messages from displaying correctly. Analysis text now shows proper diagnostic insights instead of "None"
- June 29, 2025. **ENHANCED**: TCP Handshake Analysis Messages - Replaced generic "good performance" messages with distinctive bottleneck identification, specific timing breakdowns, and actionable insights for improved user experience
- June 29, 2025. **FIXED**: TCP Handshake Data Collection - Resolved issue where newer tests weren't collecting TCP handshake timing data due to fallback method using outdated hardcoded analysis logic. Updated fallback method to use improved analysis function
- June 29, 2025. **FIXED**: TCP Handshake Exception Handling - Resolved critical issue where exceptions in TCP handshake analysis caused complete data loss (showing "N/Ams"). Added comprehensive error handling, safe null value processing, and fixed max() function comparison errors. Tests now collect timing data even when advanced analysis fails

## User Preferences

Preferred communication style: Simple, everyday language.