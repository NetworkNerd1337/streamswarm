# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system built with Flask. The system consists of a central server with a web dashboard and distributed clients that perform network tests and collect system metrics. It features AI-powered diagnostics, secure authentication, and real-time analytics for enterprise network management.

## System Architecture

### Frontend Architecture
- **Flask Web Application**: Server-side rendered templates using Jinja2
- **Bootstrap 5**: Modern responsive UI framework with dark theme
- **Chart.js**: Real-time data visualization and interactive charts
- **Progressive Enhancement**: JavaScript-enhanced functionality with graceful degradation

### Backend Architecture
- **Flask Framework**: Lightweight WSGI web application framework
- **SQLAlchemy ORM**: Database abstraction layer with model definitions
- **Client-Server Model**: Distributed architecture with REST API communication
- **Modular Design**: Separate components for web GUI, API endpoints, and client logic

### Database Architecture
- **Development**: SQLite for simplicity and quick setup
- **Production**: PostgreSQL for scalability and performance
- **ORM Models**: Comprehensive schema with relationships for clients, tests, results, and authentication

## Key Components

### Core Models
- **Client**: Stores client information and system details
- **Test**: Defines network monitoring tests with configuration
- **TestResult**: Stores collected metrics (65+ performance columns)
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Client authentication token management
- **User**: Web GUI user accounts with role-based access
- **SystemConfig**: System-wide configuration settings

### Authentication Systems
- **Dual Authentication**: Separate systems for web GUI (Flask-Login) and client API (tokens)
- **Role-Based Access Control**: Admin and regular user roles
- **Development Mode**: Authentication bypass for development/troubleshooting
- **Session Management**: Secure session handling with timeout

### AI/ML Diagnostics
- **Local Processing**: Scikit-learn models for anomaly detection and health classification
- **Zero-Trust Architecture**: No external dependencies or cloud connections
- **Diagnostic Engine**: Intelligent troubleshooting recommendations
- **Model Management**: Training and retraining capabilities

### Client System
- **Network Testing**: Comprehensive metrics including latency, packet loss, jitter, bandwidth
- **System Monitoring**: CPU, memory, disk, network interface statistics
- **Advanced Analysis**: QoS monitoring, TCP analysis, MTU discovery
- **Heartbeat System**: Client status tracking and offline detection

## Data Flow

1. **Client Registration**: Clients register with server and receive API tokens
2. **Test Assignment**: Server assigns tests to specific clients based on configuration
3. **Metric Collection**: Clients perform network tests and collect 65+ system metrics
4. **Data Transmission**: Results sent to server via HTTP API with JSON payload
5. **Data Storage**: Server stores results in database with timestamp and client association
6. **Visualization**: Web dashboard displays real-time charts and historical data
7. **AI Analysis**: ML models analyze data for anomaly detection and health classification

## External Dependencies

### Python Libraries
- **Flask**: Web framework and routing
- **SQLAlchemy**: Database ORM and migrations
- **psutil**: System and process utilities
- **requests**: HTTP client library
- **scikit-learn**: Machine learning models
- **pandas/numpy**: Data processing and analysis
- **scapy**: Advanced network packet analysis
- **reportlab**: PDF report generation

### System Dependencies
- **Network Tools**: ping, traceroute, tcpdump, ethtool
- **Monitoring Tools**: lm-sensors, smartmontools
- **Database**: PostgreSQL (production) or SQLite (development)

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for rapid prototyping
- Debug mode enabled with detailed error pages
- Authentication bypass available via development mode

### Production Environment
- **WSGI Server**: Gunicorn with multiple workers for scalability
- **Database**: PostgreSQL with connection pooling
- **Security**: Environment variable configuration for secrets
- **Process Management**: Systemd service files for auto-restart
- **Monitoring**: Built-in health checks and metrics

### Configuration Management
- Environment variable-based configuration
- Configurable via web interface for system settings
- Support for both file-based and database configuration storage

## Changelog

- June 30, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.