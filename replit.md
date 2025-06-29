# StreamSwarm - Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that provides real-time network performance testing, AI-powered diagnostics, and secure web-based management. The system follows a client-server architecture where multiple clients perform network tests and report results to a central server with a Flask-based web dashboard.

## System Architecture

### Frontend Architecture
- **Web Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5 with dark theme
- **Client-Side**: Vanilla JavaScript with Chart.js for data visualization
- **Authentication**: Flask-Login with session-based authentication
- **Styling**: Custom CSS with Bootstrap overrides

### Backend Architecture
- **Application Server**: Flask with Gunicorn WSGI server for production
- **Database ORM**: SQLAlchemy with declarative base model
- **API Layer**: RESTful API endpoints for client communication
- **Authentication**: Dual system - Flask-Login for web GUI, API tokens for clients
- **Background Processing**: Threading for concurrent operations

### Data Storage Solutions
- **Development**: SQLite database for simplicity
- **Production**: PostgreSQL with connection pooling
- **ORM**: SQLAlchemy with automatic migration support
- **Session Storage**: Database-backed sessions via Flask-Login

## Key Components

### Core Models
- **Client**: Stores client information and system details
- **Test**: Defines network monitoring tests with configuration
- **TestResult**: Stores 65+ performance metrics per test execution
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Manages client authentication tokens
- **User**: Web GUI user accounts with role-based access
- **SystemConfig**: Application configuration management

### Client System
- **Network Testing**: Comprehensive network analysis (latency, packet loss, jitter, bandwidth)
- **System Monitoring**: Hardware metrics, resource utilization
- **Advanced Features**: QoS monitoring with Scapy, speedtest integration
- **Communication**: HTTP API with JSON payload transmission

### AI/ML Diagnostic Engine
- **Local Processing**: Scikit-learn models with no external dependencies
- **Anomaly Detection**: Isolation Forest for outlier identification
- **Health Classification**: Random Forest for network health assessment
- **Performance Prediction**: Gradient Boosting for trend analysis
- **Model Management**: Automatic training and persistence

### Web Dashboard
- **Real-time Visualization**: Interactive charts with Chart.js
- **Test Management**: Create, schedule, and monitor network tests
- **Client Management**: Monitor connected clients and their status
- **Reporting**: PDF report generation with charts and analysis

## Data Flow

1. **Client Registration**: Clients register with server and receive API tokens
2. **Test Assignment**: Server assigns tests to specific clients based on configuration
3. **Metric Collection**: Clients perform network tests and collect 65+ system metrics
4. **Data Transmission**: Results sent to server via HTTP API with JSON payload
5. **Data Storage**: Server stores results with timestamp and client association
6. **Real-time Updates**: Web dashboard displays live charts and historical data
7. **AI Analysis**: ML models analyze data for anomaly detection and health classification
8. **Report Generation**: Executive PDF reports with comprehensive analysis

## External Dependencies

### Python Libraries
- **Web Framework**: Flask, Flask-SQLAlchemy, Flask-Login
- **Database**: psycopg2-binary (PostgreSQL), SQLAlchemy
- **System Monitoring**: psutil
- **Network Testing**: requests, scapy, speedtest-cli
- **ML/AI**: scikit-learn, pandas, numpy, joblib
- **Reporting**: reportlab, matplotlib
- **Security**: werkzeug, email-validator
- **Production**: gunicorn

### System Dependencies
- **Network Tools**: ping, traceroute, tcpdump
- **Hardware Monitoring**: lm-sensors, smartmontools, ethtool
- **Wireless**: iw, wireless-tools, network-manager
- **Development**: libpcap-dev for packet capture

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for rapid prototyping
- Debug mode with detailed error reporting
- Development mode bypass for authentication testing

### Production Environment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database with connection pooling
- Environment variable configuration for secrets
- Reverse proxy setup (Nginx recommended)
- SSL/TLS termination
- Process monitoring and auto-restart

### Security Considerations
- API token-based authentication for clients
- Role-based access control for web users
- Input validation and sanitization
- SQL injection prevention via ORM
- XSS protection with template escaping
- CSRF protection for web forms

## Changelog
- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.