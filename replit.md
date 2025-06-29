# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system with Flask-based web GUI, AI/ML diagnostics, and secure authentication. The system enables distributed network performance testing and system resource monitoring across multiple client hosts with a centralized web dashboard for management and visualization.

## System Architecture

### Client-Server Model
- **Distributed Architecture**: Multiple client hosts perform network monitoring and report to a central server
- **Web-based Management**: Flask application provides real-time visualization and test management
- **API-driven Communication**: RESTful API for client-server communication with JSON payloads
- **Dual Authentication**: Separate authentication systems for web GUI (Flask-Login) and client API (token-based)

### Technology Stack
- **Backend**: Python Flask with SQLAlchemy ORM
- **Database**: SQLite for development, PostgreSQL for production
- **Frontend**: Bootstrap 5 with Chart.js for data visualization
- **ML/AI**: Scikit-learn for local machine learning processing
- **Networking**: Scapy for advanced network analysis, psutil for system metrics
- **Deployment**: Gunicorn WSGI server with autoscaling support

## Key Components

### Database Models
- **Client**: Stores client information and system details
- **Test**: Defines network monitoring tests with configuration parameters
- **TestResult**: Stores collected metrics and performance data (65+ columns)
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Manages client authentication tokens
- **User**: Web GUI user accounts with role-based access control
- **OAuth**: Session storage for Flask-Login authentication

### Core Services
- **Network Testing Engine**: Comprehensive network analysis including latency, packet loss, jitter, bandwidth, MTU discovery
- **System Monitoring**: CPU, memory, disk, network interface metrics collection
- **AI Diagnostic Engine**: Local machine learning models for anomaly detection and health classification
- **PDF Report Generator**: Professional executive reports with charts and recommendations
- **Security Layer**: Input validation, XSS protection, and secure authentication

### Web Interface Components
- **Dashboard**: Real-time metrics visualization and system overview
- **Client Management**: Monitor and manage connected clients
- **Test Management**: Create, schedule, and monitor network tests
- **Results Visualization**: Interactive charts and historical data analysis
- **AI Diagnosis**: Machine learning-powered network troubleshooting
- **User Management**: Role-based access control and user administration

## Data Flow

1. **Client Registration**: Clients register with server using API tokens for authentication
2. **Test Assignment**: Server assigns network tests to specific clients based on configuration
3. **Metric Collection**: Clients perform network tests and collect 65+ system/network metrics
4. **Data Transmission**: Results transmitted to server via HTTP API with JSON payload
5. **Data Storage**: Server stores timestamped results in database with client association
6. **Real-time Visualization**: Web dashboard displays live charts and historical data
7. **AI Analysis**: Machine learning models analyze data for anomaly detection and health insights
8. **Report Generation**: Professional PDF reports with executive summaries and recommendations

## External Dependencies

### Python Libraries
- **Flask Stack**: flask, flask-sqlalchemy, flask-login, werkzeug
- **Database**: psycopg2-binary (PostgreSQL), sqlalchemy
- **System Monitoring**: psutil, requests
- **Network Testing**: scapy, speedtest-cli
- **ML/AI**: scikit-learn, pandas, numpy, joblib
- **Reporting**: reportlab, matplotlib
- **Security**: email-validator, bleach, marshmallow

### System Dependencies
- **Network Tools**: ping, traceroute, tcpdump
- **System Monitoring**: lm-sensors, smartmontools, ethtool
- **Wireless**: iw, wireless-tools, network-manager
- **Development**: libpcap-dev, python3-dev

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for simplicity and rapid development
- Debug mode enabled for detailed error reporting
- Hot reloading for code changes

### Production Environment
- Gunicorn WSGI server with multiple workers for scalability
- PostgreSQL database for enterprise-grade performance
- Environment variable configuration for security
- Reverse proxy (nginx) for static file serving and SSL termination
- Systemd service for automatic startup and monitoring

### Security Considerations
- Input validation and sanitization for all user inputs
- XSS protection with HTML sanitization
- CSRF protection for form submissions
- Secure session management with Flask-Login
- API token-based authentication for client connections
- Role-based access control for administrative functions

## Changelog

- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.