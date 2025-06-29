# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system featuring a Flask web dashboard, AI-powered network diagnostics, and secure client-server architecture. The system enables monitoring of network performance and system metrics from multiple distributed clients with real-time visualization and automated analysis.

## System Architecture

### Backend Architecture
- **Framework**: Flask web application with SQLAlchemy ORM
- **Database**: SQLite for development, PostgreSQL recommended for production
- **WSGI Server**: Gunicorn for production deployment with multi-worker support
- **Authentication**: Dual authentication system:
  - Flask-Login for web GUI users
  - API tokens for client authentication

### Frontend Architecture
- **Templates**: Jinja2 templating with Bootstrap 5 dark theme
- **Visualization**: Chart.js for real-time charts and data visualization
- **Responsive Design**: Mobile-friendly interface with progressive enhancement
- **Static Assets**: Custom CSS and JavaScript for enhanced UX

### Client Architecture
- **Distributed Clients**: Python-based monitoring agents deployable on multiple hosts
- **Communication**: RESTful API with JSON payloads for data transmission
- **Metrics Collection**: 65+ performance metrics including network, system, and application layer data
- **Network Testing**: Comprehensive network analysis including latency, packet loss, jitter, bandwidth, and QoS monitoring

## Key Components

### Database Models
1. **Client**: Stores client registration information and system details
2. **Test**: Defines network monitoring test configurations
3. **TestResult**: Stores collected metrics (65+ columns of performance data)
4. **TestClient**: Many-to-many relationship for test assignments
5. **ApiToken**: Manages client authentication tokens
6. **User**: Web GUI user accounts with role-based access control
7. **OAuth**: Session storage for Flask-Login authentication

### Core Services
1. **Web Dashboard**: Real-time monitoring interface with interactive charts
2. **Client Management**: Registration, authentication, and assignment system
3. **Test Management**: Creation, scheduling, and execution of network tests
4. **Data Collection**: Automated metrics gathering from distributed clients
5. **AI Diagnostics**: Local machine learning models for anomaly detection
6. **Report Generation**: PDF report generation with charts and analysis

### Security Features
- Input validation and sanitization to prevent injection attacks
- Role-based access control (user/admin roles)
- Secure password hashing with Werkzeug
- API token-based authentication for clients
- XSS protection with HTML sanitization

## Data Flow

1. **Client Registration**: Clients register with server and receive API tokens
2. **Test Assignment**: Server assigns network tests to specific clients based on configuration
3. **Metrics Collection**: Clients perform network tests and collect system metrics locally
4. **Data Transmission**: Results transmitted to server via HTTP API with JSON payloads
5. **Data Storage**: Server stores results in database with timestamps and client associations
6. **Visualization**: Web dashboard displays real-time charts and historical data analysis
7. **AI Analysis**: Local ML models analyze data for anomaly detection and health classification

## External Dependencies

### Core Python Libraries
- Flask (>=2.3.0) - Web framework
- SQLAlchemy (>=2.0.0) - Database ORM
- Gunicorn (>=21.0.0) - WSGI server
- Psutil (>=5.9.0) - System metrics collection
- Requests (>=2.28.0) - HTTP client library

### Optional Libraries
- Scapy (>=2.5.0) - Advanced network packet analysis
- Speedtest-cli - Bandwidth testing
- Scikit-learn - Machine learning models
- ReportLab - PDF report generation
- Matplotlib - Chart generation for reports

### System Dependencies
- Python 3.9+ with development headers
- Network utilities: ping, traceroute, netstat
- System monitoring tools: lm-sensors, smartmontools
- Wireless tools: iw, wireless-tools (for wireless monitoring)

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for simplicity and rapid development
- Debug mode enabled for detailed error reporting
- Single-threaded execution for easier debugging

### Production Environment
- Gunicorn WSGI server with multiple workers for scalability
- PostgreSQL database for improved performance and concurrency
- Environment variable configuration for sensitive data
- Reverse proxy (nginx) recommended for static file serving
- SSL/TLS termination for secure communications

### Client Deployment
- Standalone Python scripts deployable on Linux/Windows
- Automatic registration and token-based authentication
- Configurable monitoring intervals and test parameters
- Support for distributed deployment across network segments

## Changelog

- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.