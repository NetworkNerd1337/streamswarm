# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that provides real-time network performance analysis with AI-powered diagnostics. The system features a Flask-based web GUI for management, secure authentication, and collects 65+ performance metrics from distributed client nodes.

## System Architecture

### Frontend Architecture
- **Web Interface**: Flask-based web application with Bootstrap 5 dark theme
- **Authentication**: Flask-Login with session management for web GUI users
- **Real-time Visualization**: Chart.js for interactive performance charts and dashboards
- **Responsive Design**: Mobile-friendly interface with modern UI components

### Backend Architecture
- **Web Framework**: Flask with Gunicorn WSGI server for production deployment
- **Database**: SQLAlchemy ORM with support for SQLite (development) and PostgreSQL (production)
- **API Design**: RESTful API endpoints for client-server communication
- **Background Processing**: Threading for concurrent test execution and data collection

### Client Architecture
- **Distributed Clients**: Python-based monitoring agents deployed across network locations
- **Metric Collection**: 65+ performance metrics including network, system, and application layer data
- **Network Testing**: Comprehensive testing with ping, traceroute, bandwidth, and QoS analysis
- **System Monitoring**: CPU, memory, disk, and wireless interface monitoring

## Key Components

### Database Models
- **User**: Web GUI authentication with role-based access (admin/user)
- **Client**: Client registration and system information storage
- **Test**: Test configuration and management
- **TestResult**: Performance metrics storage (65+ columns)
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Client authentication token management

### Core Services
- **Authentication System**: Dual authentication (web GUI + API tokens)
- **Test Management**: Create, schedule, and manage network tests
- **Metric Collection**: Real-time data gathering from distributed clients
- **AI Diagnostics**: Local machine learning with Scikit-learn for anomaly detection
- **Report Generation**: PDF reports with charts and analysis using ReportLab

### Security Features
- **Input Validation**: Comprehensive sanitization using Bleach and Marshmallow
- **SQL Injection Prevention**: SQLAlchemy ORM with parameterized queries
- **XSS Protection**: HTML sanitization and Content Security Policy
- **Authentication**: Separate systems for web users and API clients

## Data Flow

1. **Client Registration**: Clients register with server and receive API tokens
2. **Test Assignment**: Server assigns tests to specific clients based on configuration
3. **Metric Collection**: Clients perform network tests and collect system metrics
4. **Data Transmission**: Results sent to server via HTTP API with JSON payload
5. **Data Storage**: Server stores results in database with timestamp and client association
6. **Visualization**: Web dashboard displays real-time charts and historical data
7. **AI Analysis**: ML models analyze data for anomaly detection and health classification

## External Dependencies

### Core Python Libraries
- **Flask**: Web framework and application structure
- **SQLAlchemy**: Database ORM and schema management
- **Psutil**: System and process monitoring
- **Requests**: HTTP client for API communication
- **Scapy**: Advanced network packet analysis
- **Speedtest-cli**: Bandwidth testing capabilities

### AI/ML Libraries
- **Scikit-learn**: Machine learning models for diagnostics
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing support
- **Joblib**: Model serialization and persistence

### System Dependencies
- **Linux**: iputils-ping, traceroute, lm-sensors, smartmontools, ethtool
- **Network Tools**: iw, wireless-tools, tcpdump, libpcap-dev
- **Development**: build-essential, python3-dev, gcc

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for rapid prototyping
- Debug mode enabled for development
- Hot reload for code changes

### Production Environment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database for scalability and performance
- Environment variable configuration for secrets
- Reverse proxy setup (Nginx recommended)
- SSL/TLS termination for secure communication

### Docker Deployment
- Containerized application with multi-stage builds
- Separate containers for web server and database
- Docker Compose for orchestration
- Volume mounting for persistent data

## Changelog

- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.