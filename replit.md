# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that combines Flask web GUI, AI/ML diagnostics, and secure authentication. The system enables real-time network performance testing across multiple client hosts with centralized management and analysis.

## System Architecture

### Client-Server Model
- **Central Flask Server**: Manages tests, stores results, provides web dashboard
- **Distributed Clients**: Perform network tests from multiple locations
- **Database Layer**: SQLAlchemy ORM with SQLite (dev) or PostgreSQL (production)
- **Authentication Layer**: Dual system - Flask-Login for web GUI, API tokens for clients

### Key Technologies
- **Backend**: Flask, SQLAlchemy, Gunicorn
- **Database**: SQLite (development), PostgreSQL (production)
- **ML/AI**: Scikit-learn, Pandas, NumPy (local processing only)
- **Frontend**: Bootstrap 5, Chart.js, Font Awesome
- **Network Testing**: psutil, scapy, speedtest-cli

## Key Components

### 1. Flask Web Application (`app.py`, `routes.py`)
- Main application factory with Flask-SQLAlchemy integration
- Comprehensive routing for dashboard, tests, clients, and administration
- Development mode bypass for authentication during testing
- ProxyFix middleware for production deployment

### 2. Database Models (`models.py`)
- **Client**: Stores client system information and status
- **Test**: Defines network monitoring configurations
- **TestResult**: Stores 65+ performance metrics per test execution
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Manages client authentication tokens
- **User**: Web GUI user accounts with role-based access
- **SystemConfig**: Dynamic system configuration storage

### 3. Client Agent (`client.py`)
- Standalone Python script for distributed deployment
- Network testing capabilities (ping, traceroute, bandwidth, QoS)
- System resource monitoring (CPU, memory, disk, network interfaces)
- Automatic registration and heartbeat with server
- Secure API token authentication

### 4. ML Diagnostic Engine (`ml_diagnostics.py`)
- Local machine learning models using Scikit-learn
- Anomaly detection with Isolation Forest
- Health classification with Random Forest
- Performance prediction with Gradient Boosting
- Zero external dependencies (no cloud API calls)

### 5. PDF Report Generation (`pdf_generator.py`)
- Professional executive reports with ReportLab
- Integrated charts using Matplotlib
- Comprehensive analysis and recommendations
- Automated export functionality

### 6. Input Validation (`validation.py`)
- Comprehensive security validation for all API endpoints
- XSS prevention with HTML sanitization
- SQL injection protection
- Network parameter validation (IP addresses, hostnames, ports)

## Data Flow

1. **Client Registration**: Clients register with server using API tokens
2. **Test Assignment**: Server assigns tests to specific clients based on configuration
3. **Metric Collection**: Clients perform network tests and collect 65+ system metrics
4. **Data Transmission**: Results transmitted via secure HTTP API with JSON payload
5. **Storage**: Server stores timestamped results with client association
6. **Analysis**: ML models analyze data for anomaly detection and health classification
7. **Visualization**: Web dashboard displays real-time charts and historical trends
8. **Reporting**: Professional PDF reports generated on demand

## External Dependencies

### Core Python Libraries
- Flask ecosystem (Flask, Flask-SQLAlchemy, Flask-Login)
- Database drivers (psycopg2-binary for PostgreSQL)
- System monitoring (psutil)
- HTTP client (requests)
- WSGI server (gunicorn)

### Network Testing Libraries
- scapy (packet analysis and QoS monitoring)
- speedtest-cli (bandwidth testing)
- System utilities (ping, traceroute, ethtool)

### ML/AI Libraries
- scikit-learn (machine learning models)
- pandas (data manipulation)
- numpy (numerical computing)
- joblib (model persistence)

### Report Generation
- reportlab (PDF generation)
- matplotlib (chart creation)

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for simplicity
- Debug mode enabled with authentication bypass option
- Hot reload for development

### Production Environment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database for scalability and concurrent access
- Environment variable configuration for secrets
- Proxy setup (nginx recommended)
- SSL/TLS termination
- Log aggregation and monitoring

### Client Deployment
- Standalone Python script deployment
- Service/daemon configuration for continuous operation
- Automated startup and restart capabilities
- Centralized configuration management

## Changelog
- June 29, 2025. Initial setup
- June 29, 2025. Fixed ML model loading bug: Anomaly Detection model filename inconsistency resolved
- June 30, 2025. Enhanced anomaly detection with detailed breakdown showing specific issues, timestamps, and actionable recommendations
- June 30, 2025. Implemented configurable session timeout with slider interface (10min-60min or Disabled) and smart activity tracking
- June 30, 2025. Added infinite scroll for tests page with real-time progress updates and smooth loading experience
- June 30, 2025. Implemented identical infinite scroll and search system for clients page with hostname, IP, and status filtering
- June 30, 2025. Improved tutorial page layout by making Windows/macOS installation cards display horizontally using responsive Bootstrap columns (col-md-6 instead of col-lg-6)

## User Preferences

Preferred communication style: Simple, everyday language.