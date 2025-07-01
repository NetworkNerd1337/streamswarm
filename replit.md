# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that combines Flask web framework with AI-powered diagnostics. It enables distributed network performance testing across multiple client hosts with centralized management through a web dashboard. The system provides real-time analytics, machine learning-based anomaly detection, and professional reporting capabilities for enterprise network management.

## System Architecture

### Client-Server Model
The system operates on a distributed client-server architecture where:
- **Central Server**: Flask-based web application that manages tests, collects data, and provides web interface
- **Distributed Clients**: Python agents deployed across network locations that perform monitoring tasks
- **Communication**: RESTful API with JSON payloads for client-server data exchange

### Technology Stack
- **Backend**: Flask (Python web framework)
- **Database**: SQLAlchemy ORM with SQLite (development) / PostgreSQL (production)
- **Frontend**: Bootstrap 5 with Chart.js for visualization
- **AI/ML**: Scikit-learn for local machine learning processing
- **Authentication**: Flask-Login for web GUI, API tokens for client authentication
- **Deployment**: Gunicorn WSGI server for production

## Key Components

### Database Layer
- **Client**: Stores client information and system details
- **Test**: Defines network monitoring tests with configuration
- **TestResult**: Stores 65+ performance metrics per measurement
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Manages client authentication tokens
- **User**: Web GUI user accounts with role-based access
- **SystemConfig**: System-wide configuration settings

### Web Interface
- **Dashboard**: Real-time overview with statistics and charts
- **Test Management**: Create, configure, and monitor network tests
- **Client Management**: View and manage connected monitoring clients
- **Results Visualization**: Interactive charts and historical data analysis
- **User Management**: Role-based access control for web interface
- **System Configuration**: Administrative settings and development mode

### AI/ML Diagnostics
- **Anomaly Detection**: Isolation Forest algorithm for outlier detection
- **Health Classification**: Random Forest for network health assessment
- **Performance Prediction**: Gradient Boosting for trend analysis
- **Zero-Trust Architecture**: All ML processing runs locally with no external dependencies

### Client Capabilities
- **Network Testing**: Latency, packet loss, jitter, bandwidth, traceroute
- **System Monitoring**: CPU, memory, disk, network interface statistics
- **Advanced Metrics**: QoS monitoring, TCP analysis, DNS resolution timing
- **Geolocation**: Optional network path visualization with geographic mapping

## Data Flow

1. **Client Registration**: Clients register with server using API tokens
2. **Test Assignment**: Server assigns network tests to specific clients based on configuration
3. **Metric Collection**: Clients perform tests and collect 65+ performance metrics
4. **Data Transmission**: Results transmitted to server via HTTP API with JSON payloads
5. **Data Storage**: Server stores results in database with timestamps and client associations
6. **Visualization**: Web dashboard displays real-time charts and historical analysis
7. **AI Analysis**: Machine learning models analyze data for anomaly detection and health classification
8. **Reporting**: Generate professional PDF reports with charts and recommendations

## External Dependencies

### Core Dependencies
- Flask ecosystem (Flask, Flask-SQLAlchemy, Flask-Login)
- SQLAlchemy for database operations
- psutil for system monitoring
- requests for HTTP communication

### Optional Dependencies
- speedtest-cli for bandwidth testing
- scapy for advanced network packet analysis
- matplotlib/reportlab for PDF report generation
- scikit-learn for machine learning features

### System Requirements
- Python 3.7+
- Network tools: ping, traceroute (pre-installed on most systems)
- Optional: mtr for enhanced traceroute analysis

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for simplicity
- Debug mode enabled for detailed error reporting
- Development mode bypass for authentication testing

### Production Environment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database for scalability and concurrent access
- Environment variables for configuration management
- Secure session management with proper secret keys
- Proxy configuration for load balancing

### Security Considerations
- Dual authentication system (web users and API tokens)
- Input validation and sanitization to prevent injection attacks
- Role-based access control with admin privileges
- Session timeout and activity tracking
- HTTPS recommended for production deployment

## Changelog

- June 30, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.