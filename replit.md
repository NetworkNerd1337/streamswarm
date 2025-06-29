# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a Python-based distributed network monitoring system that provides comprehensive network performance testing and AI-powered diagnostics. The system uses a client-server architecture where multiple client hosts perform network tests and report back to a central Flask server with a web-based dashboard.

## System Architecture

The system follows a distributed client-server model with the following key components:

### Backend Architecture
- **Flask Web Server**: Main server application handling web GUI and API endpoints
- **SQLAlchemy ORM**: Database abstraction layer supporting both SQLite (development) and PostgreSQL (production)
- **Client-Server Communication**: RESTful API with JSON payloads for client data transmission
- **Dual Authentication System**: 
  - Flask-Login for web GUI users with role-based access
  - Token-based authentication for client API access

### Frontend Architecture
- **Bootstrap-based Web Interface**: Dark theme responsive dashboard
- **Chart.js Integration**: Real-time data visualization and historical charts
- **Template-based Rendering**: Jinja2 templates for server-side rendering
- **Progressive Web App Features**: Offline-capable interface with service workers

### AI/ML Architecture
- **Local Processing**: Scikit-learn models run entirely on server without external dependencies
- **Zero-Trust ML**: No cloud connections or external AI services
- **Model Types**: Anomaly detection, health classification, performance prediction
- **Data Pipeline**: Automated feature extraction from 65+ collected metrics

## Key Components

### Core Application Files
- `app.py`: Flask application factory and configuration
- `main.py`: Development server entry point
- `routes.py`: Web routes and API endpoints with comprehensive input validation
- `models.py`: SQLAlchemy database models and relationships
- `client.py`: Distributed client agent for network monitoring

### Specialized Modules
- `ml_diagnostics.py`: Machine learning diagnostic engine with local model training
- `pdf_generator.py`: Professional report generation with charts and analysis
- `validation.py`: Input sanitization and security validation
- `config.py`: Environment-based configuration management

### Database Schema
- **Client**: Store client information, system details, and connection status
- **Test**: Define network monitoring tests with configuration parameters
- **TestResult**: Store 65+ performance metrics per test execution
- **TestClient**: Many-to-many relationship for flexible test assignment
- **User**: Web GUI authentication with role-based permissions
- **ApiToken**: Client authentication token management

## Data Flow

1. **Client Registration**: Clients authenticate using API tokens and register system information
2. **Test Assignment**: Server assigns specific tests to clients based on configuration
3. **Metric Collection**: Clients perform comprehensive network tests (latency, bandwidth, QoS, system resources)
4. **Data Transmission**: Results transmitted via HTTP API with JSON payload validation
5. **Real-time Processing**: Server processes and stores data with timestamp association
6. **ML Analysis**: AI models analyze patterns for anomaly detection and health scoring
7. **Visualization**: Web dashboard displays live charts and historical trends
8. **Reporting**: Generate executive-level PDF reports with recommendations

## External Dependencies

### Core Framework Dependencies
- Flask (web framework), SQLAlchemy (ORM), Flask-Login (authentication)
- Gunicorn (production WSGI server), psycopg2-binary (PostgreSQL driver)

### Client Monitoring Libraries
- psutil (system metrics), requests (HTTP client), scapy (network analysis)
- speedtest-cli (bandwidth testing), email-validator (input validation)

### AI/ML Stack
- scikit-learn (machine learning), pandas/numpy (data processing)
- matplotlib (chart generation), joblib (model persistence)

### Report Generation
- reportlab (PDF generation), bleach (HTML sanitization)

### System Tools (Linux)
- Network: ping, traceroute, tcpdump, ethtool, iw
- System: lm-sensors, smartmontools, network-manager

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for rapid prototyping
- Debug mode with hot reloading
- Environment variables for configuration

### Production Environment
- Gunicorn WSGI server with worker auto-scaling
- PostgreSQL database with connection pooling
- Environment-based secrets management
- Reverse proxy (nginx) for static files and SSL termination
- Systemd service management for reliability

### Client Deployment
- Standalone Python script deployable on any Linux host
- Automatic server discovery and token-based authentication
- Configurable test intervals and monitoring targets
- Supports multiple network interfaces and VPN connections

## Changelog

- June 29, 2025. Initial setup
- June 29, 2025. Fixed MTU Size display bug in Network Performance Metrics
- June 29, 2025. Implemented Development Mode feature for admin users to temporarily disable authentication
- June 29, 2025. Enhanced error handling for malformed URLs and graceful error pages
- June 29, 2025. Implemented case-insensitive username authentication for better user experience
- June 29, 2025. Restructured system configuration: moved Dev Mode to organized System Configuration page in user dropdown
- June 29, 2025. Implemented comprehensive TCP Window Analysis feature for advanced network bottleneck attribution with real-time monitoring and efficiency scoring

## User Preferences

- Preferred communication style: Simple, everyday language
- Development workflow: User finds authentication systems make development/troubleshooting more difficult and prefers bypass capability for faster iteration
- Error handling: User expects graceful error pages instead of white "Internal Server Error" screens for better user experience
- System organization: User prefers organized, intuitive navigation with dedicated configuration sections rather than scattered individual menu items