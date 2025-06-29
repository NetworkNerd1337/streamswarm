# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that combines Flask web application with distributed client architecture. The system provides real-time network performance testing, AI-powered diagnostics, and professional reporting capabilities for enterprise network management.

## System Architecture

### Backend Architecture
- **Flask Web Framework**: Main server application with modular route handling
- **SQLAlchemy ORM**: Database abstraction layer supporting SQLite (development) and PostgreSQL (production)
- **Client-Server Model**: Distributed monitoring clients report to central server via HTTP API
- **RESTful API**: JSON-based communication between clients and server

### Frontend Architecture
- **Server-Side Rendering**: Jinja2 templates with Bootstrap 5 dark theme
- **Real-Time Updates**: Chart.js for interactive data visualization
- **Responsive Design**: Mobile-friendly interface with Font Awesome icons
- **Progressive Enhancement**: JavaScript for enhanced user experience

### Authentication System
- **Dual Authentication**: Separate systems for web GUI (Flask-Login) and client API (token-based)
- **Role-Based Access**: Admin and regular user roles with different permissions
- **Development Mode**: Configurable authentication bypass for development/testing

## Key Components

### Database Models
- **Client**: Stores client system information and connection status
- **Test**: Defines network monitoring test configurations
- **TestResult**: Stores 65+ performance metrics per measurement
- **TestClient**: Many-to-many relationship for test assignments
- **User**: Web GUI user accounts with encrypted passwords
- **ApiToken**: Client authentication token management
- **SystemConfig**: Application configuration settings

### Core Services
- **Network Testing**: Ping, traceroute, bandwidth, MTU discovery, TCP analysis
- **System Monitoring**: CPU, memory, disk, network interface metrics
- **AI/ML Engine**: Local scikit-learn models for anomaly detection and health classification
- **PDF Reporting**: Professional reports with charts and analysis
- **Data Validation**: Comprehensive input sanitization and validation

### Client System
- **Autonomous Operation**: Self-registering clients with heartbeat mechanism
- **Multi-Protocol Testing**: ICMP, TCP, UDP, and application-layer testing
- **System Resource Monitoring**: Real-time collection of system performance metrics
- **Flexible Deployment**: Cross-platform Python client with minimal dependencies

## Data Flow

1. **Client Registration**: Clients automatically register with server and receive API tokens
2. **Test Assignment**: Server assigns network tests to specific clients based on configuration
3. **Metric Collection**: Clients perform tests and collect 65+ performance metrics
4. **Data Transmission**: Results transmitted to server via secure HTTP API with JSON payload
5. **Data Storage**: Server persists results in relational database with full audit trail
6. **Visualization**: Web dashboard displays real-time charts and historical trends
7. **AI Analysis**: Machine learning models analyze data for anomaly detection and diagnostics

## External Dependencies

### Core Python Libraries
- Flask >= 2.3.0 (web framework)
- SQLAlchemy >= 2.0.0 (ORM)
- psutil >= 5.9.0 (system monitoring)
- requests >= 2.28.0 (HTTP client)
- scikit-learn (machine learning)
- pandas, numpy (data processing)

### Optional Dependencies
- speedtest-cli (bandwidth testing)
- scapy (advanced network analysis)
- reportlab (PDF generation)
- matplotlib (chart generation)

### System Tools
- ping, traceroute (network utilities)
- lm-sensors, smartmontools (hardware monitoring)
- ethtool, iw (network interface tools)

### Frontend Libraries (CDN)
- Bootstrap 5 with dark theme
- Chart.js with date adapter
- Font Awesome icons

## Deployment Strategy

### Development Environment
- Flask development server on port 5000
- SQLite database for simplicity
- Debug mode enabled with authentication bypass option
- Environment variables for sensitive configuration

### Production Environment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database for scalability and concurrent access
- Reverse proxy (nginx recommended)
- Environment-based configuration management
- SSL/TLS termination at proxy layer

### Client Deployment
- Self-contained Python script with minimal dependencies
- Automatic server discovery and registration
- Configurable via command-line arguments or environment variables
- Cross-platform compatibility (Linux, Windows, macOS)

## Changelog

```
Changelog:
- June 29, 2025. Initial setup
- June 29, 2025. Added packet size configuration feature to network tests
  - Added packet_size field to Test model with default 64 bytes
  - Updated test creation form with packet size selector (64-1472 bytes)
  - Modified client ping test functionality to use variable packet sizes
  - Enhanced server API validation for packet size parameter (1-65535 bytes)
  - Successfully tested with 512-byte packets on Test 80
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```