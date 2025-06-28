# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that provides real-time network performance testing and system resource monitoring across multiple client hosts. The system follows a client-server architecture where a central Flask server manages tests and collects data from distributed Python clients that perform network monitoring tasks.

## System Architecture

### Overall Architecture
- **Frontend**: Flask-based web application with Bootstrap UI framework
- **Backend**: Python Flask server with SQLAlchemy ORM
- **Database**: SQLite (with PostgreSQL support via configuration)
- **Client Architecture**: Standalone Python clients that connect to the central server
- **Deployment**: Gunicorn WSGI server with autoscaling support

### Key Design Decisions
1. **Client-Server Model**: Chosen to enable distributed monitoring from multiple network locations
2. **Web-based Dashboard**: Provides real-time visualization and test management
3. **SQLAlchemy ORM**: Enables database flexibility and easy schema management
4. **Token-based Authentication**: Secures client-server communication
5. **Modular Design**: Separate client and server components for flexible deployment

## Key Components

### Server Components
- **Flask Application** (`app.py`): Core application setup with database configuration
- **Models** (`models.py`): Database schema definitions using SQLAlchemy
- **Routes** (`routes.py`): API endpoints and web interface handlers
- **PDF Generator** (`pdf_generator.py`): Executive reporting functionality
- **Templates**: HTML templates for web dashboard interface

### Client Components
- **Client Application** (`client.py`): Standalone monitoring client with 65+ metrics collection
- **Network Testing**: Ping, traceroute, bandwidth, and QoS analysis
- **System Monitoring**: CPU, memory, disk, and network interface tracking
- **Wireless Detection**: Signal strength and wireless network analysis

### Database Schema
- **Client**: Stores client information and system details
- **Test**: Defines network monitoring tests with configuration
- **TestResult**: Stores collected metrics and performance data
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Manages client authentication tokens

## Data Flow

1. **Client Registration**: Clients register with server and receive API tokens
2. **Test Assignment**: Server assigns tests to specific clients based on configuration
3. **Metric Collection**: Clients perform network tests and collect system metrics
4. **Data Transmission**: Results sent to server via HTTP API with JSON payload
5. **Data Storage**: Server stores results in database with timestamp and client association
6. **Visualization**: Web dashboard displays real-time charts and historical data
7. **Reporting**: PDF reports generated on-demand with comprehensive analysis

## External Dependencies

### Python Libraries
- **Flask**: Web framework and API server
- **SQLAlchemy**: Database ORM and schema management
- **psutil**: System resource monitoring
- **requests**: HTTP client for API communication
- **speedtest-cli**: Internet bandwidth testing
- **scapy**: Advanced network packet analysis
- **matplotlib**: Chart generation for reports
- **reportlab**: PDF report generation

### System Dependencies
- **Network Tools**: ping, traceroute, iw for network testing
- **System Tools**: lm-sensors, smartmontools for hardware monitoring
- **Database**: PostgreSQL support via psycopg2-binary

### Optional Dependencies
- **iwlib**: Wireless interface monitoring (graceful fallback if unavailable)
- **libpcap**: Advanced packet capture capabilities

### Deprecated Dependencies (No Longer Used)
- **wireless-tools**: Replaced by modern iw package for wireless interface management

## Deployment Strategy

### Development Deployment
- Flask development server on port 5000
- SQLite database for simplicity
- Debug mode enabled for development

### Production Deployment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database for scalability
- Autoscaling deployment target configured
- Environment variable configuration for secrets

### Client Deployment
- Standalone Python script deployment
- Command-line arguments for server connection
- Automatic reconnection and error handling
- Cross-platform support (Linux/Windows)

## Changelog

- June 28, 2025: Implemented Application & Infrastructure metrics collection
  - Added comprehensive application-layer metrics collection including content download time and compression ratio analysis
  - Implemented infrastructure monitoring for power consumption, memory error rates, fan speeds, and drive health
  - Enhanced client to collect HTTP performance metrics including certificate validation timing and connection reuse
  - Added automatic detection of gzip/deflate compression effectiveness and response code tracking
  - Implemented Linux-specific infrastructure monitoring using /sys filesystem and system commands
  - Application metrics include content download timing, compression analysis, and SSL certificate validation
  - Infrastructure metrics cover power consumption monitoring, ECC memory error detection, and basic drive health checks
  - All Application & Infrastructure Metrics accordion section now displays real data instead of "N/A"
- June 27, 2025: Implemented TCP retransmission rate collection and fixed metrics display
  - Added comprehensive TCP retransmission statistics collection from /proc/net/snmp on Linux systems
  - Implemented TCP retransmission rate calculation as percentage of total segments sent
  - Enhanced client to collect tcp_retransmission_rate, tcp_out_of_order_packets, and tcp_duplicate_acks
  - Fixed missing Network Performance Metrics section display issue
  - Resolved template variable reference error preventing comprehensive metrics from showing
  - Added missing 'average' template filter to Flask application for metrics calculations
  - Fixed network interface data display with proper JSON formatting and tooltips
  - Enhanced user interface with explanatory tooltips for network errors vs drops metrics
  - All 65+ network and system metrics now display correctly in test results
- June 27, 2025: Comprehensive security enhancements and input validation system
  - Implemented robust input validation across all API endpoints to prevent injection attacks
  - Added rate limiting to all API endpoints (registration, test creation, token management)
  - Enhanced URL destination validation supporting paths and parameters while maintaining security
  - Comprehensive sanitization of all user inputs with proper field validation
  - Validated numeric fields with appropriate ranges and type checking
  - Added JSON field validation with safe parsing for complex data structures
  - Signal strength data validation with proper statistical field handling
  - Test creation endpoint now supports complex URLs (e.g., video URLs with parameters)
  - Token management endpoints with enhanced validation and duplicate prevention
  - Fixed network interface data display issue by correcting JSON validation and storage format
  - Cleaned up 24 malformed network interface records from database
  - Fixed network packet metrics display issue by adding missing data population in API endpoint
  - Added comprehensive network interface metrics to test results tables (packets, bytes, errors, drops)
  - Maintained backward compatibility while strengthening security posture
- June 26, 2025: Enhanced wireless detection with comprehensive `iw` command integration
  - Replaced deprecated `iwconfig` calls with modern `iw dev <interface> info` and `iw dev <interface> link`
  - Added parsing for MAC address, channel number, frequency, and transmission power data
  - Enhanced signal strength monitoring during tests with multiple sample collection
  - Updated web interface to display additional wireless details (MAC, channel, TX power)
  - Improved wireless interface detection for compatibility with modern Linux distributions
  - Updated troubleshooting messages and documentation to reference `iw` package
- June 26, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.