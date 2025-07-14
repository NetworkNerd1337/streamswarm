# StreamSwarm - Distributed Network Monitoring System

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system that provides real-time network performance testing, system resource monitoring, and AI-powered diagnostics. The system uses a client-server architecture with Flask for the web interface and SQLAlchemy for database management.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Web Framework**: Flask with Blueprint-based route organization
- **Database**: SQLAlchemy ORM with support for SQLite (development) and PostgreSQL (production)
- **Authentication**: Dual authentication system:
  - Flask-Login for web GUI with session management
  - API token-based authentication for client connections
- **ML Engine**: Local scikit-learn models for anomaly detection and network diagnostics
- **WSGI Server**: Gunicorn with worker scaling for production deployment

### Frontend Architecture
- **Template Engine**: Jinja2 with Bootstrap 5 dark theme
- **JavaScript**: Chart.js for real-time data visualization
- **Real-time Updates**: AJAX polling for dashboard updates
- **Responsive Design**: Mobile-friendly interface with progressive enhancement

### Client Architecture
- **Distributed Clients**: Python-based monitoring agents deployed across network locations
- **Heartbeat System**: Regular client registration and health checks
- **Modular Testing**: Pluggable test modules for different network metrics
- **Data Collection**: 65+ performance metrics including network, system, and application layer data

## Key Components

### Core Models
- **Client**: Stores client information, system details, and connection status
- **Test**: Defines network monitoring tests with configuration and scheduling
- **TestResult**: Stores collected metrics and performance data (65+ columns)
- **TestClient**: Many-to-many relationship for test assignments
- **User**: Web GUI user accounts with role-based access control
- **ApiToken**: Manages client authentication tokens

### ML Diagnostics Engine
- **Anomaly Detection**: Isolation Forest and statistical models
- **Health Classification**: Random Forest and SVM models
- **Performance Prediction**: Gradient boosting and linear regression
- **Incremental Learning**: Optional River library integration for online learning

### Network Analysis Components
- **Geolocation Service**: IP geolocation lookup with fallback APIs
- **GNMI Client**: Network device telemetry collection (optional)
- **SIP Service**: VoIP analysis and testing capabilities
- **Path Analysis**: Traceroute with hop-by-hop latency measurement

## Data Flow

1. **Client Registration**: Clients register with server using API tokens
2. **Test Assignment**: Server assigns tests to clients based on configuration
3. **Data Collection**: Clients perform network tests and collect system metrics
4. **Result Transmission**: Data sent to server via HTTP API with JSON payload
5. **Storage**: Results stored in database with timestamps and client associations
6. **Visualization**: Web dashboard displays real-time charts and analytics
7. **AI Analysis**: ML models analyze data for anomaly detection and insights

## External Dependencies

### Core Dependencies
- **Flask**: Web framework with SQLAlchemy, Login, and validation
- **psutil**: System monitoring and resource collection
- **requests**: HTTP client for API communication
- **gunicorn**: Production WSGI server
- **psycopg2-binary**: PostgreSQL database adapter

### ML Dependencies
- **scikit-learn**: Machine learning models and preprocessing
- **pandas/numpy**: Data manipulation and analysis
- **joblib**: Model serialization and parallel processing
- **river**: Optional incremental learning library

### Network Testing Dependencies
- **scapy**: Packet manipulation and network analysis
- **speedtest-cli**: Internet speed testing
- **pygnmi**: GNMI protocol support (optional)

### System Dependencies
- **ping/traceroute**: Network connectivity testing
- **tcpdump/libpcap**: Packet capture capabilities
- **wireless-tools**: WiFi environment analysis

## Deployment Strategy

### Development Environment
- Flask development server on localhost:5000
- SQLite database for simplicity
- Debug mode with hot reload
- Development mode bypass for authentication

### Production Environment
- Gunicorn WSGI server with multiple workers
- PostgreSQL database with connection pooling
- Environment variable configuration
- Nginx reverse proxy (recommended)
- SSL/TLS encryption for API endpoints

### Configuration Management
- Environment variables for sensitive settings
- SystemConfig model for runtime configuration
- Development mode toggle for testing
- Configurable client timeouts and intervals

### Security Features
- Input validation and sanitization
- SQL injection protection via SQLAlchemy
- XSS protection with HTML sanitization
- CSRF protection for web forms
- Secure session management
- Role-based access control