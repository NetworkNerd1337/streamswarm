# StreamSwarm - Network Monitoring System

## Overview

StreamSwarm is a Python-based distributed network monitoring system that enables real-time network performance testing and system resource monitoring across multiple client hosts. The system features a centralized web dashboard for managing clients, scheduling tests, and visualizing performance data.

## System Architecture

The application follows a client-server architecture with:

**Backend**: Flask web application with SQLAlchemy ORM for data persistence
**Frontend**: Bootstrap-based responsive web interface with Chart.js for data visualization
**Database**: SQLite for development (configurable to PostgreSQL via environment variables)
**Client Architecture**: Standalone Python clients that connect to the central server
**Deployment**: Gunicorn WSGI server with autoscaling support

## Key Components

### Core Application (`app.py`)
- Flask application factory with SQLAlchemy database integration
- Environment-based configuration for database connections
- ProxyFix middleware for deployment behind reverse proxies
- Automatic database table creation on startup

### Data Models (`models.py`)
- **Client**: Stores client information, system details, and connection status
- **Test**: Defines network tests with scheduling and configuration parameters
- **TestResult**: Stores individual test measurements and metrics
- **TestClient**: Junction table for many-to-many client-test relationships

### Web Interface (`routes.py`)
- Dashboard with system overview and real-time statistics
- Client management interface for monitoring connected hosts
- Test creation, scheduling, and results visualization
- RESTful API endpoints for client communication

### Standalone Client (`client.py`)
- Independent monitoring client with system resource tracking
- Network performance testing capabilities (ping, traceroute)
- Automatic server registration and heartbeat mechanism
- Multi-threaded architecture for concurrent test execution

### Configuration Management (`config.py`)
- Environment variable-based configuration
- Separate settings for server, client, and testing parameters
- Security and logging configuration options

## Data Flow

1. **Client Registration**: Clients connect to server and register with system information
2. **Test Scheduling**: Web interface allows creation of tests targeting specific destinations
3. **Test Distribution**: Server assigns tests to available clients based on scheduling
4. **Data Collection**: Clients execute tests and stream results back to server
5. **Data Visualization**: Web dashboard displays real-time charts and historical data

## External Dependencies

### Python Packages
- **Flask**: Web framework and application structure
- **SQLAlchemy**: Database ORM and migrations
- **psutil**: System resource monitoring (CPU, memory, disk)
- **requests**: HTTP client for server-client communication
- **gunicorn**: Production WSGI server

### Frontend Libraries
- **Bootstrap**: Responsive UI framework with dark theme
- **Chart.js**: Interactive data visualization
- **Font Awesome**: Icon library for UI elements

### Infrastructure
- **PostgreSQL**: Production database (via environment configuration)
- **OpenSSL**: Secure communications support

## Deployment Strategy

The application is configured for Replit's autoscaling deployment platform:

- **Development**: Flask development server with hot reload
- **Production**: Gunicorn WSGI server with multiple workers
- **Database**: Environment-configurable (SQLite for dev, PostgreSQL for production)
- **Static Assets**: Served via CDN for performance
- **Process Management**: Replit's workflow system for application lifecycle

The deployment supports both single-instance development and horizontally scaled production environments through environment variable configuration.

## Recent Changes

- June 22, 2025: Major metrics enhancement and logo integration
  - Added 35+ new performance metrics including DNS resolution time, TCP connect time, SSL handshake time, TTFB, jitter analysis
  - Enhanced CPU monitoring with load averages, frequency, temperature, context switches
  - Detailed memory analysis with available/cached/buffered/shared memory tracking
  - Storage performance metrics including IOPS, throughput, and disk temperature
  - Network interface statistics with bytes/packets sent/received, errors, and drops
  - Process and system activity monitoring (process count, TCP connections, file descriptors)
  - Database schema automatically migrated with new metric columns
  - Professional logo integration across all pages in navigation and headers
  - Enhanced visual design with consistent logo placement and improved page layouts
  - Created comprehensive Enhanced Metrics Dashboard with categorized metric overview
  - Updated test results page with dropdown selector for all 40+ available metrics
  - Fixed dark mode compatibility for metric cards with proper contrast and readability

- June 21, 2025: Complete implementation of StreamSwarm network monitoring system
  - PostgreSQL database integration with all tables created
  - Full client-server architecture with real network testing capabilities
  - Web dashboard with interactive charts and real-time updates
  - Client details modal with actual data from database
  - Test management with stop/delete functionality
  - Real-time test progress tracking with automatic completion
  - Comprehensive usage documentation and deployment guide
  - In-depth instructional tutorial page with real-world use cases
  - Interactive web-based tutorial with step-by-step guides

## User Preferences

Preferred communication style: Simple, everyday language.