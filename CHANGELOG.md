# StreamSwarm - Development Changelog & Technical Architecture

## Overview

StreamSwarm is a comprehensive Python-based distributed network monitoring system with Flask-based web GUI, AI/ML diagnostics, and secure authentication. This file contains technical architecture details and development changelog for internal reference.

## Key Design Decisions

1. **Client-Server Model**: Enables distributed monitoring from multiple network locations
2. **Web-based Dashboard**: Provides real-time visualization and test management  
3. **SQLAlchemy ORM**: Enables database flexibility and easy schema management
4. **Dual Authentication**: Separate systems for web GUI (Flask-Login) and client API (tokens)
5. **Zero-Trust ML**: Local Scikit-learn processing with no external dependencies
6. **Modular Design**: Separate client and server components for flexible deployment

## Database Schema

- **Client**: Stores client information and system details
- **Test**: Defines network monitoring tests with configuration
- **TestResult**: Stores collected metrics and performance data (65+ columns)
- **TestClient**: Many-to-many relationship for test assignments
- **ApiToken**: Manages client authentication tokens
- **User**: Web GUI user accounts with role-based access
- **OAuth**: Session storage for Flask-Login authentication

## Data Flow

1. **Client Registration**: Clients register with server and receive API tokens
2. **Test Assignment**: Server assigns tests to specific clients based on configuration
3. **Metric Collection**: Clients perform network tests and collect system metrics
4. **Data Transmission**: Results sent to server via HTTP API with JSON payload
5. **Data Storage**: Server stores results in database with timestamp and client association
6. **Visualization**: Web dashboard displays real-time charts and historical data
7. **AI Analysis**: ML models analyze data for anomaly detection and health classification

## Deployment Strategy

### Development
- Flask development server on port 5000
- SQLite database for simplicity
- Debug mode enabled

### Production  
- Gunicorn WSGI server with multiple workers
- PostgreSQL database for scalability
- Environment variable configuration for secrets

## External Dependencies

### Python Libraries
- **Flask/SQLAlchemy**: Web framework and database ORM
- **psutil**: System resource monitoring
- **requests**: HTTP client for API communication
- **speedtest-cli**: Internet bandwidth testing
- **scapy**: Advanced network packet analysis
- **matplotlib/reportlab**: Chart generation and PDF reports
- **scikit-learn/pandas/numpy**: AI/ML diagnostic capabilities

### System Dependencies
- **Network Tools**: ping, traceroute, iw for network testing
- **System Tools**: lm-sensors, smartmontools for hardware monitoring
- **Database**: PostgreSQL support via psycopg2-binary

## Changelog

- **June 29, 2025: COMPLETED** secure web GUI authentication system with role-based access control
  - Implemented Flask-Login based authentication system separate from client API token system
  - Added comprehensive user management interface accessible only to admin users
  - Created secure login page with dark theme matching application aesthetics and hashed password storage using Werkzeug
  - Implemented role-based access control with admin and user roles
  - Added user CRUD operations: create, read, update, delete users with security constraints
  - Protected all web GUI routes with authentication decorators (@web_auth_required, @admin_required)
  - Enhanced navigation with user dropdown menu showing current user and admin badge
  - Automatic default admin user creation: username=admin, password=admin123 (change on first login)
  - User management features: username/email validation, password hashing, active/inactive status
  - Security features: prevent admin self-modification, preserve last admin user, email format validation
  - Authentication system completely separate from client API token system - no interference with existing client connections
  - Added user profile page with self-service password change functionality for all users
  - Styled login page with dark gradient background and proper contrast to match application theme
  - Integrated SWARM logo into login page with attractive styling and hover animations
  - Fixed security vulnerability: added authentication protection to all ML-related routes (/ml-models, ML training, diagnosis)
  - Updated all documentation (README.md, USAGE.md, TUTORIAL.md, in-app tutorial) to reflect authentication system
  - Verified comprehensive authentication protection across all web GUI routes and features
  - Database initialization automatically creates authentication tables and default admin account on first startup
  - **CONSOLIDATED DOCUMENTATION**: Merged README.md, TUTORIAL.md, USAGE.md, and replit.md into single comprehensive README.md

- **June 28, 2025: COMPLETED** comprehensive AI/ML diagnostic system with zero-trust architecture
  - Added local machine learning capabilities using Scikit-learn for network performance analysis
  - Implemented ensemble approach with Isolation Forest (anomaly detection), Random Forest (health classification), and Gradient Boosting (performance prediction)
  - Created "Diagnose Results" button functionality for AI-powered test analysis with health scoring and issue detection
  - Added "AI Models" management section with training interface and model status monitoring
  - Enhanced feature engineering to analyze 25+ metrics including network performance, system resources, and QoS data
  - Implemented zero-trust compliance with all ML processing running locally with no external dependencies
  - Added comprehensive documentation for AI/ML setup, usage instructions, and Linux package dependencies
  - Updated TUTORIAL.md, README.md, USAGE.md, and DEPENDENCIES.md with AI/ML information and installation instructions
  - Models provide intelligent recommendations, issue categorization, and troubleshooting guidance for network administrators
  - **RESOLVED** all ML training issues: division by None errors, datetime comparison bugs, and feature extraction problems
  - **VERIFIED** full functionality: 769 test results → 99% classification accuracy → successful model persistence and loading
  - **FIXED** Network Performance Metrics display bugs in test results accordion:
    - Removed misleading Network Bytes Sent/Received metrics (showed system-wide traffic like 26.8GB instead of test-specific data)
    - Fixed MTU Size metric that always showed "N/A bytes" - now properly extracts MTU from JSON network interface data
    - MTU Size now displays actual values (e.g., "1500 bytes" with "Standard" badge) from collected network interface information
  - **IMPLEMENTED** Development Mode feature for easier troubleshooting and development:
    - Added SystemConfig model to store development mode setting in database
    - Created admin-only toggle to temporarily disable authentication while preserving user accounts
    - Updated authentication decorators to respect development mode bypass
    - Added "Dev Mode" navigation link and control page for admin users
  - **ENHANCED** Error handling and user experience:
    - Fixed login page internal server errors caused by malformed URL parameters (e.g., /login%3Fnext=%)
    - Added comprehensive error handlers for 400, 404, and 500 error pages with styled error pages
    - Implemented graceful error recovery with user-friendly messages instead of white error screens
  - **IMPROVED** Authentication system usability:
    - Implemented case-insensitive username authentication for better user experience
    - Users can now login with username in any case variation (admin, ADMIN, Admin, etc.)
    - Updated all username uniqueness checks to be case-insensitive across the system
  - **RESTRUCTURED** System Configuration organization:
    - Created dedicated System Configuration page accessible from user account dropdown (admin only)
    - Moved Development Mode settings from main navigation to System Configuration subsection
    - Added breadcrumb navigation and organized foundation for future system settings
    - Improved intuitive access to admin features through structured configuration dashboard
  - **ADDED** TCP Window Analysis feature for advanced network bottleneck attribution:
    - Real-time TCP window behavior monitoring during connection establishment
    - Comprehensive RTT analysis (min/max/avg/variation) for network stability assessment  
    - Congestion window tracking to identify throughput limitations
    - Automated bottleneck classification (network congestion, server limited, optimal, etc.)
    - TCP window efficiency scoring (0-100%) based on stability and performance
    - Retransmission and congestion event detection for packet loss attribution
    - 13 new database fields for storing detailed TCP connection metrics
    - Visual dashboard with efficiency scores and bottleneck type breakdown
    - Professional interpretation guides for network administrators
  - **COMPLETED** TCP Window Analysis testing and validation:
    - Core TCP analysis functions validated with proper database schema (21 TCP-related fields total)
    - Bottleneck detection working correctly with classification types: network_congestion, server_limited, network_instability, packet_loss, optimal
    - Confirmed timing: captures data every test interval throughout entire duration (every 30 seconds for 2-hour tests = 240 separate analyses)
    - Client code requirements clarified: uses built-in socket libraries and /proc filesystem, no additional Linux packages needed
    - Clients require updated client.py code to enable TCP monitoring capabilities for live production testing
  - **RESOLVED** Critical Jinja2 template compatibility issue:
    - Fixed web interface template errors: Resolved multiple Jinja2 groupby filter issues causing 500 internal server errors
    - Root cause: Jinja2 3.1.6 compatibility problem with map(attribute) filter chains combined with select('number') and list operations
    - Created custom extract_numeric filter to safely extract numeric values from result objects
    - Systematically replaced all 32 problematic filter patterns throughout template system
    - Fixed Jinja2 syntax error with invalid break statement (not supported in Jinja2 loops)
    - Verified complete functionality: test results pages now load correctly with all TCP window analysis features operational

## Development Status

**Current State**: Production-ready distributed network monitoring system with complete authentication, AI/ML diagnostics, and comprehensive documentation.

**Next Priorities**: User feedback and feature requests from production deployment.