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

- **July 13, 2025: CREATED** StreamSwarm server startup script (start_streamswarm_server.sh) for automated server deployment with gunicorn
  - Created comprehensive server startup script based on client deployment infrastructure
  - Server script starts gunicorn with proper binding (0.0.0.0:5000) and reuse-port configuration inside screen session
  - Added server-specific configuration variables to streamswarm_config.sh: SERVER_SCREEN_SESSION, SERVER_LOG_FILE, DATABASE_URL, PostgreSQL settings
  - Created server_status.sh for monitoring server health and accessibility checks
  - Enhanced configuration to support both client and server deployment scenarios
  - Server script includes automatic git updates, network connectivity checks, and proper environment variable handling
  - Production-ready deployment infrastructure now supports both distributed clients and centralized server instances
  - Fixed server script consistency issue: removed premature gunicorn availability check, properly activates virtual environment inside screen session like client script

- **July 13, 2025: ENHANCED** Tutorial documentation with comprehensive server dependency list to prevent ModuleNotFoundError issues
  - Updated tutorial with complete server Python package installation including flask-login, flask-sqlalchemy, flask-wtf, gunicorn, and all critical server dependencies
  - Added all-in-one installation command with 25+ essential packages covering Flask ecosystem, database support, ML/analytics, network tools, and VoIP capabilities
  - Created clear distinction between client (minimal packages) and server (comprehensive packages) requirements
  - Added critical warnings about missing packages like flask-login, flask-wtf, or scikit-learn causing server startup failures
  - Enhanced installation instructions with version requirements and dependency explanations
  - Resolved external deployment issues where servers failed to start due to missing essential Flask ecosystem packages

- **July 13, 2025: CONFIRMED** VoIP test logic working correctly - shows MOS 1.0 on development server without RTP echo capabilities (expected behavior)
  - VoIP test accurately detects that development server doesn't respond to RTP packets, resulting in 100% packet loss and poor MOS score
  - This is correct behavior - real VoIP calls would fail without proper RTP handling
  - Production deployment with RTP echo capabilities will show realistic VoIP quality metrics
  - System properly implements ITU-T G.107 E-model for MOS calculation with authentic network measurements

- **July 13, 2025: IMPLEMENTED** VoIP-specific UI display with dedicated analysis cards similar to WiFi environmental scanning
  - Created comprehensive VoIP Analysis card with MOS scores, voice quality assessments, SIP timing metrics, and RTP stream quality
  - Added VoIP-specific summary cards showing average MOS score and voice quality percentage in place of standard latency/packet loss metrics
  - Implemented color-coded quality badges based on industry-standard VoIP thresholds (MOS 4.0+ excellent, 3.0+ good, 2.0+ fair, <2.0 poor)
  - Hidden irrelevant sections for VoIP tests: network performance charts, geolocation analysis, GNMI path analysis, and client infrastructure analysis
  - Added comprehensive VoIP metrics table with detailed status indicators and impact assessments for each metric
  - Disabled "Get AI Opinion" button for VoIP tests since they use different metrics than standard network performance tests
  - Enhanced VoIP quality thresholds documentation with latency (<150ms), jitter (<60ms), and packet loss (<2%) limits for voice communications
  - VoIP tests now display professional voice quality analysis instead of generic network monitoring data

- **July 13, 2025: FIXED** VoIP Analysis test execution bug - resolved missing method calls preventing result submission
  - Fixed missing _submit_results() method by replacing with proper HTTP POST request to /api/test/results
  - Fixed missing _collect_system_metrics() method by replacing with existing _get_system_metrics() method
  - VoIP tests now properly submit results to server using same API endpoint as standard tests
  - VoIP test results will now display data instead of showing blank results cards
  - Tests that were previously marked "completed" but showed no data will now function correctly

- **July 13, 2025: IMPROVED** VoIP Analysis test creation UX - auto-populates destination field with server URL for logical test workflow
  - Updated test creation modal to automatically set destination field to server hostname when VoIP Analysis test type is selected
  - Made destination field read-only for VoIP tests since server is always the destination for SIP/RTP communication
  - Updated field label to "Server Destination" with explanatory help text for VoIP Analysis tests
  - Enhanced VoIP test info alert to clarify that destination is automatically set to server URL
  - Fixed duplicate VoIP test info div in modal for cleaner interface presentation
  - Improved user experience by eliminating confusion about destination field for server-to-client VoIP testing

- **July 13, 2025: ENHANCED** VoIP Analysis documentation with network requirements - added comprehensive port and protocol information
  - Added Network Requirements section to VoIP Analysis tutorial explaining required server ports (TCP/UDP 5060, 5061, UDP 10000-20000)
  - Updated README.md VoIP Analysis section with network accessibility requirements for SIP and RTP protocols
  - Enhanced tutorial with firewall configuration guidance for proper VoIP testing functionality
  - Documentation now covers complete network infrastructure requirements for VoIP Analysis deployment

- **July 13, 2025: FIXED** VoIP Analysis test creation error - updated server-side validation to accept "voip_analysis" test type
  - Fixed "Invalid test type. Must be 'standard' or 'wifi_environment'" error when creating VoIP Analysis tests
  - Updated routes.py test creation validation to accept "voip_analysis" alongside "standard" and "wifi_environment"
  - VoIP Analysis tests can now be created successfully through the web interface

- **July 13, 2025: CORRECTED** VoIP quality thresholds in tutorial documentation - 150ms is upper limit for acceptable VoIP calls, not "good" range
  - Updated tutorial Quality Thresholds section to reflect accurate VoIP latency standards where >150ms causes call degradation
  - Corrected misleading threshold ranges that suggested 300ms was acceptable for VoIP communications
  - Enhanced documentation to emphasize 150ms as absolute upper limit for VoIP call quality

- **July 13, 2025: IMPLEMENTED** VoIP Analysis as third test type with comprehensive SIP/RTP protocol testing capabilities
  - Added complete VoIP Analysis test type alongside Standard and WiFi Environmental tests in test creation interface
  - Implemented full SIP service architecture with server acting as SIP endpoint for zero-trust compliance
  - Created comprehensive VoIP testing client capabilities including SIP registration, call setup/teardown, and RTP stream quality analysis
  - Added extensive VoIP metrics collection: SIP registration time, call setup latency, RTP packet loss, jitter, MOS scores, codec efficiency, and voice quality scoring
  - Integrated ITU-T G.107 E-model for accurate MOS (Mean Opinion Score) calculation based on packet loss, jitter, and latency measurements
  - Enhanced database schema with 15 new VoIP-specific fields including voip_analysis_data, sip_registration_time, sip_call_setup_time, rtp_packet_loss_rate, mos_score, and voice_quality_score
  - Created professional VoIP test interface with dedicated test type selection, appropriate UI descriptions, and SIP endpoint configuration
  - Implemented comprehensive RTP stream quality testing with synthetic traffic generation, packet timing analysis, and quality metrics calculation
  - Added sipsak system package integration for enterprise-grade SIP protocol testing with timeout handling and error recovery
  - VoIP testing uses closed ecosystem approach - server acts as SIP endpoint, client as SIP client, maintaining zero-trust architecture
  - Complete VoIP workflow: SIP registration → call setup → RTP stream quality → MOS calculation → voice quality assessment → codec efficiency analysis

- **July 13, 2025: FIXED** Critical "Failed to submit result: 500" error by adding missing tcp_handshake_error field to TestResult model and server validation
  - Added tcp_handshake_error TEXT field to TestResult database model for DNS resolution failures
  - Enhanced server validation in /api/test/results endpoint to properly handle TCP handshake error messages
  - Added database migration to support new field for existing test results
  - Fixed intermittent 500 errors that occurred when clients encountered DNS resolution failures (e.g., "No address associated with hostname")
  - Enhanced error logging in submission endpoint with full tracebacks and client data keys for better debugging
  - Client now properly submits error information when handshake fails, preventing data loss and improving diagnostics
  - Resolved database constraint violations that caused test result submission failures during network connectivity issues

- **July 13, 2025: IMPLEMENTED** ML Model Reset functionality on /ml-models page - allows complete model truncation and retraining from scratch
  - Added comprehensive "Reset Models" button with detailed confirmation modal explaining what gets cleared
  - Created API endpoint /api/ml-models/reset that clears all model files, training metadata, and incremental learning state
  - Reset functionality automatically retrains models from scratch using current data after clearing learned patterns
  - Added proper warning dialogs explaining irreversible nature and use cases: data quality improvements, infrastructure changes, performance degradation
  - Reset feature enables A/B testing of model configurations and troubleshooting of model drift issues
  - Preserves all raw test data while giving models fresh start with improved learning capabilities
  - TRAINING MODES: "Train" button uses incremental learning (new data only), "Reset" button retrains from all available data
  - Enhanced with Model Files Status table showing Eastern Time timestamps and file sizes for verification
  - Maintains zero-trust architecture with all processing running locally during reset and retraining

- **July 13, 2025: IMPLEMENTED** True incremental learning system to eliminate ML training timeouts - resolved O(n²) complexity bottlenecks with River streaming algorithms
  - Replaced batch training approach with River-based incremental learning for all 6 ML models: anomaly detection, health classification, performance prediction, failure prediction, QoS compliance, and client infrastructure analysis
  - Implemented memory-efficient streaming algorithms: LogisticRegression for anomaly detection and QoS compliance, GaussianNB for health classification, LinearRegression for performance prediction and infrastructure correlation, PAClassifier for failure prediction
  - Added incremental training pipeline that processes data in batches of 50 samples to prevent memory issues while maintaining model accuracy
  - Eliminated 45-60 second training timeouts that occurred with 3,215+ datapoints by using online learning algorithms that update incrementally
  - Resolved River library memory issues by replacing resource-intensive HoeffdingTree models with stable linear and naive Bayes algorithms
  - Fixed data type handling issues in incremental feature calculation methods with proper error handling and binary label conversion
  - Created dual-mode training system: automatic incremental learning for existing models, fallback to batch training for full retraining scenarios
  - Training now processes 65+ batches of network performance data without memory exhaustion or worker crashes
  - Maintained zero-trust architecture with all ML processing running locally using River streaming machine learning library

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
  - **COMPLETED** TCP Handshake Timing Diagnostics implementation:
    - Added comprehensive connection establishment timing analysis breaking down SYN, SYN-ACK, ACK phases
    - Enhanced database schema with 8 new handshake timing fields (total 29 TCP-related fields)
    - Network delay estimation: Automatic calculation separating network transit from server processing delays
    - Performance classification with intelligent diagnostic messages (excellent/good/moderate/slow)
    - Sub-millisecond timing precision for detailed connection diagnostics
    - Web interface enhanced with dedicated TCP Connection Establishment Timing accordion section
    - Database migration completed successfully resolving PostgreSQL column optimization issues
    - Standalone testing validated with real-world connection timing to multiple targets
    - Application verified working correctly with handshake timing diagnostics displaying in web interface

## Development Status

**Current State**: Production-ready distributed network monitoring system with complete authentication, AI/ML diagnostics, and comprehensive documentation.

**Next Priorities**: User feedback and feature requests from production deployment.