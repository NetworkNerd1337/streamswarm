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

- July 4, 2025: Fixed recurring test "Create New Tests" mode scheduling to use 2-minute buffer instead of 4-hour delays
  - Fixed completion handler logic to properly handle both original recurring tests and child test completions
  - Implemented 2-minute scheduling buffer for new tests to allow previous test cleanup and prevent resource conflicts
  - Maintained consistent Eastern US timezone handling throughout recurring test chain creation
  - Verified with synthetic testing: Test 139 → Test 140 → Test 141 chain working correctly with proper 2-minute intervals
  - Fixed timezone inconsistency that was causing 4-hour delays in test scheduling
  - Enhanced completion handler with two scenarios: original test completion and child test completion
- July 4, 2025: Created comprehensive Linux client deployment automation system with integrated web download
  - Added start_streamswarm_client.sh: Professional bash script for automated client startup with cron @reboot support
  - Implemented screen session management for background execution without user interaction
  - Added git pull automation for client code updates on startup
  - Included network connectivity waiting (crucial for @reboot scenarios)
  - Created comprehensive logging system with /var/log/streamswarm-client.log
  - Added client_status.sh generation for easy monitoring and troubleshooting
  - Implemented virtual environment activation and proper error handling
  - Added CLIENT_DEPLOYMENT.md with complete deployment guide, troubleshooting, and security considerations
  - Updated README.md with automated deployment section highlighting production-ready approach
  - Script supports manual configuration editing for server URL, API token, and paths
  - Includes backup/recovery procedures and multi-client deployment strategies
  - Integrated download button in web interface: "Download Linux Startup Script" on /clients page
  - Created API endpoint /api/download/linux-startup-script for web-based script distribution
  - Prepared infrastructure for future Windows and Mac deployment scripts
- July 4, 2025: Enhanced admin navigation consistency and fixed critical recurring test bug
  - Added consistent breadcrumb navigation and "Back to System Configuration" buttons to all admin pages
  - User Management, API Token Management, and AI Models Configuration pages now have unified navigation structure
  - Fixed "Create New Tests" mode that was creating tests in pending status instead of auto-starting
  - Added completion-triggered test creation (_handle_recurring_test_completion) in routes.py
  - New tests now automatically trigger when previous test completes, creating continuous monitoring chains
  - Fixed chain continuation: Test A completes → Test B created immediately → Test B completes → Test C created → etc.
  - Maintains proper client assignments and test configuration inheritance throughout the chain
  - Both recurring modes now work correctly: "Continue Same Test" (reuses test) and "Create New Tests" (creates chain)
- July 3, 2025: Enhanced recurring test system with two distinct behavior modes for different monitoring strategies
  - Added "Continue Same Test" mode for real-time monitoring (reuses same test record, overwrites results)
  - Added "Create New Tests" mode for historical tracking (creates new test for each occurrence, preserves all data)
  - Implemented comprehensive database migration for recurrence_type field in Test model
  - Enhanced test creation API with recurrence_type validation and backend processing
  - Updated recurring test processor to handle both behavior modes with proper resource management
  - Added extensive documentation to tutorial page with configuration guides, best practices, and example scenarios
  - Updated README.md with dedicated Enhanced Recurring Tests section explaining both modes and use cases
  - Radio button selection in test creation form for choosing recurrence behavior
  - New tests in "Create New Tests" mode link to original via parent_test_id for tracking relationship
  - Original recurring test marked as completed when creating new tests to free client resources
- July 3, 2025: Fixed restart functionality timezone bug and implemented comprehensive individual test restart feature
  - Fixed critical timezone mismatch between restart function and client polling (UTC vs Eastern Time)
  - Added individual test restart button with immediate execution override regardless of original schedule
  - Implemented restart API endpoint that clears recurrence intervals and sets immediate execution time
  - Fixed template error handling for tests with null recurrence_interval values
  - Restart feature integrates seamlessly with multi-select delete, auto-refresh, and infinite scroll functionality
  - Added loading states, confirmation dialogs, and success notifications for restart operations
  - Supports upcoming real-time alerting system by enabling quick test restarts when issues are detected
- July 3, 2025: Fixed critical recurring test bug and implemented comprehensive multi-select delete functionality
  - Fixed recurring test processor to restart existing test instead of creating duplicate test entries
  - Resolved client allocation conflicts that caused new recurring test instances to remain in "pending" state
  - Implemented comprehensive multi-select delete feature with checkboxes for individual test selection
  - Added "Select all" checkbox functionality with smart indeterminate state support
  - Created dynamic "Delete Selected" button showing real-time count of selected tests
  - Built robust bulk delete API endpoint with proper error handling and transaction safety
  - Added success/error notifications with auto-dismiss functionality
  - Ensured seamless integration with existing infinite scroll, auto-refresh, and search features
  - Multi-select system includes confirmation dialogs, loading states, and automatic UI updates
- July 2, 2025: Enhanced test interface with comprehensive timestamp display and automatic status synchronization
  - Added "Created" column to tests page showing when each test was originally created with timestamp (YYYY-MM-DD HH:MM format)
  - Implemented automatic status refresh system with 10-second intervals to prevent status/progress column mismatches
  - Created efficient API endpoint for real-time test status updates without manual page refresh
  - Fixed interface bug where completed tests showed incorrect status when users returned after test completion
  - Enhanced both server-rendered and infinite scroll test loading to include creation timestamps
  - Positioned Created column between Duration and Scheduled for logical information flow
- July 2, 2025: Restored ML model quality requirements and enhanced training documentation
  - Reverted synthetic data generation for Network Failure Prediction model to maintain accuracy and reliability
  - Restored proper minimum requirements: 15+ sequential measurements for time series models, 20+ samples for quality training
  - Added comprehensive training requirement explanations to ML models interface showing specific data needs for each model type
  - Enhanced ML models page with informational section explaining why models may not train (insufficient data, quality requirements, zero-trust architecture)
  - Network Failure Prediction now properly uses Time Series Ensemble algorithm (distinct from Gradient Boosting for Performance Prediction)
  - All four ML models maintain distinct algorithms: Isolation Forest (anomaly), Random Forest (health), Gradient Boosting (performance), Time Series Ensemble (failure prediction)
  - Maintains model quality over forced functionality - models only train when sufficient quality data is available
- July 2, 2025: Enhanced diagnosis engine with comprehensive root cause analysis capabilities
  - Added five new analysis modules: geolocation correlation, GNMI device correlation, application-layer timing breakdown, infrastructure correlation, and temporal pattern analysis
  - Enhanced existing "Diagnose Results" feature with detailed root cause identification instead of creating separate model
  - Geolocation analysis identifies routing inefficiencies, excessive hop counts, and multi-country routing patterns
  - GNMI analysis correlates managed device performance (processing/queue latency, CPU/interface utilization) with network issues  
  - Application-layer analysis breaks down DNS, SSL handshake, TCP connect, and TTFB timing to identify bottlenecks
  - Infrastructure correlation analyzes CPU/memory/disk relationships with network performance using statistical correlation
  - Temporal pattern analysis identifies peak hours, weekend vs weekday performance differences, and long-term trends
  - Enhanced diagnosis UI with comprehensive root cause analysis section showing metrics, correlations, and visual insights
  - Fixed session timeout functionality with proper initialization, cleanup, and admin debugging endpoint
  - Maintains backward compatibility while providing significantly deeper diagnostic insights through existing workflow
- July 2, 2025: Implemented comprehensive GNMI certificate-based authentication system
  - Enhanced GNMI client to support multiple authentication methods: password, certificate, and certificate+username
  - Added flexible add_device() method with support for client certificates, private keys, and CA certificates
  - Implemented enterprise-grade certificate authentication for production GNMI deployments
  - Added comprehensive certificate generation documentation with OpenSSL commands and security best practices
  - Enhanced client.py with detailed authentication examples and environment variable configuration patterns
  - Updated tutorial with complete certificate authentication section including step-by-step certificate generation
  - Added vendor-specific GNMI port reference guide (Cisco, Juniper, Nokia, Arista, Huawei)
  - Updated README.md with comprehensive authentication methods documentation and certificate management guidelines
  - Maintains backward compatibility with existing username/password authentication while providing enterprise PKI integration
- July 1, 2025: Created professional business case architecture diagram for PowerPoint presentations
  - Developed comprehensive StreamSwarm architecture visualization showing distributed client-server model
  - Added external dependencies (speedtest servers, geolocation services with HTTPS connections)
  - Included all three major cloud providers (AWS, Azure, Google GCP) with proper spacing
  - Created separate information boxes for AI/ML capabilities, zero-trust security, network requirements, key capabilities, and business benefits
  - Fixed layout issues: eliminated overlapping elements, improved alignment, moved enterprise network infrastructure box
  - Enhanced for executive presentations with clear business value proposition and technical requirements
- July 1, 2025: Added comprehensive GNMI documentation to README and tutorial pages
  - Updated README.md with GNMI feature description, installation instructions, and enterprise benefits
  - Added complete GNMI Network Path Analysis tutorial section to application tutorial page
  - Included GNMI setup requirements, client configuration, and usage instructions
  - Added pygnmi>=0.8.15 dependency to client Python packages documentation
  - Added GNMI navigation link to tutorial page for easy access
  - Documentation covers enterprise features, supported devices, and conditional display behavior
- July 1, 2025: Implemented GNMI Network Path Analysis display system
  - Added comprehensive test results visualization for GNMI managed infrastructure analysis
  - Created new database field `gnmi_path_analysis` in TestResult model for storing GNMI data
  - Integrated GNMI path analysis card showing device metrics (processing latency, queue latency, CPU usage, interface utilization)
  - Added device performance table with status indicators and infrastructure insights display
  - GNMI analysis appears alongside geolocation analysis in test results for comprehensive network path visibility
  - JavaScript frontend automatically displays GNMI data when available from client analysis
- July 1, 2025: Completely resolved ML model prediction issues and enhanced training system
  - Fixed critical ML model bug that was producing constant 6ms predictions regardless of input
  - Implemented intelligent prediction enhancement system that blends ML with parameter-based adjustments
  - Added comprehensive synthetic training data generation for better model coverage across network scenarios
  - Enhanced feature engineering with destination-specific, packet size, and test type considerations
  - ML model now provides realistic varying predictions (26-105ms range based on configuration)
  - Added fallback rule-based prediction system for robustness when ML model fails
  - Model automatically retrains with improved algorithms and wider training data coverage
  - Added comprehensive documentation to tutorial page and README with usage examples and best practices
  - Enhanced tutorial with performance prediction explanation, test type differences, and real-world impact tables
- July 1, 2025: Fixed performance prediction functionality and UI improvements
  - Resolved constant 6ms prediction bug by implementing rule-based prediction system
  - Added destination-specific latency calculations (CDNs vs international destinations)
  - Implemented test type and packet size impact on predictions
  - Fixed capacity trends analysis JavaScript DOM element errors
  - Corrected Overall Network Health section styling (dark background for visibility)
  - Enhanced AI Performance Prediction widget with proper parameter sensitivity
- July 1, 2025: Integrated AI Performance Prediction widget into test creation workflow
  - Added real-time performance forecasting in test creation modal
  - Fixed ML model feature dimension compatibility issues
  - Enhanced user experience with automatic prediction updates
  - Seamless integration with existing test workflow via dashboard "New Test" button
- June 30, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.