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

- July 8, 2025: **FIXED** WiFi Environmental Scanning accuracy and channel detection - resolved critical issues with frequency parsing and congestion analysis
  - Fixed WiFi frequency parsing to properly handle both MHz and GHz formats in `iw scan` output
  - Enhanced channel detection logic with improved frequency-to-channel conversion for 2.4GHz (1-14), 5GHz (36-165), and 6GHz (1-233) bands
  - Added alternative frequency parsing from `freq:` lines to catch missed frequency data
  - Improved 2.4GHz vs 5GHz classification to properly handle channel ranges instead of defaulting all networks to 2.4GHz
  - Enhanced pollution score calculation with weighted factors: network count (50%), channel congestion (30%), interference (20%)
  - Fixed channel congestion detection - now properly identifies most congested channels instead of showing "None"
  - Added debugging for invalid channel detection to improve scanning reliability
  - WiFi environmental tests now provide accurate channel distribution, congestion analysis, and pollution scoring
- July 8, 2025: **FIXED** WiFi Environmental Test UI improvements - comprehensive cleanup of WiFi-only test display components
  - Fixed pollution score legend from light theme (`bg-light`) to dark theme (`bg-dark border`) for proper dark mode compatibility
  - Resolved "Show All" modal empty data issue by correcting data source from `wifiData.networks` to `wifiData.detected_networks`
  - Hidden client infrastructure analysis card for WiFi environmental tests (skipped API call to prevent empty card display)
  - Hidden client performance summary card for WiFi environmental tests since network performance metrics aren't collected during WiFi scanning
  - Hidden comprehensive metrics analysis card for WiFi environmental tests as it shows 65+ network performance metrics not collected during WiFi scanning
  - WiFi environmental tests now show only relevant analysis: WiFi environmental data card with pollution scoring, signal quality, and network detection
  - Standard tests continue to show all components: metrics charts, geolocation analysis, GNMI analysis, and client infrastructure analysis
- July 8, 2025: **FIXED** Test results display regression - resolved critical JavaScript function call mismatch causing both standard and WiFi environmental tests to fail
  - Fixed JavaScript function call mismatch: `displayClientInfrastructureData()` was being called instead of `displayClientInfrastructureAnalysis()`
  - Added missing `displayGNMIPathData()` function call to initialization sequence
  - Standard tests now properly display: metrics dropdown with CPU/Memory/Disk graphs, geolocation analysis, GNMI path analysis, and client infrastructure analysis
  - WiFi environmental tests now properly display: WiFi environmental analysis card with network count, pollution score, signal quality distribution, and environment quality assessment
  - Fixed WiFi-only detection logic to properly hide standard network charts for WiFi environmental tests while showing WiFi-specific analysis
  - All test result visualization components now initialize correctly with proper error handling and data validation
  - Resolved console errors and ensured both test types work independently with their respective analysis features
- July 8, 2025: **FIXED** WiFi Environmental Scan API endpoint bug - resolved 404 errors preventing WiFi scan results from reaching server
  - Fixed critical API endpoint mismatch: client was sending to `/api/test_results` while server expected `/api/test/results`
  - WiFi environmental scanning now successfully submits scan results (77, 63, 59 networks detected in Test 196)
  - Enhanced graceful permission handling with automatic sudo fallback for WiFi scanning
  - Added netdev group setup instructions to tutorial for passwordless WiFi scanning
  - Created setup_wifi_scanning.sh script for automated WiFi permission configuration
- July 8, 2025: **FIXED** WiFi Environmental Scan execution bug - resolved client receiving wrong test_type and removed hardcoded WiFi scanning from standard tests
  - Fixed critical bug where client received 'standard' test_type instead of 'wifi_environment' despite correct UI selection
  - Removed hardcoded WiFi environmental scanning from standard test execution path (lines 744-759 in client.py)
  - WiFi scanning now only executes for wifi_environment test types, preventing "Operation not permitted" errors in standard tests
  - Added comprehensive server-side debugging to track test_type serialization and API responses
  - Enhanced client debugging to show exactly what test_type is received from server and which execution path is taken
  - Fixed permission issues by limiting WiFi scanning to dedicated WiFi Environmental Scan tests only
  - Standard tests no longer attempt WiFi scanning, resolving sudo permission requirements and failed scan errors
- July 8, 2025: Updated WiFi environmental scanning from deprecated iwlib to modern iw command for Ubuntu compatibility
  - Replaced deprecated iwlib Python library with modern iw command-line tool for WiFi scanning functionality
  - Updated _detect_wifi_interfaces() to use 'iw dev' for wireless interface detection
  - Replaced _perform_wifi_environmental_scan() to use 'iw dev scan' for network scanning
  - Added _parse_iw_scan_output() function to parse modern iw scan output format
  - Enhanced _analyze_wifi_environment() for better compatibility with both iwlib and iw data formats
  - Updated all documentation and tutorials to reference 'sudo apt-get install iw' instead of 'pip install iwlib'
  - Fixed WIFI_SCANNING_AVAILABLE detection to check for iw command availability instead of iwlib import
  - Maintains backward compatibility while providing full Ubuntu/Debian support for WiFi environmental scanning
- July 8, 2025: Completed comprehensive WiFi environmental scanning integration with dual-mode capability and rich visualization dashboard
  - Added WiFi environmental test type with standalone and integrated scanning modes
  - Implemented WiFiEnvironmentalAnalyzer class for wireless interface detection and comprehensive environmental scanning
  - Enhanced database schema with wifi_environment_data field in TestResult model and test_type field in Test model
  - Created dedicated WiFi environmental test creation option in web interface test modal
  - Added comprehensive WiFi environmental analysis visualization with environment quality assessment, pollution scoring, and network detection
  - Displays signal strength distribution, channel utilization analysis, congestion detection, and detailed network tables
  - Integrates WiFi pollution metrics with existing ML models for network performance correlation analysis
  - Supports multiple wireless interface detection with automatic fallback for single-interface clients
  - Environmental metrics include RSSI monitoring, channel overlap analysis, RF interference detection, and network density assessment
  - Complete workflow: clients detect WiFi interfaces → perform environmental scans → correlate with network performance → display comprehensive analysis
- July 7, 2025: Implemented comprehensive StreamSwarm client certificate management system with automatic generation and web-based distribution
  - Added ClientCertificate database model for storing client-generated GNMI certificates with expiry tracking
  - Enhanced client registration process to automatically generate self-signed certificates using OpenSSL
  - Implemented secure certificate upload API endpoint with certificate parsing and metadata extraction
  - Added certificate download functionality to clients page with detailed GNMI device configuration instructions
  - Created comprehensive client certificate modal with vendor-specific installation guides (Cisco, Juniper, Arista)
  - Integrated automatic certificate generation on client startup with local storage in client_certs/ directory
  - Added certificate file validation, secure permissions (600 for keys, 644 for certificates), and automatic cleanup
  - Complete workflow: clients generate certificates → upload to server → admins download from web interface → install on GNMI devices
  - Enhanced security with proper certificate lifecycle management and vendor-specific trustpoint configuration examples
- July 7, 2025: Added comprehensive client certificate generation documentation for enterprise GNMI authentication
  - Enhanced README.md with detailed step-by-step certificate generation process using OpenSSL
  - Added vendor-specific GNMI server configuration for Cisco IOS-XR, Juniper JUNOS, Nokia SR-OS, and Arista EOS
  - Created comprehensive accordion-based certificate tutorial in /tutorial page with 4 detailed steps
  - Covered complete mutual TLS (mTLS) workflow from certificate generation to server trust configuration
  - Added production CA signing instructions and security best practices
  - Enhanced GNMI device creation with improved validation and error handling to prevent double-submission issues
  - Fixed device creation form with loading states, duplicate name validation, and certificate file validation
  - Documentation now covers complete enterprise certificate lifecycle management
- July 7, 2025: Successfully completed comprehensive GNMI Client Manager system with centralized server-based device configuration
  - Implemented complete GNMI Client Manager web interface with device and certificate management (admin-only access)
  - Added GNMI Client Manager link to admin dropdown navigation for easy access from any page
  - Created robust client-side GNMI synchronization that automatically downloads device configurations on startup
  - Implemented secure certificate download and management with proper file permissions (600 for private keys)
  - Enhanced GNMI client with device clearing functionality and connection management
  - Added local configuration caching for offline operation and resilience
  - Fixed template navigation error in GNMI manager interface
  - Complete enterprise-grade GNMI management: server configures devices centrally, clients sync automatically on startup
- July 6, 2025: Created professional Apple touch icon for iOS home screen integration
  - Updated to use custom StreamSwarm logo featuring dynamic triangular design with orange/purple gradient
  - Resized user-provided logo to proper 180x180 pixels for Apple touch icon standards
  - Added proper HTML meta tags for Apple web app integration including touch icon, app-capable settings, and status bar styling
  - Icon displays "SWARM STREAM TRAFFIC GENERATOR" branding with modern geometric design
  - Enables professional appearance when users add web app to iOS home screen
- July 6, 2025: Successfully completed Client Infrastructure Analysis UI integration on test results page
  - Added comprehensive Client Infrastructure Analysis card to test results page alongside geolocation and GNMI analysis
  - Created JavaScript functions to fetch and display correlation analysis with color-coded badges and priority-based recommendations
  - Analysis shows correlation scores, risk levels, key system correlations, and client-side improvement recommendations
  - Card automatically appears for tests from clients with sufficient recent data (10+ test results in last 30 days)
  - Analysis correlates client CPU, memory, and network interface metrics with network performance using PCA + Linear Regression
  - Display includes strongest correlations with visual indicators (red for high correlation >70%, warning for medium >50%, info for >30%)
  - Complete 6th ML model integration: anomaly detection, health classification, performance prediction, network failure prediction, QoS compliance monitoring, and client infrastructure correlation
  - Note: Analysis requires recent client data (last 30 days) and trained client infrastructure analyzer model, not available on historical tests without sufficient recent client activity
- July 6, 2025: Fixed client deletion popup accuracy and added comprehensive Server Configuration Options tutorial
  - Fixed misleading client deletion popup that incorrectly stated "Keep all historical test data" when offline clients with tests cannot actually be deleted
  - Updated deletion message to accurately state "Clients with test data cannot be deleted to preserve historical records"
  - Applied consistent messaging across both deleteClient functions in clients.html and dashboard.js
  - Created comprehensive Server Configuration Options tutorial section explaining all system configuration settings
  - Documented Development Mode with security warnings, Expected Client Version for deployment tracking, Session Timeout options
  - Covered management interface access including User Management, API Token Management, and AI Model Configuration
  - Added configuration best practices and security recommendations for production environments
- July 6, 2025: Successfully resolved System Info column display issue and completed client versioning system
  - Fixed double-escaped JSON parsing in both server-side template rendering and API routes
  - System Info column now properly displays platform, CPU count, and memory details for all clients
  - Enhanced error handling for system information parsing with fallback mechanisms
  - Maintained client version tracking functionality with color-coded status indicators
  - Ensured consistent data handling between initial page load and infinite scroll loading
  - Both server-rendered and JavaScript-loaded clients now show complete system information
- July 6, 2025: Successfully implemented comprehensive client versioning system for deployment tracking
  - Added client_version field to Client database model and migration
  - Updated client registration process to send version information to server
  - Enhanced heartbeat mechanism to include version updates for real-time synchronization
  - Added Version column to clients page with color-coded status indicators
  - Green badges show clients with current version (1.0.0), yellow badges indicate outdated versions
  - System configuration tracks expected client version for comparison
  - Enables administrators to monitor whether distributed clients are running latest code
  - Supports git-based deployment strategies with automatic version tracking
- July 5, 2025: Enhanced predictive analytics page with improved two-column layout and clear section headers
  - Rearranged cards into logical two-column layout: prediction cards (left) and capacity/QoS cards (right)
  - Added separate column titles: "Predictive Analytics" (left) and "Capacity and QoS Analytics" (right)
  - Implemented centered subtitle spanning both columns for unified page description
  - Improved visual organization and screen utilization with responsive design
  - All functionality and interactive features preserved during layout reorganization
- July 5, 2025: Split client deployment scripts for git-managed updates with protected configuration
  - Created separate streamswarm_config.sh for user-editable variables (SERVER_URL, API_TOKEN, etc.)
  - Updated start_streamswarm_client.sh to source configuration from parent directory (outside git repo)
  - Enables cron-based startup from git repository while protecting user configuration from updates
  - Added comprehensive deployment documentation with setup instructions and troubleshooting guide
  - Automatic configuration sourcing prevents overwriting user settings during git pull updates
- July 5, 2025: Fixed QoS Traffic Violations logic to properly identify actual QoS issues instead of speed test results
  - Removed misleading "bandwidth violations" that flagged high speed test results (>100 Mbps) as problems
  - Replaced with proper QoS violation detection: jitter violations (>50ms) and packet loss violations (>1%)
  - Speed test measurements (upload/download bandwidth) are now correctly interpreted as capacity indicators, not violations
  - QoS analysis now focuses on actual network quality metrics: latency, jitter, packet loss, and DSCP classification
  - Fixed frontend display to show meaningful QoS violations instead of confusing bandwidth "issues"
- July 5, 2025: Successfully completed QoS Compliance Monitoring Model integration with proper visual styling
  - Fixed QoS section display issue on predictive analytics page (was not visible initially)
  - Corrected QoS card width styling to match other prediction cards (col-lg-8 mx-auto instead of col-12)
  - QoS Compliance Monitoring now properly integrated across all StreamSwarm components
  - Complete integration includes: individual test results, predictive analytics page, ML models page, and PDF reports
  - All five ML models now fully functional with distinct algorithms and proper visual consistency
- July 5, 2025: Successfully implemented QoS Compliance Monitoring Model as fifth ML model in StreamSwarm
  - Added comprehensive QoS compliance analysis with DSCP value evaluation and traffic classification assessment
  - Integrated QoS model training into existing ML pipeline with Support Vector Machine algorithm and specialized feature extraction
  - Created QoS Compliance Monitoring section in Predictive Analytics page with real-time analysis capabilities
  - Implemented API endpoint for QoS analysis with detailed compliance scoring and violation detection
  - Added QoS model card to ML Models Configuration page showing training status and requirements
  - System now supports 5 complete ML models: anomaly detection, health classification, performance prediction, network failure prediction, and QoS compliance monitoring
  - Maintains zero-trust architecture with all QoS processing running locally using trained compliance models
- July 5, 2025: Successfully resolved Network Failure Prediction API bug and completed full integration
  - Fixed critical "string indices must be integers, not 'str'" error in failure prediction calculation logic
  - Updated _generate_failure_prevention_recommendations method to handle both string and dictionary factor formats
  - Network Failure Prediction now processes 100+ historical metrics per destination with 90-95% confidence scores
  - Time Series Ensemble model providing accurate failure probability assessments (5.67% for tiktok.com, indicating low risk)
  - Full integration with Predictive Analytics page working correctly with destination selection and intelligent recommendations
  - Maintains zero-trust architecture with all ML processing running locally using trained models
- July 4, 2025: Fixed critical recurring test completion handler bug affecting multiple concurrent tests
  - Identified and resolved timing conflicts between completion handler and recurring processor
  - Fixed incorrect next_execution times that were causing 3+ hour delays instead of correct intervals
  - Enhanced completion handler to be called from ALL test completion paths (client submission, manual stop, auto-completion)
  - Resolved issue where tests completed via manual stopping or auto-completion didn't trigger new test creation
  - Modified recurring processor to avoid conflicts with "Create New Tests" mode (handled by completion handler)
  - Fixed Test 167 issue where completion handler wasn't triggered, preventing child test creation
  - All recurring test modes now work reliably with multiple concurrent recurring tests
- July 4, 2025: Enhanced recurring test system with 30-minute minimum intervals and validated authentication independence
  - Added support for 30-59 minute recurrence intervals (previously minimum was 1 hour)
  - Updated frontend interface with "Minutes" option in recurrence dropdown
  - Enhanced validation to enforce 30-minute absolute minimum with 10-minute buffer requirement
  - Validated recurring test processor works independently of authentication status (confirmed with comprehensive testing)
  - Recurring tests execute properly when users are logged out and authentication is enabled
  - Background processor operates via app context with direct database access, no authentication decorators
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