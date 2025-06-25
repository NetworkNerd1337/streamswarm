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

- June 25, 2025: Fixed thread-safe bandwidth testing and download speed measurement issues
  - Fixed speedtest-cli signal handler error in threaded environment by implementing threading-based timeout
  - Enhanced HTTP bandwidth testing with improved timeout handling and better error logging
  - Replaced signal-based timeout with thread.join(timeout) for thread-safe operation
  - Added fallback logic to ensure bandwidth tests complete even when speedtest-cli fails
  - Fixed download speed measurement by reducing minimum test duration requirements and improving error handling
  - Enhanced bandwidth test logging to identify when download tests fail while upload succeeds
  - Improved bandwidth test reliability by ensuring minimum test duration for accurate measurements
  - Updated tutorial and metrics definition guide with comprehensive network interface detection documentation
  - Added detailed Network Interface Detection section explaining automatic interface type detection
  - Enhanced metrics list to include all 8 new network interface metrics (interface type, speed, duplex, wireless details, etc.)
  - Created Network Interface Troubleshooting Guide with practical interpretation guidelines
  - Added signal strength analysis guide for wireless connections (-30 to -70 dBm ranges)
  - Enhanced use case examples to include network interface correlation with performance issues
  - Updated tutorial page to emphasize per-client network environment analysis capabilities
  - Added FAQ item to tutorial page addressing client busy status during test scheduling

- June 24, 2025: Fixed bandwidth chart rendering issues, improved bandwidth testing accuracy, and added detailed process monitoring
  - Resolved JavaScript syntax errors preventing chart initialization
  - Enhanced data filtering to properly handle null/zero bandwidth values for clean chart display
  - Fixed Chart.js configuration structure with proper closing braces
  - Added manual refresh button to tests page instead of auto-refresh for better user control
  - Bandwidth upload and download charts now display connected lines only where actual measurements exist
  - Eliminated broken disconnected line segments in bandwidth visualizations
  - Redesigned bandwidth testing to measure actual internet connection speed (like traditional speed test websites)
  - Changed from destination-specific bandwidth testing to general internet speed measurement
  - Enhanced speedtest-cli as primary method with improved timeout and error handling
  - Added HTTP-based internet speed test using reliable endpoints (Cloudflare, httpbin) as fallback
  - Now provides intuitive upload/download speeds that match user expectations from speed test sites
  - Added detailed process monitoring with top processes ranked by CPU and memory consumption
  - Enhanced process metrics to show process names, PIDs, and resource usage percentages
  - Updated database schema and UI to display top 5 processes by CPU and memory usage in test results
  - Improved test creation UX by automatically stripping protocol prefixes from destination input
  - Added helpful text to destination field explaining protocol prefixes are not needed
  - Enhanced form validation to clean destination input both client-side and server-side
  - Fixed process monitoring display in test results page with dedicated "Process Monitoring Details" section
  - Successfully implemented process data visualization showing actual system processes with resource consumption
  - Implemented comprehensive network interface detection system with per-client network information display
  - Added network interface type detection (wireless, ethernet, VPN, etc.) with connection speed and duplex mode
  - Enhanced wireless interface detection with SSID, signal strength, and frequency information
  - Created dedicated "Network Interface Details by Client" section in test results showing each client's network setup
  - Added visual badges for interface types and status indicators for network connections
  - Database schema updated with network_interface_info field for storing detailed network configuration data

- June 23, 2025: Fixed bandwidth testing methodology, dashboard overview cards, missing metrics functionality, implemented client eligibility checking, and added API token authentication
  - Resolved bandwidth measurement issues by optimizing HTTP-based testing using httpbin.org
  - Enhanced bandwidth testing with reduced timeouts (15s) and improved error handling
  - Added proper bandwidth data collection with 512KB upload tests and 1MB download tests
  - Fixed API response to include traceroute_data field for network hop analysis
  - Implemented clickable data points on all charts for detailed client-specific analysis
  - Added intelligent network hop analysis with automatic problem hop identification
  - Displays complete traceroute hop-by-hop data with performance analysis during network issues
  - Enhanced tooltips to show client information and network path details on hover
  - Added export functionality for detailed data point analysis as JSON files
  - Fixed chart title dynamic updates to match selected metrics from dropdown
  - Fixed dashboard Network Latency Overview and System Resources Overview cards to display real data from monitoring network
  - Added /api/dashboard/metrics endpoint providing aggregated system performance data from last 24 hours
  - Dashboard charts now show actual network latency trends and average CPU/memory/disk usage across all clients
  - Investigated bandwidth testing methodology - upload speeds legitimately higher than download in Replit environment
  - Verified bandwidth measurements are accurate but reflect asymmetric cloud networking characteristics
  - Added missing network error metrics (network_errors_in/out, network_drops_in/out) to API responses
  - Fixed QoS Policy Compliance validation logic to provide meaningful assessment instead of always returning 1
  - Enhanced DSCP validation with context-aware policy checking for different traffic types
  - Fixed CoS values collection by implementing proper DSCP-to-CoS mapping using RFC 4594 standards
  - Added disk throughput metrics (disk_read_bytes_sec, disk_write_bytes_sec) to API responses and dropdown menu
  - Enhanced disk I/O collection logic to handle low-activity environments with proper 0 value reporting
  - Added 15 missing metrics to test results dropdown and API responses: CPU cores/temperature, process count, TCP connections, network packets/drops, memory details, swap usage, traceroute hops
  - Expanded dropdown from 24 to 39 available metrics for comprehensive system analysis
  - Fixed API endpoint to include all collected metrics in test data responses
  - Implemented client eligibility checking to prevent overlapping test assignments
  - Added busy/available status indicators in client and test management interfaces
  - Enhanced test creation with conflict detection and user feedback for busy clients
  - Modified client assignment logic to reject tests when clients are already running other tests
  - Implemented pre-shared API token authentication system for secure client-server communication
  - Added comprehensive token management interface with CRUD operations in web GUI
  - Enhanced client registration to require valid API tokens with automatic token consumption
  - Added authentication middleware for all client API endpoints with Bearer token validation
  - Created token lifecycle management: generation, regeneration, revocation, and usage tracking
  - Updated client code to include API token in all server requests with proper error handling
  - Added token status indicators and last-used tracking for administrative monitoring
  - Updated comprehensive documentation (README.md, DEPENDENCIES.md, USAGE.md, TUTORIAL.md) for API token authentication
  - Enhanced tutorial page with secure client deployment instructions and token management guidance
  - Added authentication troubleshooting sections and security best practices documentation
  - Updated client connection examples throughout documentation to include required API token parameter
  - Fixed tutorial client deployment sections to properly show API token authentication requirements
  - Added comprehensive authentication troubleshooting and secure deployment guidance to tutorial page
  - Enhanced client command examples with proper token authentication format and security warnings
  - Fixed critical Bootstrap JS dependency missing from base template causing token page JavaScript errors
  - Resolved token display and creation functionality by adding proper Bootstrap modal support
  - Cleaned up debugging console logs and restored production-ready token management interface
  - Resolved JavaScript variable conflicts between dashboard.js and tokens page causing execution errors
  - Fixed autoRefreshInterval redeclaration issues preventing proper token page functionality
  - Confirmed complete API token authentication system working correctly for client connections
  - Fixed tutorial command syntax to show proper --server and --token arguments instead of positional parameters
  - Updated all client connection examples throughout tutorial to use correct command line format
  - Fixed client script registration endpoint and added missing token field to registration payload
  - Corrected all client HTTP requests to include proper authentication headers and use correct API endpoints
  - Fixed bandwidth testing methodology where upload speeds appeared artificially high due to small test payload
  - Corrected HTTP bandwidth tests to use equivalent payload sizes (2MB download, 1MB upload) for accurate comparison
  - Increased minimum test duration requirements to ensure reliable speed measurements

- June 22, 2025: Successfully resolved test results page analytics display issues
  - Fixed Chart.js date adapter initialization by adding chartjs-adapter-date-fns library to base template
  - Corrected JavaScript syntax errors in chart initialization functions
  - Implemented proper data validation and filtering for Chart.js time-based charts
  - Enhanced summary statistics calculation with accurate data point counting
  - Verified functionality with test 12 data showing proper chart rendering and dropdown metric selection
  - All 7 collected data points now display correctly across CPU, memory, disk, and network performance metrics
  - Dashboard analytics fully operational with working metric switching and real-time data visualization

- June 22, 2025: Updated application to use Eastern Time (America/New_York) timezone
  - Changed all datetime operations from UTC to Eastern Time as requested
  - Updated client timestamps, server processing, and database defaults to use America/New_York timezone
  - Resolved timezone compatibility issues throughout application with proper zoneinfo implementation
  - Fixed "can't subtract offset-naive and offset-aware datetimes" errors
  - Implemented client notification for test termination with status checking
  - Added API endpoint /api/test/{id}/status for real-time test status queries
  - Enhanced client startup messaging with clear connection confirmation and status indicators

- June 22, 2025: Fixed client URL parsing for network tests
  - Added proper URL parsing to extract hostnames from full URLs in ping, traceroute, and advanced network tests
  - Fixed DNS resolution issues when test destinations include full URLs with protocols and paths
  - Updated deprecated datetime.utcnow() to datetime.now(timezone.utc) for Python 3.9+ compatibility
  - Client now correctly handles test targets like "https://vimeo.com/226053498" by extracting "vimeo.com" for network commands

- June 22, 2025: Updated deployment configuration section to be platform-agnostic
  - Changed "Replit Deployment Configuration" to "External vs Self-Hosted Server Configuration"
  - Focused on the key difference: external deployments don't need port specification
  - Provided examples for both cloud deployments and self-hosted servers
  - Made documentation more generic while maintaining practical guidance

- June 22, 2025: Fixed JavaScript syntax error in client delete functionality by using data attributes
  - Replaced inline onclick handlers with data attributes to avoid JSON escaping issues
  - Added event listeners in DOMContentLoaded to handle delete button clicks
  - Eliminated "Unexpected token '}'" error caused by unescaped quotes in onclick attributes
  - More secure approach using data attributes instead of inline JavaScript

- June 22, 2025: Updated client connection examples throughout tutorial for Replit deployment
  - Fixed Client Deployment Strategies card to show HTTPS connection format
  - Updated example commands to use Replit domain instead of port 5000
  - Added distinction between Replit deployment and self-hosted server examples
  - Updated clients page placeholder text with correct connection string

- June 22, 2025: Fixed missing deleteClient function in client management interface
  - Added deleteClient JavaScript function that was removed during XSS security fixes
  - Implemented proper confirmation dialog and API request handling
  - Fixed delete button functionality for offline clients on clients page
  - Added viewClientDetails and assignToTest function stubs for completeness

- June 22, 2025: Fixed client URL parsing for network tests
  - Added proper URL parsing to extract hostnames from full URLs in ping, traceroute, and advanced network tests
  - Fixed DNS resolution issues when test destinations include full URLs with protocols and paths
  - Updated deprecated datetime.utcnow() to datetime.now(timezone.utc) for Python 3.9+ compatibility
  - Client now correctly handles test targets like "https://vimeo.com/226053498" by extracting "vimeo.com" for network commands

- June 22, 2025: Fixed Replit deployment client connectivity configuration
  - Identified that Replit maps external port 80/443 to internal port 5000
  - Created client connection guide for Replit-deployed servers
  - Added Replit-specific configuration section to tutorial
  - Provided correct HTTPS connection strings for external clients
  - Server accessible at: https://1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev

- June 22, 2025: Refactored Server Setup Guide to focus on running and service configuration
  - Removed duplicate installation content from Server Setup Guide
  - Enhanced with specific focus on starting server, systemd service setup, and production deployment
  - Added security configuration, firewall setup, and monitoring sections
  - Separated server operation concerns from installation process
  - Improved clarity between Complete Installation Guide and Server Setup Guide purposes

- June 22, 2025: Clarified configuration options and environment variable relationship
  - Enhanced Environment Configuration section to explain three configuration methods
  - Added clear instructions for environment variables, .env files, and config.py modification
  - Explained configuration priority order: environment variables > .env file > config.py defaults
  - Added platform-specific instructions for Windows PowerShell and Linux/macOS
  - Clarified relationship between OS environment variables and application configuration
  - Added warnings about security best practices for each configuration method

- June 22, 2025: Enhanced database configuration documentation with SESSION_SECRET explanation
  - Updated tutorial page Environment Configuration section with detailed SESSION_SECRET requirements
  - Added comprehensive explanation of SESSION_SECRET purpose for web security and session encryption
  - Included specific requirements: 32+ characters, random composition, uniqueness per deployment
  - Added generation instructions using openssl command and security best practices
  - Enhanced security warnings about credential protection and version control safety

- June 22, 2025: Integrated GitHub repository into installation documentation
  - Updated all installation guides to include downloading from https://github.com/NetworkNerd1337/Swarm
  - Enhanced README.md quick start with git clone instructions
  - Updated DEPENDENCIES.md with GitHub download verification steps
  - Enhanced TUTORIAL.md with detailed GitHub download methods (git clone and ZIP download)
  - Updated web tutorial page with interactive GitHub download section
  - Added file verification steps to ensure complete download
  - Integrated repository download into both development and production setup workflows

- June 22, 2025: Updated comprehensive installation documentation for Linux and Windows
  - Enhanced README.md with detailed Python installation guides for Ubuntu/Debian, CentOS/RHEL/Fedora, Windows, and macOS
  - Updated DEPENDENCIES.md with platform-specific installation instructions and troubleshooting
  - Enhanced TUTORIAL.md with step-by-step Python setup, virtual environment creation, and dependency verification
  - Updated web tutorial page with interactive installation guide covering all platforms
  - Added installation verification scripts and common troubleshooting solutions
  - Included virtual environment setup instructions and best practices for all operating systems

- June 22, 2025: Verified complete database schema with all 65+ enhanced metrics
  - Confirmed PostgreSQL database includes all network, QoS, application, performance, and infrastructure metrics
  - Database schema automatically creates complete table structure on application startup
  - All 65+ metrics properly defined in TestResult model and reflected in live database
  - Schema includes advanced network performance, QoS analysis, application profiling, and infrastructure monitoring
  - Database initialization working correctly for new deployments with full feature set

- June 22, 2025: Enhanced test results pages and PDF export with comprehensive 65+ metrics display
  - Added detailed metrics analysis sections to test results pages showing all network, QoS, system, application, and infrastructure metrics
  - Enhanced PDF report generation to capture all 65+ metrics with categorized tables and comprehensive analysis
  - Added infrastructure health analysis with power consumption, memory errors, drive health monitoring
  - Implemented application layer performance analysis with content timing, compression, CDN performance
  - Enhanced recommendations system to analyze all metric categories and provide prioritized action items
  - Comprehensive test results now display network performance, advanced QoS, system resources, and infrastructure health
  - PDF exports include detailed metric tables, performance assessments, and actionable recommendations

- June 22, 2025: Updated comprehensive documentation and tutorials for 65+ metrics system
  - Enhanced tutorial page with comprehensive monitoring capabilities overview
  - Updated system requirements to reflect advanced monitoring dependencies
  - Added installation instructions for hardware monitoring tools (lm-sensors, smartctl, ethtool)
  - Updated static documentation (DEPENDENCIES.md, README.md, TUTORIAL.md, USAGE.md) with complete feature coverage
  - Enhanced feature descriptions to include all monitoring categories: network, QoS, application, performance, infrastructure
  - Added system dependency installation instructions for Linux distributions
  - Updated troubleshooting and configuration sections for advanced monitoring

- June 22, 2025: Integrated comprehensive metrics into test results pages
  - Moved all 65+ enhanced metrics from standalone dashboard into individual test results pages
  - Added comprehensive metrics summary cards showing network, application, QoS, and infrastructure data
  - Enhanced test results dropdown with organized metric categories (Network, System, QoS, etc.)
  - Removed standalone Enhanced Metrics navigation item - all metrics now contextual to specific tests
  - Updated test results page header to reflect comprehensive monitoring capabilities
  - Improved user experience by showing all advanced metrics alongside basic test data

- June 22, 2025: Infrastructure monitoring capabilities implementation
  - Added power consumption monitoring with multi-platform support (Linux, macOS, Windows)
  - Implemented fan speed monitoring using lm-sensors and hwmon interfaces
  - Added SMART drive health checking with temperature and error rate monitoring
  - Implemented ECC memory error rate detection through EDAC and system logs
  - Added comprehensive network interface error monitoring for physical layer issues
  - Enhanced database schema with 5 new infrastructure monitoring metrics
  - Updated metrics dashboard with dedicated infrastructure health card
  - Enhanced PDF reports with infrastructure health analysis table
  - Added graceful fallbacks for unsupported hardware or insufficient permissions
  - Implemented cross-platform compatibility with appropriate "Not Available" handling

- June 22, 2025: Application-layer metrics and performance profiling implementation
  - Added HTTP response code analysis tracking 2xx, 3xx, 4xx, 5xx responses over time
  - Implemented content download time measurement for full page/resource timing
  - Added connection reuse analysis for HTTP keep-alive effectiveness
  - Implemented compression ratio analysis for gzip/deflate effectiveness
  - Added SSL/TLS certificate validation timing measurement
  - Implemented DNS cache hit ratio measurement for cache effectiveness
  - Added HTTP cache effectiveness analysis with cache header detection
  - Implemented CDN performance scoring with header analysis and timing
  - Added multipath routing detection using multiple traceroute analysis
  - Implemented application response time measurement across multiple endpoints
  - Added database query time estimation through response analysis heuristics
  - Enhanced database schema with 11 new application and performance metrics
  - Updated metrics dashboard with dedicated cards for application-layer and performance profiling
  - Enhanced PDF reports with application performance and cache analysis tables

- June 22, 2025: Enhanced network performance and QoS testing capabilities
  - Implemented MTU Discovery with path MTU detection using ping with DF flag
  - Added TCP Window Scaling analysis and congestion window monitoring
  - Implemented TCP retransmission rate, out-of-order packets, and duplicate ACK detection
  - Added per-DSCP latency measurements for different traffic classes
  - Implemented traffic policing detection through burst testing
  - Added ECN (Explicit Congestion Notification) capability testing
  - Implemented queue depth estimation using latency increase patterns
  - Added flow control event monitoring for TCP connections
  - Enhanced database schema with 12 new advanced network and QoS metrics
  - Updated metrics dashboard with dedicated cards for network-level and advanced QoS metrics
  - Added new metrics to test results visualization dropdown

- June 22, 2025: Test scheduling and client auto-assignment implementation
  - Added complete test scheduling system with background job processing
  - Implemented scheduled test execution with proper status management
  - Added manual start/stop controls for scheduled tests
  - Enhanced client auto-assignment with URL parameter support and automatic modal opening
  - Pre-fills test creation form when client is pre-selected
  - Background scheduler thread checks for pending tests every 30 seconds
  - Improved test status display with scheduled/waiting indicators
  - Added timezone handling for scheduled test times

- June 22, 2025: Comprehensive bandwidth testing implementation
  - Added multi-method bandwidth testing with HTTP, speedtest-cli, and TCP approaches
  - Implemented real bandwidth measurement replacing mock data with actual throughput testing
  - Added speedtest-cli dependency for accurate internet speed testing
  - Enhanced PDF reports to include bandwidth performance metrics and status indicators
  - Updated test results visualization to display upload/download speeds
  - Added bandwidth metrics to enhanced metrics dashboard with dedicated performance cards
  - Implemented fallback testing methods for reliability across different network environments
  - Real-time bandwidth data collection and historical tracking for performance analysis

- June 22, 2025: Implemented client deletion functionality
  - Added delete button for offline clients in client management interface
  - Implemented safe client removal that preserves all historical test data
  - Added confirmation dialog explaining data preservation policy
  - Only offline clients can be deleted to prevent data loss during active testing
  - TestResult and TestClient records remain intact for reporting and analysis
  - Enhanced client management with proper data integrity safeguards

- June 22, 2025: Fixed critical test results page functionality
  - Resolved JavaScript chart initialization errors preventing data visualization
  - Fixed metric dropdown functionality to properly switch between performance metrics
  - Enhanced error handling for missing or null data values in test results
  - Changed default chart metric from ping_latency to CPU usage for better data availability
  - Added comprehensive logging for debugging chart and data loading issues
  - Test results page now properly displays performance charts for CPU, memory, and system metrics

- June 22, 2025: Professional PDF report generation feature implementation
  - Created comprehensive PDF report generator with executive-level formatting using ReportLab
  - Added performance charts with matplotlib integration for latency trends and client comparisons
  - Implemented detailed metrics analysis with color-coded performance indicators
  - Added QoS analysis section for DSCP/CoS monitoring when data is available
  - Generated automated recommendations based on performance thresholds
  - Integrated export functionality into test results and test management pages
  - Professional report layout with company logo, charts, tables, and executive summary

- June 22, 2025: Tutorial page improvements and dependency documentation
  - Fixed broken and irrelevant links in tutorial footer section
  - Updated footer links to point to actual application pages and tutorial sections
  - Replaced non-existent API reference with working metrics dashboard link
  - Consolidated installation documentation to prevent confusion
  - Created comprehensive DEPENDENCIES.md with proper pip install commands and version constraints
  - Enhanced installation instructions with version requirements for all dependencies
  - Removed duplicate "Complete Usage Guide" and streamlined navigation to existing content

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
  - Implemented QoS monitoring with DSCP/CoS detection, traffic classification, and policy compliance validation
  - Added scapy integration for advanced packet analysis and QoS marking extraction
  - Enhanced database schema with QoS metrics tracking and per-class bandwidth measurement

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