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