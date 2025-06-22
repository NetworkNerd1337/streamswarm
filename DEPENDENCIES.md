# StreamSwarm Dependencies

This document lists all required dependencies for StreamSwarm's comprehensive distributed network monitoring system with 65+ performance metrics.

## Python Package Requirements

Copy and paste the following commands to install all required dependencies:

```bash
# Core application dependencies
pip install flask>=2.3.0
pip install flask-sqlalchemy>=3.0.0
pip install sqlalchemy>=2.0.0
pip install psycopg2-binary>=2.9.0
pip install psutil>=5.9.0
pip install requests>=2.28.0
pip install gunicorn>=21.0.0
pip install werkzeug>=2.3.0
pip install email-validator>=2.0.0

# Advanced network analysis and QoS monitoring
pip install scapy>=2.5.0

# Bandwidth testing capabilities
pip install speedtest-cli>=2.1.3

# Professional report generation and visualization
pip install reportlab>=4.4.0
pip install matplotlib>=3.10.0
```

## Single Command Installation

Install all dependencies at once:

```bash
pip install flask>=2.3.0 flask-sqlalchemy>=3.0.0 sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0 psutil>=5.9.0 requests>=2.28.0 gunicorn>=21.0.0 werkzeug>=2.3.0 email-validator>=2.0.0 scapy>=2.5.0 speedtest-cli>=2.1.3 reportlab>=4.4.0 matplotlib>=3.10.0
```

## Package Descriptions

- **flask**: Web framework for the server component
- **flask-sqlalchemy**: Database ORM integration
- **sqlalchemy**: Database abstraction layer
- **psycopg2-binary**: PostgreSQL database adapter
- **psutil**: System resource monitoring
- **requests**: HTTP client for client-server communication
- **gunicorn**: Production WSGI server
- **werkzeug**: WSGI utilities
- **email-validator**: Email validation utilities
- **scapy**: Advanced packet analysis for QoS monitoring, DSCP/CoS detection, and traffic classification
- **speedtest-cli**: Internet bandwidth testing and speed measurement with multi-method approach
- **reportlab**: Professional PDF document generation with executive-level reporting
- **matplotlib**: Chart and graph generation for performance trend analysis

## Optional: Create requirements.txt

If you prefer using a requirements.txt file, create one with this content:

```
flask>=2.3.0
flask-sqlalchemy>=3.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
psutil>=5.9.0
requests>=2.28.0
gunicorn>=21.0.0
werkzeug>=2.3.0
email-validator>=2.0.0
scapy>=2.5.0
speedtest-cli>=2.1.3
reportlab>=4.4.0
matplotlib>=3.10.0
```

Then install with:
```bash
pip install -r requirements.txt
```