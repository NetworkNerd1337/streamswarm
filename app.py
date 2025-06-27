import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
database_url = os.environ.get("DATABASE_URL", "sqlite:///streamswarm.db")
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models to ensure tables are created
    import models
    # Import routes to register them with the app
    import routes
    db.create_all()

# Custom Jinja filters
@app.template_filter('tojsonfilter')
def tojson_filter(value):
    """Convert JSON string to Python object"""
    if isinstance(value, str):
        try:
            import json
            return json.loads(value)
        except json.JSONDecodeError:
            # For malformed data, extract basic info with regex
            try:
                import re
                # Create a basic dict with extracted values for display
                result = {}
                
                # Extract primary interface
                if match := re.search(r'primary_interface:\s*([a-zA-Z0-9]+)', value):
                    result['primary_interface'] = match.group(1)
                
                # Extract interface type
                if match := re.search(r'interface_type:\s*([a-zA-Z]+)', value):
                    result['interface_type'] = match.group(1)
                
                # Extract wireless flag
                if 'is_wireless: true' in value:
                    result['is_wireless'] = True
                elif 'is_wireless: false' in value:
                    result['is_wireless'] = False
                
                # Extract wireless info
                if result.get('is_wireless'):
                    wireless_info = {}
                    if match := re.search(r'ssid:\s*([^,}]+)', value):
                        wireless_info['ssid'] = match.group(1).strip()
                    if match := re.search(r'signal_strength:\s*(-?\d+)', value):
                        wireless_info['signal_strength'] = int(match.group(1))
                    if match := re.search(r'frequency:\s*([^,}]+)', value):
                        wireless_info['frequency'] = match.group(1).strip()
                    if match := re.search(r'channel:\s*(\d+)', value):
                        wireless_info['channel'] = int(match.group(1))
                    if match := re.search(r'mac_address:\s*([^,}]+)', value):
                        wireless_info['mac_address'] = match.group(1).strip()
                    if match := re.search(r'txpower:\s*([^,}]+)', value):
                        wireless_info['txpower'] = match.group(1).strip()
                    
                    if wireless_info:
                        result['wireless_info'] = wireless_info
                
                return result if result else {}
            except Exception:
                return {}
    return value or {}

@app.template_filter('from_json')
def from_json_filter(value):
    """Convert JSON string to Python object"""
    if isinstance(value, str):
        try:
            import json
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    return value if value else []

@app.template_filter('parse_signal_data')
def parse_signal_data_filter(value):
    """Parse comma-delimited signal strength data and return statistics"""
    if not value or value == 'null':
        return {'min': None, 'max': None, 'avg': None, 'count': 0, 'values': []}
    
    try:
        # Parse comma-delimited values
        values = [float(x.strip()) for x in value.split(',') if x.strip()]
        if not values:
            return {'min': None, 'max': None, 'avg': None, 'count': 0, 'values': []}
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values),
            'values': values
        }
    except:
        return {'min': None, 'max': None, 'avg': None, 'count': 0, 'values': []}

# Import routes
import routes
