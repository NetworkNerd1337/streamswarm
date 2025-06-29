import os
import logging
import json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
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
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}

@app.template_filter('from_json')
def from_json_filter(value):
    """Convert JSON string to Python object"""
    if isinstance(value, str):
        try:
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

@app.template_filter('average')
def average_filter(values):
    """Calculate the average of a list of values"""
    if not values:
        return None
    try:
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None
        return sum(valid_values) / len(valid_values)
    except (TypeError, ValueError):
        return None

# Import routes
import routes
