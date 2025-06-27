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
            result = json.loads(value)
            print(f"Debug: JSON parsed successfully: {type(result)}")
            return result
        except json.JSONDecodeError:
            # Try to handle Python dict notation (malformed JSON)
            try:
                import ast
                # First try to fix common JSON issues
                fixed_value = value.replace('null', 'None').replace('true', 'True').replace('false', 'False')
                result = ast.literal_eval(fixed_value)
                print(f"Debug: AST parsed successfully: {type(result)}")
                return result
            except (ValueError, SyntaxError) as e:
                print(f"Debug: JSON parse failed: {str(e)[:100]}")
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
