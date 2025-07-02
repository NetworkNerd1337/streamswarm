import os
import logging
import json
from flask import Flask, redirect, url_for
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

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

# Session timeout and activity tracking
@app.before_request
def check_session_timeout():
    """Check for session timeout and track user activity"""
    from models import SystemConfig
    
    # Skip timeout check for static files, login, and logout routes
    if (request.endpoint in ['static', 'login', 'logout'] or 
        request.path.startswith('/static/') or
        not current_user.is_authenticated):
        return
    
    # Get session timeout setting
    timeout_minutes = SystemConfig.get_session_timeout_minutes()
    
    # Get current time for session tracking
    now = datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None)
    current_time_iso = now.isoformat()
    
    # Initialize last_activity for new sessions
    if 'last_activity' not in session:
        session['last_activity'] = current_time_iso
    
    # Check if user has been idle too long (only if timeout is enabled)
    if timeout_minutes > 0:
        last_activity = session.get('last_activity')
        if last_activity:
            try:
                last_activity_time = datetime.fromisoformat(last_activity)
                
                # Check if session has timed out
                if now - last_activity_time > timedelta(minutes=timeout_minutes):
                    logging.info(f"Session timed out for user {current_user.id if hasattr(current_user, 'id') else 'unknown'} after {timeout_minutes} minutes")
                    logout_user()
                    session.clear()
                    return redirect(url_for('login', timeout='1'))
            except (ValueError, TypeError) as e:
                # Invalid date format, reset last_activity
                logging.warning(f"Invalid last_activity format, resetting: {e}")
                session['last_activity'] = current_time_iso
    
    # Update last activity for navigation requests (not AJAX/auto-refresh)
    # Only track activity for main pages to avoid auto-refresh interference
    main_pages = [
        'dashboard', 'tutorial', 'clients', 'tests', 'profile', 
        'system_configuration', 'user_management', 'tokens',
        'ml_models', 'test_results', 'create_test', 'edit_test'
    ]
    
    # Update activity timestamp for legitimate user interactions
    should_update_activity = (
        request.endpoint in main_pages and 
        request.method == 'GET' and 
        'XMLHttpRequest' not in request.headers.get('X-Requested-With', '') and
        not request.path.startswith('/api/')
    )
    
    if should_update_activity:
        session['last_activity'] = current_time_iso
        session.permanent = True  # Ensure session persists properly

# Development mode bypass decorator
from functools import wraps
from flask import session, request, g
from flask_login import current_user, logout_user
from datetime import datetime, timedelta
import zoneinfo

def login_required_with_dev_bypass(f):
    """Custom login_required decorator that respects development mode"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from models import SystemConfig
        
        # Check if development mode is enabled
        if SystemConfig.is_development_mode():
            # In development mode, bypass authentication
            # Set a mock user for templates that might need current_user
            g.dev_mode_active = True
            return f(*args, **kwargs)
        
        # Normal authentication check
        from flask_login import login_required
        return login_required(f)(*args, **kwargs)
    
    return decorated_function

# Helper function to check if user is admin or dev mode is active
def require_admin_or_dev_mode():
    """Check if current user is admin or development mode is active"""
    from models import SystemConfig
    
    if SystemConfig.is_development_mode():
        return True
    
    if current_user.is_authenticated and current_user.is_admin():
        return True
    
    return False

with app.app_context():
    # Import models to ensure tables are created
    import models
    # Import routes to register them with the app
    import routes
    db.create_all()
    
    # Create default admin user if no users exist
    if models.User.query.count() == 0:
        admin_user = models.User(
            username='admin',
            email='admin@streamswarm.local',
            role='admin'
        )
        admin_user.set_password('admin123')
        db.session.add(admin_user)
        db.session.commit()
        logging.info("Created default admin user: username=admin, password=admin123")
    
    # Start recurring test processor
    try:
        from recurring_test_processor import start_recurring_processor
        start_recurring_processor()
        logging.info("Recurring test processor started successfully")
    except Exception as e:
        logging.error(f"Failed to start recurring test processor: {e}")

# Custom Jinja filters
@app.template_filter('extract_numeric')
def extract_numeric_filter(items, attribute_name):
    """Safely extract numeric values from a list of objects"""
    result = []
    for item in items:
        if hasattr(item, attribute_name):
            value = getattr(item, attribute_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result.append(value)
    return result

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
            return {}
    return value if value else {}

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

# Error handlers for graceful error pages
from flask import render_template

@app.errorhandler(400)
def bad_request(error):
    return render_template('error.html', 
                         error_code=400,
                         error_title="Bad Request",
                         error_message="The request could not be understood due to malformed syntax."), 400

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         error_code=404,
                         error_title="Page Not Found", 
                         error_message="The requested page could not be found."), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', 
                         error_code=500,
                         error_title="Internal Server Error",
                         error_message="An unexpected error occurred. Please try again later."), 500
