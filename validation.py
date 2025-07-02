"""
Input Validation and Security Module for StreamSwarm
Provides comprehensive validation for all API endpoints to prevent injection attacks
"""

import re
import math
import bleach
import ipaddress
from marshmallow import Schema, fields, validate, ValidationError as MarshmallowValidationError, pre_load, validates_schema
from wtforms import validators
from flask import request
import json


class ValidationError(Exception):
    """Custom validation error"""
    pass


def sanitize_html(html_content):
    """Sanitize HTML content to prevent XSS attacks"""
    if not html_content:
        return html_content
    
    # Allow only safe HTML tags and attributes
    allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    allowed_attributes = {}
    
    return bleach.clean(html_content, tags=allowed_tags, attributes=allowed_attributes, strip=True)


def sanitize_string(text):
    """Sanitize string input to prevent injection attacks"""
    if not text:
        return text
    
    # Remove any null bytes
    text = text.replace('\x00', '')
    
    # Strip dangerous characters
    text = re.sub(r'[<>&"\'\\]', '', str(text))
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000]
    
    return text.strip()


def validate_hostname(hostname):
    """Validate hostname format"""
    if not hostname:
        raise ValidationError("Hostname is required")
    
    # Sanitize first
    hostname = sanitize_string(hostname)
    
    # Check length
    if len(hostname) > 253:
        raise ValidationError("Hostname too long")
    
    # Check format
    hostname_pattern = re.compile(
        r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    )
    
    if not hostname_pattern.match(hostname):
        raise ValidationError("Invalid hostname format")
    
    return hostname


def validate_ip_address(ip):
    """Validate IP address format"""
    if not ip:
        raise ValidationError("IP address is required")
    
    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        raise ValidationError("Invalid IP address format")


def validate_json_string(json_str):
    """Validate and sanitize JSON string"""
    if not json_str:
        return json_str
    
    try:
        # Parse to validate JSON structure
        parsed = json.loads(json_str)
        
        # Re-serialize to ensure clean JSON
        return json.dumps(parsed)
    except (json.JSONDecodeError, TypeError):
        raise ValidationError("Invalid JSON format")


def validate_positive_number(value, field_name="Value"):
    """Validate positive number"""
    try:
        num = float(value)
        if num < 0:
            raise ValidationError(f"{field_name} must be positive")
        return num
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid number")


def validate_integer_range(value, min_val=None, max_val=None, field_name="Value"):
    """Validate integer within range"""
    try:
        num = int(value)
        if min_val is not None and num < min_val:
            raise ValidationError(f"{field_name} must be at least {min_val}")
        if max_val is not None and num > max_val:
            raise ValidationError(f"{field_name} must be at most {max_val}")
        return num
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid integer")


# Marshmallow Schemas for API validation

class ClientRegistrationSchema(Schema):
    """Schema for client registration"""
    hostname = fields.Str(required=True, validate=validate.Length(min=1, max=253))
    ip_address = fields.Str(required=True)
    api_token = fields.Str(required=True, validate=validate.Length(min=1, max=64))
    system_info = fields.Str(required=False, allow_none=True)
    
    @pre_load
    def sanitize_data(self, data, **kwargs):
        """Sanitize input data"""
        if 'hostname' in data:
            data['hostname'] = validate_hostname(data['hostname'])
        if 'ip_address' in data:
            data['ip_address'] = validate_ip_address(data['ip_address'])
        if 'api_token' in data:
            data['api_token'] = sanitize_string(data['api_token'])
        if 'system_info' in data and data['system_info']:
            data['system_info'] = validate_json_string(data['system_info'])
        return data


class TestResultSchema(Schema):
    """Schema for test result submission"""
    test_id = fields.Int(required=True, validate=validate.Range(min=1))
    client_id = fields.Int(required=True, validate=validate.Range(min=1))
    
    # Network metrics
    ping_latency = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0))
    ping_packet_loss = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0, max=100))
    bandwidth_upload = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0))
    bandwidth_download = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0))
    
    # System metrics
    cpu_percent = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0, max=100))
    memory_percent = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0, max=100))
    memory_used = fields.Int(required=False, allow_none=True, validate=validate.Range(min=0))
    memory_total = fields.Int(required=False, allow_none=True, validate=validate.Range(min=0))
    
    # JSON fields
    traceroute_data = fields.Str(required=False, allow_none=True)
    network_interface_info = fields.Str(required=False, allow_none=True)
    top_processes_cpu = fields.Str(required=False, allow_none=True)
    top_processes_memory = fields.Str(required=False, allow_none=True)
    
    # Signal strength fields
    signal_strength_min = fields.Float(required=False, allow_none=True)
    signal_strength_max = fields.Float(required=False, allow_none=True)
    signal_strength_avg = fields.Float(required=False, allow_none=True)
    signal_strength_samples = fields.Int(required=False, allow_none=True, validate=validate.Range(min=0))
    signal_strength_data = fields.Str(required=False, allow_none=True)
    
    @pre_load
    def sanitize_data(self, data, **kwargs):
        """Sanitize input data"""
        # Validate JSON fields
        json_fields = ['traceroute_data', 'network_interface_info', 'top_processes_cpu', 
                      'top_processes_memory']
        
        for field in json_fields:
            if field in data and data[field]:
                data[field] = validate_json_string(data[field])
        
        # Validate signal strength data format
        if 'signal_strength_data' in data and data['signal_strength_data']:
            signal_data = data['signal_strength_data']
            # Should be comma-separated numbers
            try:
                values = [float(x.strip()) for x in signal_data.split(',') if x.strip()]
                data['signal_strength_data'] = ','.join(map(str, values))
            except (ValueError, TypeError):
                raise ValidationError("Invalid signal strength data format")
        
        return data


class TestCreationSchema(Schema):
    """Schema for test creation"""
    name = fields.Str(required=True, validate=validate.Length(min=1, max=255))
    description = fields.Str(required=False, allow_none=True, validate=validate.Length(max=1000))
    destination = fields.Str(required=True, validate=validate.Length(min=1, max=255))
    duration = fields.Int(required=False, validate=validate.Range(min=1, max=86400))  # Max 24 hours
    interval = fields.Int(required=False, validate=validate.Range(min=1, max=3600))   # Max 1 hour
    is_recurring = fields.Bool(required=False, load_default=False)
    recurrence_interval = fields.Int(required=False, allow_none=True, validate=validate.Range(min=600))  # Min 10 minutes
    scheduled_time = fields.Str(required=False, allow_none=True)
    client_ids = fields.List(fields.Int(validate=validate.Range(min=1)), required=True)
    
    @pre_load
    def sanitize_data(self, data, **kwargs):
        """Sanitize input data"""
        if 'name' in data:
            data['name'] = sanitize_string(data['name'])
        if 'description' in data and data['description']:
            data['description'] = sanitize_html(data['description'])
        if 'destination' in data:
            # Validate destination as hostname or IP
            dest = sanitize_string(data['destination'])
            try:
                # Try as IP first
                validate_ip_address(dest)
            except ValidationError:
                # Try as hostname
                try:
                    validate_hostname(dest)
                except ValidationError:
                    raise ValidationError("Destination must be valid IP address or hostname")
            data['destination'] = dest
        return data
    
    @validates_schema
    def validate_recurrence(self, data, **kwargs):
        """Validate recurrence settings"""
        if data.get('is_recurring', False):
            duration = data.get('duration', 300)
            recurrence_interval = data.get('recurrence_interval')
            
            if not recurrence_interval:
                raise MarshmallowValidationError("Recurrence interval is required when recurring is enabled")
            
            # Minimum recurrence interval: test duration + 10 minute buffer
            min_interval = duration + (10 * 60)  # 10 minutes buffer
            if recurrence_interval < min_interval:
                min_minutes = math.ceil(min_interval / 60)
                raise MarshmallowValidationError(f"Recurrence interval must be at least {min_minutes} minutes (test duration + 10 minute buffer)")
        
        return data


class ApiTokenSchema(Schema):
    """Schema for API token creation"""
    name = fields.Str(required=True, validate=validate.Length(min=1, max=255))
    description = fields.Str(required=False, allow_none=True, validate=validate.Length(max=1000))
    
    @pre_load
    def sanitize_data(self, data, **kwargs):
        """Sanitize input data"""
        if 'name' in data:
            data['name'] = sanitize_string(data['name'])
        if 'description' in data and data['description']:
            data['description'] = sanitize_html(data['description'])
        return data


def validate_request_data(schema_class, data=None):
    """Validate request data using marshmallow schema"""
    if data is None:
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()
    
    schema = schema_class()
    try:
        validated_data = schema.load(data)
        return validated_data, None
    except MarshmallowValidationError as e:
        return None, e.messages


def validate_query_params(**validations):
    """Validate query parameters"""
    errors = {}
    validated = {}
    
    for param, validator_func in validations.items():
        value = request.args.get(param)
        if value is not None:
            try:
                validated[param] = validator_func(value)
            except ValidationError as e:
                errors[param] = str(e)
        else:
            validated[param] = None
    
    return validated, errors if errors else None


def sanitize_filename(filename):
    """Sanitize filename to prevent path traversal"""
    if not filename:
        return filename
    
    # Remove directory separators and other dangerous characters
    filename = re.sub(r'[/\\:*?"<>|]', '', filename)
    
    # Remove leading dots and spaces
    filename = filename.lstrip('. ')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename


def validate_pagination_params():
    """Validate common pagination parameters"""
    page = request.args.get('page', '1')
    per_page = request.args.get('per_page', '50')
    
    try:
        page = validate_integer_range(page, min_val=1, field_name="Page")
        per_page = validate_integer_range(per_page, min_val=1, max_val=1000, field_name="Per page")
        return page, per_page, None
    except ValidationError as e:
        return 1, 50, str(e)


# Global rate limiting storage
_rate_limit_storage = {}

def rate_limit_check(client_ip, endpoint, max_requests=100, time_window=3600):
    """Basic rate limiting check (in-memory for now)"""
    # This is a simple in-memory rate limiter
    # In production, use Redis or database-backed solution
    import time
    
    global _rate_limit_storage
    
    current_time = time.time()
    key = f"{client_ip}:{endpoint}"
    
    if key not in _rate_limit_storage:
        _rate_limit_storage[key] = []
    
    # Clean old requests
    _rate_limit_storage[key] = [
        req_time for req_time in _rate_limit_storage[key]
        if current_time - req_time < time_window
    ]
    
    # Check limit
    if len(_rate_limit_storage[key]) >= max_requests:
        return False
    
    # Add current request
    _rate_limit_storage[key].append(current_time)
    return True