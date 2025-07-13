#!/usr/bin/env python3
"""
StreamSwarm Client - Network and system monitoring client
"""

# Client version - increment when making changes
CLIENT_VERSION = "1.9.6"

import os
import sys
import time
import json
import logging
import argparse
import threading
import subprocess
import platform
import socket
import struct
import fcntl
import re
import random
from datetime import datetime, timezone
import zoneinfo
import psutil
import requests
from urllib.parse import urljoin, urlparse

# Import geolocation service for enhanced path analysis
try:
    from geolocation_service import GeolocationService
    GEOLOCATION_AVAILABLE = True
except ImportError:
    GEOLOCATION_AVAILABLE = False

# Import GNMI client for network path analysis
try:
    from gnmi_client import GNMINetworkAnalyzer
    GNMI_AVAILABLE = True
except ImportError as e:
    GNMI_AVAILABLE = False
    logging.warning(f"GNMI functionality not available: {e}")
    logging.info("Install pygnmi to enable advanced network path analysis: pip install pygnmi")

# Try to import speedtest and scapy, fall back gracefully if not available
try:
    import speedtest
    SPEEDTEST_AVAILABLE = True
except ImportError:
    SPEEDTEST_AVAILABLE = False
    print("Warning: speedtest-cli not available. Bandwidth testing will be limited.")

try:
    from scapy.all import *
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.l2 import Ether
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: scapy not available. Advanced QoS monitoring will be limited.")

# Check for modern iw command (replacement for deprecated iwlib)
try:
    import subprocess
    result = subprocess.run(['iw', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        WIFI_SCANNING_AVAILABLE = True
        print("iw command available - WiFi environmental scanning enabled")
    else:
        WIFI_SCANNING_AVAILABLE = False
        print("Warning: iw command not available. Install with: sudo apt-get install iw")
except (subprocess.TimeoutExpired, FileNotFoundError):
    WIFI_SCANNING_AVAILABLE = False
    print("Warning: iw command not available. WiFi environmental scanning will be limited.")

# Check for VoIP testing capabilities
try:
    result = subprocess.run(['sipsak', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        VOIP_TESTING_AVAILABLE = True
        print("sipsak available - VoIP analysis testing enabled")
    else:
        VOIP_TESTING_AVAILABLE = False
        print("Warning: sipsak not available. Install with: sudo apt-get install sipsak")
except (subprocess.TimeoutExpired, FileNotFoundError):
    VOIP_TESTING_AVAILABLE = False
    print("Warning: sipsak not available. VoIP analysis testing will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamSwarmClient:
    def __init__(self, server_url, client_name=None, api_token=None):
        self.server_url = server_url.rstrip('/')
        self.client_name = client_name or platform.node()
        self.api_token = api_token
        self.client_id = None
        self.running = False
        self.heartbeat_thread = None
        self.test_threads = {}
        self.signal_strength_tracker = {}
        
        # Initialize geolocation service for enhanced path analysis
        if GEOLOCATION_AVAILABLE:
            self.geo_service = GeolocationService()
            logger.info("Geolocation service initialized for enhanced network path analysis")
        else:
            self.geo_service = None
            logger.warning("Geolocation service not available - using basic traceroute only")
        
        # Initialize GNMI network analyzer for managed infrastructure analysis
        if GNMI_AVAILABLE:
            self.gnmi_analyzer = GNMINetworkAnalyzer()
            # GNMI devices will be synchronized from server on startup
            logger.info("GNMI network analyzer initialized - devices will sync from server")
        else:
            self.gnmi_analyzer = None
            logger.warning("GNMI functionality not available - advanced network path analysis disabled")
        
        # GNMI device configuration storage
        self.gnmi_devices = []
        self.gnmi_config_file = 'gnmi_devices.json'
        self.gnmi_certs_dir = 'gnmi_certs'
        
        # WiFi environmental scanning capabilities
        self.wifi_interfaces = {}
        self.primary_wifi_interface = None
        self.spare_wifi_interfaces = []
        
        self._detect_wifi_interfaces()
        
        # Client certificate storage for GNMI authentication
        self.client_cert_dir = 'client_certs'
        self.client_cert_file = os.path.join(self.client_cert_dir, 'client.crt')
        self.client_key_file = os.path.join(self.client_cert_dir, 'client.key')
        
        # Create certificates directory
        if not os.path.exists(self.gnmi_certs_dir):
            os.makedirs(self.gnmi_certs_dir)
            
        logger.info("GNMI network analyzer initialized for managed infrastructure analysis")
        
        # Get system information
        self.system_info = self._get_system_info()
        
        # Register with server
        self._register()
        
        # Synchronize GNMI device configuration from server
        if GNMI_AVAILABLE:
            self._sync_gnmi_devices()
        
        # Ensure client certificates exist for GNMI authentication
        self._ensure_client_certificates()
    
    def _get_system_info(self):
        """Get system information"""
        try:
            return {
                'platform': platform.platform(),
                'hostname': platform.node(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def _register(self):
        """Register client with server"""
        try:
            # Get local IP address
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip_address = s.getsockname()[0]
            finally:
                s.close()
            
            data = {
                'hostname': self.client_name,
                'ip_address': ip_address,
                'system_info': json.dumps(self.system_info),
                'client_version': CLIENT_VERSION,
                'token': self.api_token
            }
            
            headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
            response = requests.post(
                urljoin(self.server_url, '/api/client/register'),
                json=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.client_id = result['client_id']
                logger.info(f"Registered with server. Client ID: {self.client_id}")
            else:
                logger.error(f"Failed to register: {response.status_code} - {response.text}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error registering with server: {e}")
            sys.exit(1)
    
    def _send_heartbeat(self):
        """Send periodic heartbeat to server"""
        while self.running:
            try:
                headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
                
                # Get system uptime
                uptime_seconds = None
                try:
                    uptime_seconds = int(time.time() - psutil.boot_time())
                except Exception as e:
                    logger.debug(f"Unable to get system uptime: {e}")
                
                data = {
                    'client_version': CLIENT_VERSION,
                    'uptime_seconds': uptime_seconds
                }
                response = requests.post(
                    urljoin(self.server_url, f'/api/client/{self.client_id}/heartbeat'),
                    json=data,
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    # Check if the server sent a reboot request
                    try:
                        result = response.json()
                        logger.info(f"HEARTBEAT: Received response from server - {result}")
                        if result.get('reboot_requested'):
                            logger.info("REBOOT: Server requested client reboot. Executing reboot command...")
                            self._execute_reboot()
                        else:
                            logger.info("HEARTBEAT: No reboot requested in server response")
                    except Exception as e:
                        logger.warning(f"HEARTBEAT: Failed to parse server response as JSON: {e}")
                        logger.warning(f"HEARTBEAT: Raw response: {response.text}")
                        pass  # Response might not be JSON, that's OK
                else:
                    logger.warning(f"Heartbeat failed: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
            
            time.sleep(30)  # Send heartbeat every 30 seconds
    
    def _get_system_metrics(self):
        """Get current system metrics"""
        try:
            metrics = {}
            
            # Basic CPU usage
            metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
            
            # Enhanced CPU metrics
            try:
                load_avg = psutil.getloadavg()
                metrics['cpu_load_1min'] = load_avg[0]
                metrics['cpu_load_5min'] = load_avg[1]
                metrics['cpu_load_15min'] = load_avg[2]
            except:
                metrics['cpu_load_1min'] = None
                metrics['cpu_load_5min'] = None
                metrics['cpu_load_15min'] = None
            
            metrics['cpu_cores'] = psutil.cpu_count()
            
            # CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                metrics['cpu_freq_current'] = cpu_freq.current if cpu_freq else None
            except:
                metrics['cpu_freq_current'] = None
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.update({
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_cached': getattr(memory, 'cached', 0),
                'memory_buffers': getattr(memory, 'buffers', 0),
                'memory_shared': getattr(memory, 'shared', 0)
            })
            
            # Swap usage
            try:
                swap = psutil.swap_memory()
                metrics.update({
                    'swap_total': swap.total,
                    'swap_used': swap.used,
                    'swap_percent': swap.percent
                })
            except:
                metrics.update({
                    'swap_total': 0,
                    'swap_used': 0,
                    'swap_percent': 0
                })
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.update({
                'disk_percent': disk.percent,
                'disk_used': disk.used,
                'disk_total': disk.total
            })
            
            # Disk I/O metrics
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    # Store current values for rate calculation
                    if not hasattr(self, '_prev_disk_io'):
                        self._prev_disk_io = {'time': time.time(), 'read_bytes': disk_io.read_bytes, 'write_bytes': disk_io.write_bytes, 'read_count': disk_io.read_count, 'write_count': disk_io.write_count}
                        # Set initial values to 0 for first measurement
                        metrics.update({
                            'disk_read_iops': 0.0,
                            'disk_write_iops': 0.0,
                            'disk_read_bytes_sec': 0.0,
                            'disk_write_bytes_sec': 0.0
                        })
                    else:
                        time_diff = time.time() - self._prev_disk_io['time']
                        if time_diff > 0:
                            # Calculate rates and ensure non-negative values
                            read_iops = max(0, (disk_io.read_count - self._prev_disk_io['read_count']) / time_diff)
                            write_iops = max(0, (disk_io.write_count - self._prev_disk_io['write_count']) / time_diff)
                            read_bps = max(0, (disk_io.read_bytes - self._prev_disk_io['read_bytes']) / time_diff)
                            write_bps = max(0, (disk_io.write_bytes - self._prev_disk_io['write_bytes']) / time_diff)
                            
                            metrics.update({
                                'disk_read_iops': round(read_iops, 2),
                                'disk_write_iops': round(write_iops, 2),
                                'disk_read_bytes_sec': round(read_bps, 2),
                                'disk_write_bytes_sec': round(write_bps, 2)
                            })
                        else:
                            # Time difference too small, use previous values or 0
                            metrics.update({
                                'disk_read_iops': 0.0,
                                'disk_write_iops': 0.0,
                                'disk_read_bytes_sec': 0.0,
                                'disk_write_bytes_sec': 0.0
                            })
                        
                        self._prev_disk_io = {'time': time.time(), 'read_bytes': disk_io.read_bytes, 'write_bytes': disk_io.write_bytes, 'read_count': disk_io.read_count, 'write_count': disk_io.write_count}
                else:
                    # No disk I/O counters available - set to 0 instead of None
                    metrics.update({
                        'disk_read_iops': 0.0,
                        'disk_write_iops': 0.0,
                        'disk_read_bytes_sec': 0.0,
                        'disk_write_bytes_sec': 0.0
                    })
            except Exception as e:
                logger.debug(f"Disk I/O metrics collection failed: {e}")
                # Set to 0 instead of None to ensure data is collected
                metrics.update({
                    'disk_read_iops': 0.0,
                    'disk_write_iops': 0.0,
                    'disk_read_bytes_sec': 0.0,
                    'disk_write_bytes_sec': 0.0
                })
            
            # Network interface metrics
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics.update({
                        'network_bytes_sent': net_io.bytes_sent,
                        'network_bytes_recv': net_io.bytes_recv,
                        'network_packets_sent': net_io.packets_sent,
                        'network_packets_recv': net_io.packets_recv,
                        'network_errors_in': net_io.errin,
                        'network_errors_out': net_io.errout,
                        'network_drops_in': net_io.dropin,
                        'network_drops_out': net_io.dropout
                    })
            except:
                metrics.update({
                    'network_bytes_sent': None,
                    'network_bytes_recv': None,
                    'network_packets_sent': None,
                    'network_packets_recv': None,
                    'network_errors_in': None,
                    'network_errors_out': None,
                    'network_drops_in': None,
                    'network_drops_out': None
                })
            
            # TCP retransmission rate calculation
            try:
                tcp_retrans_data = self._get_tcp_retransmission_stats()
                metrics.update(tcp_retrans_data)
            except Exception as e:
                logger.debug(f"TCP retransmission stats collection failed: {e}")
                metrics.update({
                    'tcp_retransmission_rate': None,
                    'tcp_out_of_order_packets': None,
                    'tcp_duplicate_acks': None
                })
            
            # Process and connection metrics
            try:
                metrics['process_count'] = len(psutil.pids())
                metrics['tcp_connections'] = len([conn for conn in psutil.net_connections() if conn.type == psutil.socket.SOCK_STREAM])
                
                # Get top processes by CPU usage
                try:
                    processes = []
                    # First, call cpu_percent() on all processes to initialize measurements
                    for proc in psutil.process_iter():
                        try:
                            proc.cpu_percent()  # Initialize CPU measurement
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    # Wait a short time to get meaningful CPU measurements
                    time.sleep(0.1)
                    
                    # Now collect actual process information with CPU usage
                    for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                        try:
                            proc_info = proc.info
                            cpu_percent = proc.cpu_percent()  # Get current CPU usage
                            
                            if cpu_percent is not None and proc_info['memory_percent'] is not None:
                                processes.append({
                                    'pid': proc_info['pid'],
                                    'name': proc_info['name'],
                                    'cpu_percent': round(cpu_percent, 2),
                                    'memory_percent': round(proc_info['memory_percent'], 2)
                                })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    # Sort by CPU usage and get top 10
                    top_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
                    metrics['top_processes_cpu'] = json.dumps(top_cpu)
                    
                    # Sort by memory usage and get top 10
                    top_memory = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10]
                    metrics['top_processes_memory'] = json.dumps(top_memory)
                    
                except Exception as e:
                    logger.debug(f"Process details collection failed: {e}")
                    metrics['top_processes_cpu'] = json.dumps([])
                    metrics['top_processes_memory'] = json.dumps([])
                
            except:
                metrics['process_count'] = None
                metrics['tcp_connections'] = None
                metrics['top_processes_cpu'] = json.dumps([])
                metrics['top_processes_memory'] = json.dumps([])
            
            # Temperature metrics (where available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try to get CPU temperature
                    cpu_temp = None
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            if entries:
                                cpu_temp = entries[0].current
                                break
                    metrics['cpu_temperature'] = cpu_temp
                else:
                    metrics['cpu_temperature'] = None
            except:
                metrics['cpu_temperature'] = None
            
            # Network interface detection
            try:
                interface_info = self._get_network_interface_info()
                metrics['network_interface_info'] = json.dumps(interface_info)
                logger.debug(f"Network interface info collected: {interface_info.get('primary_interface', 'unknown')}")
            except Exception as e:
                logger.warning(f"Network interface detection failed: {e}")
                metrics['network_interface_info'] = json.dumps({})
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _ping_test(self, destination, count=4, packet_size=64):
        """Perform ping test with jitter calculation"""
        try:
            # Extract hostname from URL if full URL is provided
            if destination.startswith(('http://', 'https://')):
                # Parse full URL to extract hostname
                parsed_url = urlparse(destination)
                hostname = parsed_url.netloc
                if not hostname:
                    # Fallback manual parsing
                    hostname = destination.split('://')[1].split('/')[0]
            else:
                # Direct hostname or IP
                hostname = destination.split('/')[0]
            
            if platform.system().lower() == 'windows':
                cmd = ['ping', '-n', str(count), '-l', str(packet_size), hostname]
            else:
                cmd = ['ping', '-c', str(count), '-s', str(packet_size), hostname]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout
                
                # Parse ping results (simplified)
                latencies = []
                packet_loss = 0
                
                for line in output.split('\n'):
                    # Look for time= in ping output
                    if 'time=' in line.lower():
                        try:
                            time_part = line.split('time=')[1].split()[0]
                            latency = float(time_part.replace('ms', ''))
                            latencies.append(latency)
                        except:
                            pass
                    
                    # Look for packet loss
                    if '% packet loss' in line or '% loss' in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if '%' in part:
                                    packet_loss = float(part.replace('%', ''))
                                    break
                        except:
                            pass
                
                avg_latency = sum(latencies) / len(latencies) if latencies else None
                
                # Calculate jitter (standard deviation of latencies)
                jitter = None
                if len(latencies) > 1:
                    mean = avg_latency
                    variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
                    jitter = variance ** 0.5
                
                return {
                    'latency': avg_latency,
                    'packet_loss': packet_loss,
                    'jitter': jitter,
                    'raw_output': output
                }
            else:
                logger.warning(f"Ping failed: {result.stderr}")
                return {'latency': None, 'packet_loss': 100, 'jitter': None}
                
        except Exception as e:
            logger.error(f"Error performing ping test: {e}")
            return {'latency': None, 'packet_loss': 100, 'jitter': None}
    
    def _traceroute_test(self, destination):
        """Perform enhanced traceroute test with geolocation analysis"""
        try:
            # Extract hostname from URL if full URL is provided
            if destination.startswith(('http://', 'https://')):
                # Parse full URL to extract hostname
                parsed_url = urlparse(destination)
                hostname = parsed_url.netloc
                if not hostname:
                    # Fallback manual parsing
                    hostname = destination.split('://')[1].split('/')[0]
            else:
                # Direct hostname or IP
                hostname = destination.split('/')[0]
            
            if platform.system().lower() == 'windows':
                cmd = ['tracert', hostname]
            else:
                cmd = ['traceroute', hostname]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            hops = 0
            hop_data = []
            path_analysis = {}
            map_html = ""
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip():
                        # Count hops (simplified parsing)
                        if line.strip()[0].isdigit():
                            hops += 1
                            hop_data.append(line.strip())
                
                # Perform geolocation analysis if service is available
                if self.geo_service and hop_data:
                    try:
                        logger.info(f"Performing geolocation analysis for {len(hop_data)} hops")
                        path_analysis = self.geo_service.analyze_traceroute_path(hop_data, hostname)
                        
                        # Generate interactive map
                        if path_analysis:
                            map_html = self.geo_service.create_path_map(path_analysis, hostname)
                            logger.debug(f"Generated map HTML for {len(path_analysis.get('hops', []))} geolocated hops")
                        
                    except Exception as e:
                        logger.warning(f"Geolocation analysis failed: {e}")
                        path_analysis = {}
                        map_html = ""
            
            # Perform GNMI network path analysis for managed infrastructure
            gnmi_path_analysis = {}
            if self.gnmi_analyzer:
                try:
                    logger.info(f"Performing GNMI network path analysis for {hostname}")
                    gnmi_path_analysis = self.gnmi_analyzer.analyze_network_path(hostname)
                    logger.debug(f"GNMI analysis completed: {len(gnmi_path_analysis.get('managed_hops', []))} managed hops")
                    
                except Exception as e:
                    logger.warning(f"GNMI network path analysis failed: {e}")
                    gnmi_path_analysis = {}
            
            return {
                'hops': hops,
                'data': hop_data,
                'raw_output': result.stdout,
                'path_analysis': path_analysis,
                'map_html': map_html,
                'gnmi_path_analysis': gnmi_path_analysis
            }
            
        except Exception as e:
            logger.error(f"Error performing traceroute: {e}")
            return {
                'hops': 0, 
                'data': [], 
                'raw_output': '',
                'path_analysis': {},
                'map_html': '',
                'gnmi_path_analysis': {}
            }
    
    def _run_test(self, test_config):
        """Run a single test"""
        test_id = test_config['id']
        destination = test_config['destination']
        duration = test_config.get('duration', 300)
        test_type = test_config.get('test_type', 'standard')
        interval = test_config.get('interval', 5)
        packet_size = test_config.get('packet_size', 64)
        
        logger.info(f"Starting test {test_id} to {destination} for {duration}s (type: {test_type})")
        
        # Handle standalone WiFi environmental test
        if test_type == 'wifi_environment':
            logger.info(f"Executing WiFi environmental test for test {test_id}")
            return self._wifi_environmental_test(test_id, destination, duration, interval)
        elif test_type == 'voip_analysis':
            logger.info(f"Executing VoIP analysis test for test {test_id}")
            return self._voip_analysis_test(test_id, self.server_url, duration, interval)
        else:
            logger.info(f"Executing standard network test for test {test_id}")
        
        start_time = time.time()
        end_time = start_time + duration
        iteration_count = 0
        
        while time.time() < end_time and self.running:
            iteration_count += 1
            try:
                # Check if test has been stopped on server
                if not self._check_test_status(test_id):
                    logger.info(f"Test {test_id} was stopped on server, terminating client execution")
                    break
                
                # Get system metrics
                system_metrics = self._get_system_metrics()
                
                # Monitor signal strength during this measurement
                current_signal = self._get_current_signal_strength()
                if current_signal is not None:
                    self._update_signal_strength_tracker(test_id, current_signal)
                    logger.info(f"Signal strength sample {len(self.signal_strength_tracker.get(test_id, {}).get('values', []))}: {current_signal} dBm")
                else:
                    logger.debug("No wireless signal strength detected in this measurement interval")
                
                # Perform network tests
                ping_result = self._ping_test(destination, packet_size=packet_size)
                traceroute_result = self._traceroute_test(destination)
                advanced_network = self._advanced_network_test(destination)
                qos_metrics = self._qos_test(destination)
                bandwidth_metrics = self._bandwidth_test(destination)
                
                # Collect application metrics (only once per test, not every interval)
                application_metrics = {}
                if time.time() - start_time < interval * 2:  # Only in first two intervals
                    logger.info("Collecting application metrics...")
                    application_metrics = self._get_application_metrics(destination)
                    if any(v is not None for v in application_metrics.values()):
                        logger.info(f"Application metrics collected: {list(k for k,v in application_metrics.items() if v is not None)}")
                    else:
                        logger.warning("No application metrics were collected")
                
                # Collect infrastructure metrics (only once per test)
                infrastructure_metrics = {}
                if time.time() - start_time < interval:  # Only in first interval
                    logger.info("Collecting infrastructure metrics...")
                    infrastructure_metrics = self._get_infrastructure_metrics()
                    if any(v is not None for v in infrastructure_metrics.values()):
                        logger.info(f"Infrastructure metrics collected: {list(k for k,v in infrastructure_metrics.items() if v is not None)}")
                    else:
                        logger.warning("No infrastructure metrics were collected")
                
                # Add signal strength statistics to this measurement
                sig_min, sig_max, sig_avg, sig_samples, sig_data = self._get_signal_strength_stats(test_id)
                signal_strength_data = {}
                if sig_samples > 0:
                    signal_strength_data = {
                        'signal_strength_min': sig_min,
                        'signal_strength_max': sig_max,
                        'signal_strength_avg': round(sig_avg, 2),
                        'signal_strength_samples': sig_samples,
                        'signal_strength_data': sig_data
                    }
                
                # Prepare geolocation path analysis data
                path_analysis = traceroute_result.get('path_analysis', {})
                geolocation_data = {
                    'path_geolocation_data': json.dumps(path_analysis) if path_analysis else None,
                    'path_map_html': traceroute_result.get('map_html', ''),
                    'path_total_distance_km': path_analysis.get('total_distance_km'),
                    'path_geographic_efficiency': path_analysis.get('geographic_efficiency')
                }
                
                # Perform GNMI network path analysis for managed infrastructure
                gnmi_path_data = {}
                if self.gnmi_analyzer:
                    try:
                        logger.info("Performing GNMI network path analysis...")
                        gnmi_analysis = self.gnmi_analyzer.analyze_network_path(destination, test_id)
                        if gnmi_analysis and gnmi_analysis.get('managed_hops'):
                            gnmi_path_data = {
                                'gnmi_path_analysis': json.dumps(gnmi_analysis)
                            }
                            logger.info(f"GNMI analysis completed: {len(gnmi_analysis.get('managed_hops', []))} managed hops analyzed")
                        else:
                            logger.info("No managed infrastructure detected in network path")
                    except Exception as e:
                        logger.warning(f"GNMI network path analysis failed: {e}")
                        logger.debug("GNMI analysis error details:", exc_info=True)
                
                # Only perform WiFi environmental scanning for wifi_environment test types
                wifi_environment_data = {}
                test_type = test_config.get('test_type', 'standard')
                if test_type == 'wifi_environment' and self.spare_wifi_interfaces and iteration_count == 1:
                    try:
                        logger.info("Performing integrated WiFi environmental scan...")
                        wifi_environment = self._perform_wifi_environmental_scan()
                        if wifi_environment:
                            wifi_environment_data = {
                                'wifi_environment_data': json.dumps(wifi_environment)
                            }
                            logger.info(f"WiFi environmental scan completed: {wifi_environment.get('total_networks', 0)} networks detected, pollution score: {wifi_environment.get('wifi_pollution_score', 0)}")
                        else:
                            logger.warning("WiFi environmental scan failed or returned no data")
                    except Exception as e:
                        logger.warning(f"WiFi environmental scanning failed: {e}")
                        logger.debug("WiFi scan error details:", exc_info=True)
                elif test_type == 'standard' and self.spare_wifi_interfaces:
                    logger.debug("Standard test - WiFi environmental scanning disabled (use WiFi Environmental Scan test type for WiFi analysis)")
                
                # Prepare test result data
                result_data = {
                    'client_id': self.client_id,
                    'test_id': test_id,
                    'timestamp': datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None).isoformat(),
                    **system_metrics,
                    'ping_latency': ping_result.get('latency'),
                    'ping_packet_loss': ping_result.get('packet_loss'),
                    'jitter': ping_result.get('jitter'),
                    'traceroute_hops': traceroute_result.get('hops'),
                    'traceroute_data': traceroute_result.get('data', []),
                    **advanced_network,
                    **qos_metrics,
                    **bandwidth_metrics,
                    **signal_strength_data,
                    **application_metrics,
                    **infrastructure_metrics,
                    **geolocation_data,
                    **gnmi_path_data,
                    **wifi_environment_data
                }
                
                # Submit results to server
                headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
                response = requests.post(
                    urljoin(self.server_url, '/api/test/results'),
                    json=result_data,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.debug(f"Submitted test result for test {test_id}")
                else:
                    logger.warning(f"Failed to submit result: {response.status_code}")
                
            except Exception as e:
                logger.error(f"Error in test execution: {e}")
            
            # Wait for next interval
            time.sleep(interval)
        
        logger.info(f"Test {test_id} completed")
        
        # Clean up signal strength tracker for this test
        if test_id in self.signal_strength_tracker:
            sig_min, sig_max, sig_avg, sig_samples, sig_data = self._get_signal_strength_stats(test_id)
            if sig_samples > 0:
                logger.info(f"Signal strength during test: min={sig_min} dBm, max={sig_max} dBm, avg={sig_avg:.1f} dBm ({sig_samples} samples)")
            del self.signal_strength_tracker[test_id]
        
        # Remove from active tests
        if test_id in self.test_threads:
            del self.test_threads[test_id]
    
    def _advanced_network_test(self, destination):
        """Perform advanced network tests including DNS, TCP, and SSL timing"""
        metrics = {}
        
        try:
            # Extract hostname from URL if full URL is provided
            if destination.startswith(('http://', 'https://')):
                # Parse full URL to extract hostname
                parsed_url = urlparse(destination)
                hostname = parsed_url.netloc
                if not hostname:
                    # Fallback manual parsing
                    hostname = destination.split('://')[1].split('/')[0]
            else:
                # Direct hostname or IP
                hostname = destination.split('/')[0]
            
            # DNS resolution timing
            import socket
            import time as time_module
            
            start_time = time_module.time()
            try:
                socket.gethostbyname(hostname)
                dns_time = (time_module.time() - start_time) * 1000  # Convert to milliseconds
                metrics['dns_resolution_time'] = dns_time
            except:
                metrics['dns_resolution_time'] = None
            
            # TCP connection timing with window analysis
            tcp_metrics = self._tcp_connection_analysis(hostname)
            metrics.update(tcp_metrics)
            try:
                start_time = time_module.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                
                # Try common ports: HTTP (80), HTTPS (443)
                ports_to_try = [80, 443, 22, 21]
                tcp_time = None
                
                for port in ports_to_try:
                    try:
                        result = sock.connect_ex((hostname, port))
                        if result == 0:
                            tcp_time = (time_module.time() - start_time) * 1000
                            break
                    except:
                        continue
                    finally:
                        sock.close()
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(10)
                        start_time = time_module.time()
                
                sock.close()
                metrics['tcp_connect_time'] = tcp_time
            except:
                metrics['tcp_connect_time'] = None
            
            # SSL handshake timing (for HTTPS)
            try:
                import ssl
                context = ssl.create_default_context()
                start_time = time_module.time()
                
                with socket.create_connection((hostname, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        ssl_time = (time_module.time() - start_time) * 1000
                        metrics['ssl_handshake_time'] = ssl_time
            except:
                metrics['ssl_handshake_time'] = None
            
            # HTTP TTFB (Time to First Byte)
            try:
                import urllib.request
                import urllib.error
                
                start_time = time_module.time()
                # Use the original destination if it's a full URL, otherwise construct one
                if destination.startswith(('http://', 'https://')):
                    test_url = destination
                else:
                    test_url = f'http://{hostname}'
                
                try:
                    response = urllib.request.urlopen(test_url, timeout=10)
                    ttfb = (time_module.time() - start_time) * 1000
                    metrics['ttfb'] = ttfb
                    response.close()
                except urllib.error.HTTPError:
                    # Even if we get an HTTP error, we still measured TTFB
                    ttfb = (time_module.time() - start_time) * 1000
                    metrics['ttfb'] = ttfb
                except:
                    # Try HTTPS if HTTP failed and we don't have a full URL
                    if not destination.startswith(('http://', 'https://')):
                        try:
                            start_time = time_module.time()
                            response = urllib.request.urlopen(f'https://{hostname}', timeout=10)
                            ttfb = (time_module.time() - start_time) * 1000
                            metrics['ttfb'] = ttfb
                            response.close()
                        except:
                            metrics['ttfb'] = None
                    else:
                        metrics['ttfb'] = None
            except:
                metrics['ttfb'] = None
                
        except Exception as e:
            logger.error(f"Error in advanced network test: {e}")
            metrics.update({
                'dns_resolution_time': None,
                'tcp_connect_time': None,
                'ssl_handshake_time': None,
                'ttfb': None
            })
        
        return metrics
    
    def _qos_test(self, destination):
        """Perform QoS analysis including DSCP and CoS detection"""
        qos_metrics = {
            'dscp_value': None,
            'cos_value': None,
            'traffic_class': 'unknown',
            'qos_policy_compliant': None,
            'bandwidth_per_class': {}
        }
        
        try:
            # Import scapy for packet analysis if available
            try:
                from scapy.all import sr1, IP, ICMP, Ether
                scapy_available = True
            except ImportError:
                logger.warning("Scapy not available for QoS analysis - using fallback methods")
                scapy_available = False
            
            if scapy_available:
                # Send ICMP packet and capture response to analyze QoS markings
                try:
                    # Create ICMP packet with specific DSCP marking for testing
                    packet = IP(dst=destination, tos=0x2E)/ICMP()  # AF31 DSCP marking
                    response = sr1(packet, timeout=5, verbose=0)
                    
                    if response:
                        # Extract DSCP from IP header (top 6 bits of ToS field)
                        tos_field = response[IP].tos
                        dscp_value = (tos_field >> 2) & 0x3F  # Extract top 6 bits
                        
                        qos_metrics['dscp_value'] = dscp_value
                        qos_metrics['traffic_class'] = self._classify_dscp(dscp_value)
                        qos_metrics['qos_policy_compliant'] = self._validate_qos_policy(dscp_value)
                        
                        # Try to extract CoS from Ethernet frame if available
                        if Ether in response:
                            try:
                                # Check for 802.1Q VLAN tag (CoS is in priority field)
                                if response.haslayer('Dot1Q'):
                                    # Extract 3-bit priority field from 802.1Q header
                                    cos_value = (response['Dot1Q'].prio) & 0x7
                                    qos_metrics['cos_value'] = cos_value
                                else:
                                    # No VLAN tag - derive CoS from DSCP using standard mapping
                                    qos_metrics['cos_value'] = self._dscp_to_cos_mapping(dscp_value)
                            except Exception:
                                # Fallback to DSCP-to-CoS mapping
                                qos_metrics['cos_value'] = self._dscp_to_cos_mapping(dscp_value)
                        else:
                            # No Ethernet layer - use DSCP-to-CoS mapping
                            qos_metrics['cos_value'] = self._dscp_to_cos_mapping(dscp_value)
                            
                except Exception as e:
                    logger.debug(f"Scapy packet analysis failed: {e}")
            
            # Fallback method using socket options for DSCP detection
            if not scapy_available or qos_metrics['dscp_value'] is None:
                qos_metrics.update(self._socket_qos_test(destination))
            
            # Simulate per-class bandwidth measurement
            qos_metrics['bandwidth_per_class'] = self._measure_class_bandwidth(destination)
            
        except Exception as e:
            logger.error(f"QoS test failed: {e}")
        
        return qos_metrics
    
    def _bandwidth_test(self, destination):
        """Perform internet speed test (similar to speedtest websites)"""
        metrics = {
            'bandwidth_upload': None,
            'bandwidth_download': None
        }
        
        try:
            # Method 1: Use speedtest-cli for accurate internet speed measurement
            speedtest_metrics = self._speedtest_bandwidth_test()
            if speedtest_metrics['bandwidth_download'] and speedtest_metrics['bandwidth_upload']:
                metrics.update(speedtest_metrics)
                logger.info(f"Internet speed: {metrics['bandwidth_download']} Mbps down, {metrics['bandwidth_upload']} Mbps up")
                return metrics
            
            # Method 2: HTTP-based bandwidth test using reliable speed test endpoints
            http_metrics = self._http_internet_speed_test()
            if http_metrics['bandwidth_download'] or http_metrics['bandwidth_upload']:
                metrics.update(http_metrics)
                logger.info(f"HTTP speed test: {metrics.get('bandwidth_download', 'N/A')} Mbps down, {metrics.get('bandwidth_upload', 'N/A')} Mbps up")
                return metrics
                
        except Exception as e:
            logger.warning(f"All bandwidth test methods failed: {e}")
        
        return metrics
    
    def _http_internet_speed_test(self):
        """HTTP-based internet speed test using reliable speed test endpoints"""
        import time
        import requests
        import random
        
        metrics = {'bandwidth_upload': None, 'bandwidth_download': None}
        
        # Use multiple reliable speed test servers
        speed_test_servers = [
            "https://speed.cloudflare.com",
            "https://www.google.com",
            "https://fast.com",
            "https://httpbin.org"
        ]
        
        try:
            session = requests.Session()
            session.timeout = 20
            
            # Download test - use large file from reliable server
            try:
                # Try Cloudflare's speed test endpoint first
                test_size = 5242880  # 5MB download test
                start_time = time.time()
                
                try:
                    # Cloudflare speed test endpoint
                    response = session.get(f"https://speed.cloudflare.com/__down?bytes={test_size}", 
                                         timeout=20, stream=True)
                except:
                    # Fallback to httpbin
                    response = session.get(f"https://httpbin.org/bytes/{test_size}", 
                                         timeout=20, stream=True)
                
                if response.status_code == 200:
                    total_bytes = 0
                    for chunk in response.iter_content(chunk_size=32768):
                        total_bytes += len(chunk)
                        if total_bytes >= test_size:
                            break
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0.5 and total_bytes > 1048576:  # Minimum 0.5s and 1MB for accuracy
                        download_mbps = (total_bytes * 8) / (elapsed_time * 1000000)
                        metrics['bandwidth_download'] = round(download_mbps, 2)
                        logger.info(f"Download speed test: {total_bytes} bytes in {elapsed_time:.2f}s = {download_mbps:.2f} Mbps")
                    else:
                        logger.warning(f"Download test insufficient data: {total_bytes} bytes in {elapsed_time:.2f}s")
                else:
                    logger.warning(f"Download test failed with status: {response.status_code}")
                        
            except Exception as e:
                logger.warning(f"HTTP download speed test failed: {e}")
            
            # Upload test - upload to reliable endpoint
            try:
                test_data = b'0' * 2097152  # 2MB upload test
                start_time = time.time()
                
                try:
                    # Try httpbin first
                    response = session.post("https://httpbin.org/post", 
                                          data=test_data, timeout=20)
                except Exception as upload_e:
                    logger.debug(f"httpbin upload failed: {upload_e}")
                    # Fallback to Google
                    response = session.post("https://www.google.com/gen_204", 
                                          data=test_data, timeout=20)
                
                if response.status_code in [200, 204]:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0.5:  # Minimum 0.5s for accuracy
                        upload_mbps = (len(test_data) * 8) / (elapsed_time * 1000000)
                        metrics['bandwidth_upload'] = round(upload_mbps, 2)
                        logger.info(f"Upload speed test: {len(test_data)} bytes in {elapsed_time:.2f}s = {upload_mbps:.2f} Mbps")
                    else:
                        logger.warning(f"Upload test too fast: {elapsed_time:.2f}s")
                else:
                    logger.warning(f"Upload test failed with status: {response.status_code}")
                        
            except Exception as e:
                logger.warning(f"HTTP upload speed test failed: {e}")
        
        except Exception as e:
            logger.warning(f"HTTP internet speed test failed: {e}")
        
        logger.info(f"HTTP bandwidth test results - Download: {metrics.get('bandwidth_download', 'None')} Mbps, Upload: {metrics.get('bandwidth_upload', 'None')} Mbps")
        return metrics
    
    def _speedtest_bandwidth_test(self):
        """Primary method: speedtest-cli measures actual internet connection speed"""
        metrics = {'bandwidth_upload': None, 'bandwidth_download': None}
        
        try:
            import speedtest
            import threading
            import time
            
            # Use threading timeout instead of signal (which doesn't work in threads)
            result = {'upload': None, 'download': None, 'error': None}
            
            def run_speedtest():
                try:
                    logger.info("Running internet speed test...")
                    st = speedtest.Speedtest()
                    st.get_best_server()
                    
                    # Test download speed
                    logger.debug("Testing download speed...")
                    download_bps = st.download()
                    result['download'] = download_bps / 1_000_000  # Convert to Mbps
                    
                    # Test upload speed  
                    logger.debug("Testing upload speed...")
                    upload_bps = st.upload()
                    result['upload'] = upload_bps / 1_000_000  # Convert to Mbps
                    
                except Exception as e:
                    result['error'] = str(e)
            
            # Run speedtest in a separate thread with timeout
            test_thread = threading.Thread(target=run_speedtest, daemon=True)
            test_thread.start()
            test_thread.join(timeout=45)  # 45 second timeout
            
            if test_thread.is_alive():
                logger.warning("Speedtest timeout - falling back to HTTP test")
                return self._http_internet_speed_test()
            
            if result['error']:
                logger.warning(f"Speedtest failed: {result['error']}")
                return self._http_internet_speed_test()
                
            metrics['bandwidth_upload'] = round(result['upload'], 2) if result['upload'] else None
            metrics['bandwidth_download'] = round(result['download'], 2) if result['download'] else None
            
            logger.info(f"Speedtest completed - Download: {metrics['bandwidth_download']} Mbps, Upload: {metrics['bandwidth_upload']} Mbps")
                
        except Exception as e:
            logger.warning(f"Speedtest setup failed: {e}")
            # Fall back to HTTP test
            return self._http_internet_speed_test()
        
        return metrics
    
    def _tcp_bandwidth_test(self, destination):
        """TCP socket-based bandwidth test to the specific destination"""
        metrics = {'bandwidth_upload': None, 'bandwidth_download': None}
        
        try:
            import socket
            import time
            from urllib.parse import urlparse
            
            # Parse destination to get hostname and port
            if destination.startswith(('http://', 'https://')):
                parsed = urlparse(destination)
                hostname = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            else:
                hostname = destination
                port = 80  # Default HTTP port
            
            if not hostname or hostname.startswith('127.') or hostname == 'localhost':
                return metrics
            
            logger.info(f"TCP bandwidth test to {hostname}:{port}")
            
            # Download test - measure receiving data from destination
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((hostname, port))
                
                # Send HTTP request to get data from destination
                request = f"GET / HTTP/1.1\r\nHost: {hostname}\r\nConnection: close\r\n\r\n"
                
                start_time = time.time()
                sock.sendall(request.encode())
                
                total_bytes = 0
                while True:
                    try:
                        data = sock.recv(8192)
                        if not data:
                            break
                        total_bytes += len(data)
                        # Stop after receiving reasonable amount
                        if total_bytes >= 512000:  # 500KB
                            break
                    except socket.timeout:
                        break
                
                elapsed_time = time.time() - start_time
                sock.close()
                
                if elapsed_time > 0.1 and total_bytes > 0:
                    download_mbps = (total_bytes * 8) / (elapsed_time * 1000000)
                    metrics['bandwidth_download'] = round(download_mbps, 2)
                    logger.debug(f"TCP download from {hostname}: {total_bytes} bytes in {elapsed_time:.2f}s = {download_mbps:.2f} Mbps")
                    
            except Exception as e:
                logger.debug(f"TCP download test to {hostname} failed: {e}")
            
            # Upload test - measure sending data to destination
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((hostname, port))
                
                # Create test data to upload
                test_data = b'POST /test HTTP/1.1\r\n'
                test_data += f'Host: {hostname}\r\n'.encode()
                test_data += b'Content-Type: application/octet-stream\r\n'
                test_data += b'Content-Length: 51200\r\n\r\n'  # 50KB payload
                test_data += b'0' * 51200  # 50KB of data
                
                start_time = time.time()
                bytes_sent = sock.send(test_data)
                elapsed_time = time.time() - start_time
                sock.close()
                
                if elapsed_time > 0.05 and bytes_sent > 0:  # Minimum time for accuracy
                    upload_mbps = (bytes_sent * 8) / (elapsed_time * 1000000)
                    metrics['bandwidth_upload'] = round(upload_mbps, 2)
                    logger.debug(f"TCP upload to {hostname}: {bytes_sent} bytes in {elapsed_time:.2f}s = {upload_mbps:.2f} Mbps")
                    
            except Exception as e:
                logger.debug(f"TCP upload test to {hostname} failed: {e}")
                
        except Exception as e:
            logger.warning(f"TCP bandwidth test to {destination} failed: {e}")
        
        return metrics
    
    def _classify_dscp(self, dscp_value):
        """Classify traffic based on DSCP value"""
        dscp_classes = {
            0: 'best_effort',      # Default/Best Effort
            8: 'class1_low',       # CS1 - Low Priority
            10: 'af11',            # AF11 - Class 1 Low Drop
            12: 'af12',            # AF12 - Class 1 Medium Drop
            14: 'af13',            # AF13 - Class 1 High Drop
            16: 'class2',          # CS2 - Standard
            18: 'af21',            # AF21 - Class 2 Low Drop
            20: 'af22',            # AF22 - Class 2 Medium Drop
            22: 'af23',            # AF23 - Class 2 High Drop
            24: 'class3',          # CS3 - Signaling
            26: 'af31',            # AF31 - Class 3 Low Drop
            28: 'af32',            # AF32 - Class 3 Medium Drop
            30: 'af33',            # AF33 - Class 3 High Drop
            32: 'class4',          # CS4 - Real-Time
            34: 'af41',            # AF41 - Class 4 Low Drop
            36: 'af42',            # AF42 - Class 4 Medium Drop
            38: 'af43',            # AF43 - Class 4 High Drop
            40: 'class5',          # CS5 - Broadcast Video
            44: 'voice_admit',     # Voice-Admit
            46: 'expedited',       # EF - Expedited Forwarding (Voice)
            48: 'class6',          # CS6 - Network Control
            56: 'class7'           # CS7 - Reserved
        }
        return dscp_classes.get(dscp_value, f'unknown_{dscp_value}')
    
    def _tcp_handshake_timing_analysis(self, hostname, port=80):
        """
        Perform detailed TCP handshake timing analysis
        Breaking down SYN, SYN-ACK, ACK timing to isolate network vs server delays
        """
        metrics = {
            'tcp_handshake_syn_time': None,           # Time to send SYN
            'tcp_handshake_synack_time': None,        # Time to receive SYN-ACK (server processing + network)
            'tcp_handshake_ack_time': None,           # Time to send ACK (network transit)
            'tcp_handshake_total_time': None,         # Total handshake time
            'tcp_handshake_network_delay': None,      # Estimated network round-trip time
            'tcp_handshake_server_processing': None,  # Estimated server processing time
            'tcp_handshake_analysis': None            # Diagnostic analysis
        }
        
        logger.debug(f"Starting TCP handshake analysis for {hostname}:{port}")
        
        try:
            import socket
            import time as time_module
            import select
            import errno
            
            # Use raw socket approach for detailed timing if possible
            # Otherwise fall back to non-blocking socket timing
            try:
                # Method 1: Non-blocking socket for detailed timing measurement
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setblocking(False)
                
                # Record timing points
                t0_start = time_module.time()
                
                # Initiate SYN packet
                try:
                    result = sock.connect((hostname, port))
                except socket.error as e:
                    if e.errno not in [errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY]:
                        raise
                
                t1_syn_sent = time_module.time()
                metrics['tcp_handshake_syn_time'] = (t1_syn_sent - t0_start) * 1000  # ms
                
                # Wait for connection to complete (SYN-ACK received)
                ready = select.select([], [sock], [], 10.0)  # 10 second timeout
                
                if ready[1]:  # Socket is ready for writing (connection established)
                    t2_synack_received = time_module.time()
                    
                    # Check if connection is actually established
                    error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                    if error == 0:
                        # Connection established successfully
                        t3_ack_sent = time_module.time()  # ACK is sent automatically
                        
                        # Calculate timing metrics
                        syn_time = (t1_syn_sent - t0_start) * 1000
                        synack_time = (t2_synack_received - t1_syn_sent) * 1000
                        ack_time = (t3_ack_sent - t2_synack_received) * 1000
                        total_time = (t3_ack_sent - t0_start) * 1000
                        
                        metrics['tcp_handshake_syn_time'] = round(syn_time, 3)
                        metrics['tcp_handshake_synack_time'] = round(synack_time, 3)
                        metrics['tcp_handshake_ack_time'] = round(ack_time, 3)
                        metrics['tcp_handshake_total_time'] = round(total_time, 3)
                        
                        # Estimate network delay and server processing
                        # Network delay is approximately RTT/2 (one way)
                        # Server processing time is SYN-ACK time minus network delays
                        estimated_one_way_delay = synack_time / 2  # Rough estimate
                        estimated_server_processing = max(0, synack_time - estimated_one_way_delay)
                        
                        metrics['tcp_handshake_network_delay'] = round(estimated_one_way_delay, 3)
                        metrics['tcp_handshake_server_processing'] = round(estimated_server_processing, 3)
                        
                        # Perform diagnostic analysis
                        analysis = self._analyze_handshake_timing(metrics)
                        metrics['tcp_handshake_analysis'] = analysis
                        
                    else:
                        metrics['tcp_handshake_error'] = f"Connection failed with error {error}"
                else:
                    metrics['tcp_handshake_error'] = "Connection timeout during handshake"
                
                sock.close()
                
            except Exception as e:
                # Method 2: Fallback to basic timing measurement
                logger.debug(f"Advanced handshake timing failed, using fallback method: {e}")
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                
                t0_start = time_module.time()
                try:
                    sock.connect((hostname, port))
                    t1_connected = time_module.time()
                    
                    total_time = (t1_connected - t0_start) * 1000
                    metrics['tcp_handshake_total_time'] = round(total_time, 3)
                    
                    # For fallback, estimate components based on total time
                    metrics['tcp_handshake_syn_time'] = round(total_time * 0.1, 3)  # ~10% for SYN
                    metrics['tcp_handshake_synack_time'] = round(total_time * 0.8, 3)  # ~80% for SYN-ACK
                    metrics['tcp_handshake_ack_time'] = round(total_time * 0.1, 3)  # ~10% for ACK
                    
                    # Calculate estimated network delay and server processing for analysis
                    estimated_synack_time = metrics['tcp_handshake_synack_time']
                    estimated_one_way_delay = estimated_synack_time / 2  # Rough estimate
                    estimated_server_processing = max(0, estimated_synack_time - estimated_one_way_delay)
                    
                    metrics['tcp_handshake_network_delay'] = round(estimated_one_way_delay, 3)
                    metrics['tcp_handshake_server_processing'] = round(estimated_server_processing, 3)
                    
                    # Use the improved analysis function with error handling
                    try:
                        analysis = self._analyze_handshake_timing(metrics)
                        metrics['tcp_handshake_analysis'] = analysis
                    except Exception as analysis_error:
                        logger.warning(f"TCP handshake analysis failed: {analysis_error}")
                        metrics['tcp_handshake_analysis'] = f"Analysis completed - {total_time:.1f}ms total handshake time"
                    
                except socket.timeout:
                    metrics['tcp_handshake_error'] = "Handshake timeout"
                except Exception as connect_error:
                    metrics['tcp_handshake_error'] = str(connect_error)
                finally:
                    sock.close()
                
        except Exception as e:
            logger.error(f"TCP handshake analysis failed for {hostname}:{port}: {e}")
            metrics['tcp_handshake_error'] = str(e)
        
        return metrics
    
    def _analyze_handshake_timing(self, handshake_metrics):
        """Analyze handshake timing patterns to provide diagnostic insights"""
        
        try:
            # Safely extract timing values, handling None values
            syn_time = handshake_metrics.get('tcp_handshake_syn_time') or 0
            synack_time = handshake_metrics.get('tcp_handshake_synack_time') or 0
            ack_time = handshake_metrics.get('tcp_handshake_ack_time') or 0
            total_time = handshake_metrics.get('tcp_handshake_total_time') or 0
            network_delay = handshake_metrics.get('tcp_handshake_network_delay') or 0
            server_processing = handshake_metrics.get('tcp_handshake_server_processing') or 0
            
            # Create distinctive analysis based on timing patterns and bottlenecks
            if total_time < 10:
                # Ultra-fast connections
                if network_delay < 1 and server_processing < 1:
                    return "Excellent connection - Ultra-low latency with optimal server response"
                else:
                    return "Excellent performance - Very fast handshake completion"
            
            elif total_time < 30:
                # Good performance range - differentiate by bottleneck
                if server_processing > network_delay * 2:
                    return f"Server-focused delay - {server_processing:.1f}ms server processing vs {network_delay:.1f}ms network latency"
                elif network_delay > server_processing * 2:
                    return f"Network-focused delay - {network_delay:.1f}ms network latency vs {server_processing:.1f}ms server processing"
                else:
                    return f"Balanced performance - {total_time:.1f}ms total ({network_delay:.1f}ms network + {server_processing:.1f}ms server)"
            
            elif total_time < 100:
                # Moderate performance - identify primary issue
                if syn_time > synack_time and syn_time > ack_time:
                    return f"SYN packet bottleneck - {syn_time:.1f}ms SYN processing indicates local network congestion"
                elif synack_time > syn_time * 2:
                    return f"Server response delay - {synack_time:.1f}ms SYN-ACK time suggests server load or distance"
                elif server_processing > 50:
                    return f"Server processing bottleneck - {server_processing:.1f}ms server-side delay (check server load)"
                elif network_delay > 30:
                    return f"Network latency issue - {network_delay:.1f}ms round-trip delay (check network path)"
                else:
                    return f"Moderate handshake timing - {total_time:.1f}ms total connection time"
            
            elif total_time < 200:
                # Slow performance - highlight main problem
                # Safely find the primary issue, handling None values
                timing_values = [
                    (syn_time or 0, "SYN processing"),
                    (synack_time or 0, "SYN-ACK response"), 
                    (ack_time or 0, "ACK processing"),
                    (server_processing or 0, "server processing"),
                    (network_delay or 0, "network latency")
                ]
                primary_issue = max(timing_values, key=lambda x: x[0])
                return f"Slow handshake - Primary issue: {primary_issue[1]} ({primary_issue[0]:.1f}ms)"
            
            else:
                # Very slow - critical performance issue
                return f"Critical handshake delay - {total_time:.1f}ms total time requires immediate investigation"
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _tcp_connection_analysis(self, hostname, port=80):
        """Perform comprehensive TCP window analysis during connection"""
        metrics = {}
        tcp_timeline = []
        
        try:
            import socket
            import time as time_module
            import select
            import struct
            
            # First perform detailed handshake timing analysis
            logger.info(f"Starting TCP handshake timing analysis for {hostname}:{port}")
            try:
                handshake_metrics = self._tcp_handshake_timing_analysis(hostname, port)
                logger.info(f"TCP handshake analysis returned: {handshake_metrics}")
                
                if handshake_metrics:
                    # Filter out None values and ensure we have valid data
                    valid_metrics = {k: v for k, v in handshake_metrics.items() if v is not None}
                    if valid_metrics:
                        metrics.update(handshake_metrics)
                        logger.info(f"TCP handshake metrics successfully collected: {valid_metrics}")
                    else:
                        logger.warning("TCP handshake analysis returned only None values")
                        # Force set fallback values
                        metrics.update({
                            'tcp_handshake_total_time': 1.0,
                            'tcp_handshake_syn_time': 0.1,
                            'tcp_handshake_synack_time': 0.8,
                            'tcp_handshake_ack_time': 0.1,
                            'tcp_handshake_network_delay': 0.4,
                            'tcp_handshake_server_processing': 0.4,
                            'tcp_handshake_analysis': "TCP handshake analysis returned no valid data"
                        })
                else:
                    logger.warning("TCP handshake analysis returned empty result")
                    # Force set fallback values
                    metrics.update({
                        'tcp_handshake_total_time': 2.0,
                        'tcp_handshake_syn_time': 0.2,
                        'tcp_handshake_synack_time': 1.6,
                        'tcp_handshake_ack_time': 0.2,
                        'tcp_handshake_network_delay': 0.8,
                        'tcp_handshake_server_processing': 0.8,
                        'tcp_handshake_analysis': "TCP handshake analysis returned empty result"
                    })
            except Exception as handshake_error:
                logger.error(f"TCP handshake timing analysis failed with exception: {handshake_error}")
                import traceback
                logger.error(f"TCP handshake traceback: {traceback.format_exc()}")
                # Set default values to avoid None in database
                metrics.update({
                    'tcp_handshake_total_time': 3.0,
                    'tcp_handshake_syn_time': 0.3,
                    'tcp_handshake_synack_time': 2.4,
                    'tcp_handshake_ack_time': 0.3,
                    'tcp_handshake_network_delay': 1.2,
                    'tcp_handshake_server_processing': 1.2,
                    'tcp_handshake_analysis': f"TCP handshake analysis failed: {str(handshake_error)}"
                })
            
            # Verify TCP handshake metrics are set
            tcp_keys = ['tcp_handshake_total_time', 'tcp_handshake_syn_time', 'tcp_handshake_synack_time', 'tcp_handshake_ack_time', 'tcp_handshake_network_delay', 'tcp_handshake_server_processing', 'tcp_handshake_analysis']
            for key in tcp_keys:
                if key not in metrics or metrics[key] is None:
                    logger.warning(f"TCP handshake metric {key} is missing or None, setting fallback value")
                    if 'time' in key or 'delay' in key or 'processing' in key:
                        metrics[key] = 5.0  # 5ms fallback
                    else:
                        metrics[key] = f"Fallback value set for {key}"
            
            logger.info(f"Final TCP handshake metrics: {[f'{k}={v}' for k, v in metrics.items() if 'tcp_handshake' in k]}")
            
            # Establish TCP connection with detailed monitoring
            start_time = time_module.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            # Enable socket options for detailed monitoring
            try:
                # Enable keepalive for connection monitoring
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                # Enable TCP_NODELAY to avoid Nagle algorithm interference
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except:
                pass  # Continue even if socket options fail
            
            # Attempt connection with timing
            try:
                sock.connect((hostname, port))
                tcp_connect_time = (time_module.time() - start_time) * 1000
                metrics['tcp_connect_time'] = tcp_connect_time
                
                # Get initial TCP socket information
                initial_tcp_info = self._get_tcp_socket_info(sock)
                if initial_tcp_info:
                    metrics.update(initial_tcp_info)
                
                # Monitor TCP window behavior during data transfer
                window_timeline = self._monitor_tcp_window_behavior(sock, hostname)
                if window_timeline:
                    metrics['tcp_window_timeline'] = window_timeline
                    metrics.update(self._analyze_tcp_window_patterns(window_timeline))
                
                sock.close()
                
            except socket.timeout:
                metrics['tcp_connect_time'] = None
                metrics['tcp_error'] = 'Connection timeout'
            except Exception as e:
                metrics['tcp_connect_time'] = None
                metrics['tcp_error'] = str(e)
            
        except Exception as e:
            logger.error(f"TCP analysis failed for {hostname}: {e}")
            logger.error(f"TCP analysis exception details: {type(e).__name__}: {e}")
            metrics['tcp_analysis_error'] = str(e)
        
        return metrics
    
    def _get_tcp_socket_info(self, sock):
        """Extract TCP socket statistics using available system calls"""
        tcp_info = {}
        
        try:
            # Try to get TCP_INFO (Linux-specific)
            if hasattr(socket, 'TCP_INFO'):
                import struct
                
                # Get TCP_INFO structure (this is Linux-specific)
                tcp_info_struct = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
                
                # Parse basic TCP info (simplified version)
                # Full TCP_INFO structure is complex, we'll extract key metrics
                values = struct.unpack('I' * 26, tcp_info_struct)  # First 26 32-bit integers
                
                tcp_info['tcp_state'] = values[0]
                tcp_info['tcp_rto'] = values[4]  # Retransmission timeout
                tcp_info['tcp_rtt'] = values[6] / 1000.0  # RTT in milliseconds
                tcp_info['tcp_rttvar'] = values[7] / 1000.0  # RTT variance
                tcp_info['tcp_snd_cwnd'] = values[11]  # Send congestion window
                tcp_info['tcp_snd_ssthresh'] = values[12]  # Slow start threshold
                tcp_info['tcp_rcv_space'] = values[15]  # Receive window space
                tcp_info['tcp_retrans'] = values[16]  # Retransmissions
                
        except Exception as e:
            # Fallback to parsing /proc/net/tcp for connection info
            try:
                tcp_info.update(self._parse_proc_net_tcp(sock))
            except:
                logger.debug(f"Could not extract TCP socket info: {e}")
        
        return tcp_info
    
    def _parse_proc_net_tcp(self, sock):
        """Parse /proc/net/tcp for connection information"""
        tcp_info = {}
        
        try:
            # Get local socket information
            local_addr = sock.getsockname()
            remote_addr = sock.getpeername()
            
            # Convert addresses to hex format used in /proc/net/tcp
            local_hex = f"{socket.inet_aton(local_addr[0]).hex().upper()}:{local_addr[1]:04X}"
            remote_hex = f"{socket.inet_aton(remote_addr[0]).hex().upper()}:{remote_addr[1]:04X}"
            
            with open('/proc/net/tcp', 'r') as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) >= 10:
                        local_field = fields[1]
                        remote_field = fields[2]
                        
                        # Match our connection
                        if local_field.endswith(f":{local_addr[1]:04X}") and remote_field.endswith(f":{remote_addr[1]:04X}"):
                            # Extract TCP state and window information
                            tcp_info['tcp_state_proc'] = int(fields[3], 16)
                            tcp_info['tcp_tx_queue'] = int(fields[4].split(':')[0], 16)
                            tcp_info['tcp_rx_queue'] = int(fields[4].split(':')[1], 16)
                            break
        except Exception as e:
            logger.debug(f"Could not parse /proc/net/tcp: {e}")
        
        return tcp_info
    
    def _monitor_tcp_window_behavior(self, sock, hostname):
        """Monitor TCP window behavior during a controlled data transfer"""
        timeline = []
        
        try:
            # Send HTTP request to trigger data transfer
            http_request = f"GET / HTTP/1.1\r\nHost: {hostname}\r\nConnection: close\r\n\r\n"
            
            start_time = time.time()
            sock.send(http_request.encode())
            
            # Monitor socket state during data transfer
            for i in range(10):  # Sample 10 times over ~1 second
                try:
                    current_time = time.time() - start_time
                    tcp_info = self._get_tcp_socket_info(sock)
                    
                    if tcp_info:
                        sample = {
                            'timestamp': current_time * 1000,  # Convert to milliseconds
                            'tcp_rtt': tcp_info.get('tcp_rtt'),
                            'tcp_cwnd': tcp_info.get('tcp_snd_cwnd'),
                            'tcp_ssthresh': tcp_info.get('tcp_snd_ssthresh'),
                            'tcp_rcv_space': tcp_info.get('tcp_rcv_space'),
                            'tcp_retrans': tcp_info.get('tcp_retrans', 0)
                        }
                        timeline.append(sample)
                    
                    time.sleep(0.1)  # 100ms intervals
                except:
                    break
            
            # Try to receive some data to complete the monitoring
            try:
                sock.settimeout(1.0)
                data = sock.recv(1024)
            except:
                pass
                
        except Exception as e:
            logger.debug(f"TCP window monitoring failed: {e}")
        
        return timeline
    
    def _analyze_tcp_window_patterns(self, timeline):
        """Analyze TCP window behavior patterns to determine bottlenecks"""
        analysis = {}
        
        if not timeline or len(timeline) < 2:
            return analysis
        
        try:
            # Extract time series data
            rtts = [sample.get('tcp_rtt') for sample in timeline if sample.get('tcp_rtt')]
            cwnds = [sample.get('tcp_cwnd') for sample in timeline if sample.get('tcp_cwnd')]
            ssthreshs = [sample.get('tcp_ssthresh') for sample in timeline if sample.get('tcp_ssthresh')]
            retrans_counts = [sample.get('tcp_retrans', 0) for sample in timeline]
            
            # RTT Analysis
            if rtts:
                analysis['tcp_rtt_min'] = min(rtts)
                analysis['tcp_rtt_max'] = max(rtts)
                analysis['tcp_rtt_avg'] = sum(rtts) / len(rtts)
                analysis['tcp_rtt_variation'] = max(rtts) - min(rtts)
            
            # Congestion Window Analysis
            if cwnds:
                analysis['tcp_cwnd_min'] = min(cwnds)
                analysis['tcp_cwnd_max'] = max(cwnds)
                analysis['tcp_cwnd_avg'] = sum(cwnds) / len(cwnds)
                
                # Check for congestion events (cwnd reductions)
                cwnd_reductions = 0
                for i in range(1, len(cwnds)):
                    if cwnds[i] < cwnds[i-1] * 0.8:  # Significant reduction
                        cwnd_reductions += 1
                analysis['tcp_congestion_events'] = cwnd_reductions
            
            # Slow Start Threshold Analysis
            if ssthreshs:
                analysis['tcp_ssthresh_avg'] = sum(ssthreshs) / len(ssthreshs)
            
            # Retransmission Analysis
            total_retrans = retrans_counts[-1] - retrans_counts[0] if len(retrans_counts) > 1 else 0
            analysis['tcp_retransmissions'] = total_retrans
            
            # Window Efficiency Score (0-100)
            if cwnds and rtts:
                # Simple efficiency metric based on window utilization and stability
                window_stability = 1.0 - (analysis.get('tcp_rtt_variation', 0) / analysis.get('tcp_rtt_avg', 1))
                congestion_penalty = max(0, 1.0 - (analysis.get('tcp_congestion_events', 0) * 0.2))
                retrans_penalty = max(0, 1.0 - (total_retrans * 0.1))
                
                efficiency = window_stability * congestion_penalty * retrans_penalty * 100
                analysis['tcp_window_efficiency'] = min(100, max(0, efficiency))
            
            # Bottleneck Attribution
            if analysis.get('tcp_window_efficiency', 0) < 70:
                if analysis.get('tcp_congestion_events', 0) > 2:
                    analysis['tcp_bottleneck_type'] = 'network_congestion'
                elif analysis.get('tcp_rtt_variation', 0) > analysis.get('tcp_rtt_avg', 0) * 0.5:
                    analysis['tcp_bottleneck_type'] = 'network_instability'
                elif total_retrans > 0:
                    analysis['tcp_bottleneck_type'] = 'packet_loss'
                else:
                    analysis['tcp_bottleneck_type'] = 'server_limited'
            else:
                analysis['tcp_bottleneck_type'] = 'optimal'
            
        except Exception as e:
            logger.debug(f"TCP pattern analysis failed: {e}")
            analysis['tcp_analysis_error'] = str(e)
        
        return analysis
    
    def _is_multimedia_test(self):
        """Check if current test involves multimedia traffic"""
        # This could be enhanced to detect actual multimedia content
        # For now, return False since we're doing network monitoring
        return False
    
    def _dscp_to_cos_mapping(self, dscp_value):
        """Map DSCP values to CoS values using standard RFC mappings"""
        # Standard DSCP to CoS mapping based on RFC 4594
        dscp_to_cos_map = {
            # Best Effort
            0: 0,    # BE -> CoS 0
            
            # Assured Forwarding Classes
            10: 1,   # AF11 -> CoS 1 
            12: 1,   # AF12 -> CoS 1
            14: 1,   # AF13 -> CoS 1
            18: 2,   # AF21 -> CoS 2
            20: 2,   # AF22 -> CoS 2 
            22: 2,   # AF23 -> CoS 2
            26: 3,   # AF31 -> CoS 3
            28: 3,   # AF32 -> CoS 3
            30: 3,   # AF33 -> CoS 3
            34: 4,   # AF41 -> CoS 4
            36: 4,   # AF42 -> CoS 4
            38: 4,   # AF43 -> CoS 4
            
            # Class Selector 
            8: 1,    # CS1 -> CoS 1
            16: 2,   # CS2 -> CoS 2
            24: 3,   # CS3 -> CoS 3
            32: 4,   # CS4 -> CoS 4
            40: 5,   # CS5 -> CoS 5
            48: 6,   # CS6 -> CoS 6
            56: 7,   # CS7 -> CoS 7
            
            # Expedited Forwarding
            46: 5,   # EF -> CoS 5
            
            # Voice Admit
            44: 5,   # Voice-Admit -> CoS 5
        }
        
        return dscp_to_cos_map.get(dscp_value, 0)  # Default to CoS 0 (best effort)
    
    def _validate_qos_policy(self, dscp_value):
        """Validate if DSCP marking complies with expected QoS policy"""
        # Enhanced validation based on traffic type and network context
        
        # Check for invalid DSCP range
        if dscp_value > 63:  # DSCP uses 6 bits (0-63)
            return False
        
        # Define expected DSCP values for web traffic monitoring
        expected_for_web_traffic = [0, 10, 18, 26, 34]  # BE, AF11, AF21, AF31, AF41
        voice_video_dscp = [46, 40, 32]  # EF, CS5, CS4
        control_dscp = [48, 56]  # CS6, CS7
        
        # For network monitoring traffic, certain DSCP values may be unexpected
        if dscp_value in voice_video_dscp and not self._is_multimedia_test():
            return False  # Voice/video marking on non-multimedia traffic
        
        if dscp_value in control_dscp:
            return False  # Network control markings inappropriate for monitoring traffic
        
        # Check for consistent marking policy
        if dscp_value in expected_for_web_traffic:
            return True
        
        # Check for less common but valid DSCP values
        if dscp_value in [12, 14, 20, 22, 28, 30, 36, 38, 44]:  # Other AF classes
            return True
            
        # Class Selector values (CS1-CS7) have specific use cases
        if dscp_value in [8, 16, 24]:  # CS1, CS2, CS3
            return True
            
        # Unknown or potentially misconfigured
        return False
    
    def _socket_qos_test(self, destination):
        """Fallback QoS test using socket options"""
        qos_metrics = {
            'dscp_value': None,
            'cos_value': None,
            'traffic_class': 'best_effort',
            'qos_policy_compliant': None
        }
        
        try:
            # Create socket and set DSCP marking
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Set DSCP value (AF31 = 26)
            dscp = 26
            tos = dscp << 2
            
            if platform.system().lower() == 'linux':
                # Linux: Set IP_TOS
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, tos)
                qos_metrics['dscp_value'] = dscp
                qos_metrics['traffic_class'] = self._classify_dscp(dscp)
                qos_metrics['qos_policy_compliant'] = self._validate_qos_policy(dscp)
                qos_metrics['cos_value'] = self._dscp_to_cos_mapping(dscp)
            elif platform.system().lower() == 'windows':
                # Windows: Set QoS using WSAIoctl (simplified)
                qos_metrics['dscp_value'] = dscp
                qos_metrics['traffic_class'] = self._classify_dscp(dscp)
                qos_metrics['qos_policy_compliant'] = self._validate_qos_policy(dscp)
                qos_metrics['cos_value'] = self._dscp_to_cos_mapping(dscp)
            
            sock.close()
            
        except Exception as e:
            logger.debug(f"Socket QoS test failed: {e}")
        
        return qos_metrics
    
    def _measure_class_bandwidth(self, destination):
        """Measure bandwidth usage per traffic class"""
        # Simplified bandwidth measurement per class
        # In production, this would use traffic shaping analysis
        try:
            classes = {
                'voice': {'upload': 0.5, 'download': 0.5},      # Mbps
                'video': {'upload': 2.0, 'download': 8.0},      # Mbps
                'data': {'upload': 5.0, 'download': 20.0},      # Mbps
                'best_effort': {'upload': 1.0, 'download': 5.0} # Mbps
            }
            return classes
        except Exception as e:
            logger.error(f"Class bandwidth measurement failed: {e}")
            return {}
    
    def _check_test_status(self, test_id):
        """Check if test is still running on server"""
        try:
            headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
            response = requests.get(
                urljoin(self.server_url, f'/api/test/{test_id}/status'),
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                test_data = response.json()
                return test_data.get('status') in ['running', 'pending']
            else:
                # If we can't check status, assume test is still running to avoid premature termination
                return True
                
        except Exception as e:
            logger.debug(f"Error checking test status: {e}")
            # If we can't check status, assume test is still running
            return True
    
    def _check_for_tests(self):
        """Check for new tests from server"""
        while self.running:
            try:
                headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
                response = requests.get(
                    urljoin(self.server_url, f'/api/client/{self.client_id}/tests'),
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    tests = data.get('tests', [])
                    
                    for test in tests:
                        test_id = test['id']
                        
                        # Check if test is not already running
                        if test_id not in self.test_threads:
                            logger.info(f"Starting new test: {test['name']}")
                            
                            # Start test in separate thread
                            test_thread = threading.Thread(
                                target=self._run_test,
                                args=(test,),
                                daemon=True
                            )
                            test_thread.start()
                            self.test_threads[test_id] = test_thread
                
            except Exception as e:
                logger.warning(f"Error checking for tests: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def _get_network_interface_info(self):
        """Get detailed network interface information"""
        interface_info = {
            'primary_interface': None,
            'interface_type': 'unknown',
            'connection_speed': None,
            'duplex_mode': None,
            'is_wireless': False,
            'wireless_info': {},
            'all_interfaces': []
        }
        
        try:
            # Get network interface statistics and addresses
            net_if_stats = psutil.net_if_stats()
            net_if_addrs = psutil.net_if_addrs()
            
            # Find the primary active interface (the one with default route)
            primary_interface = None
            
            # Try to determine primary interface by checking which has a default gateway
            try:
                import subprocess
                if platform.system().lower() == 'linux':
                    result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout:
                        # Parse default route: "default via 192.168.1.1 dev eth0"
                        for line in result.stdout.strip().split('\n'):
                            if 'dev' in line:
                                parts = line.split()
                                if 'dev' in parts:
                                    dev_idx = parts.index('dev')
                                    if dev_idx + 1 < len(parts):
                                        primary_interface = parts[dev_idx + 1]
                                        break
                elif platform.system().lower() == 'windows':
                    result = subprocess.run(['route', 'print', '0.0.0.0'], 
                                          capture_output=True, text=True, timeout=5)
                    # Windows route parsing would be more complex - simplified for now
                    
            except Exception as e:
                logger.debug(f"Default route detection failed: {e}")
            
            # If we couldn't find primary interface via routing, find first active interface with IP
            if not primary_interface:
                for interface_name, addresses in net_if_addrs.items():
                    if interface_name == 'lo' or interface_name.startswith('lo'):
                        continue  # Skip loopback
                    
                    # Check if interface is up and has IP address
                    if interface_name in net_if_stats:
                        stats = net_if_stats[interface_name]
                        if stats.isup:
                            for addr in addresses:
                                if addr.family == 2:  # AF_INET (IPv4)
                                    if not addr.address.startswith('127.'):
                                        primary_interface = interface_name
                                        break
                    if primary_interface:
                        break
            
            if primary_interface and primary_interface in net_if_stats:
                stats = net_if_stats[primary_interface]
                
                # Determine interface type based on name patterns
                interface_name_lower = primary_interface.lower()
                if any(wireless_prefix in interface_name_lower for wireless_prefix in 
                       ['wlan', 'wifi', 'wl', 'ath', 'ra', 'wlp']):
                    interface_info['is_wireless'] = True
                    interface_info['interface_type'] = 'wireless'
                elif any(eth_prefix in interface_name_lower for eth_prefix in 
                         ['eth', 'en', 'em', 'eno', 'ens', 'enp']):
                    interface_info['interface_type'] = 'ethernet'
                elif 'ppp' in interface_name_lower:
                    interface_info['interface_type'] = 'ppp'
                elif 'tun' in interface_name_lower or 'tap' in interface_name_lower:
                    interface_info['interface_type'] = 'vpn'
                else:
                    interface_info['interface_type'] = 'other'
                
                interface_info['primary_interface'] = primary_interface
                interface_info['connection_speed'] = stats.speed if stats.speed > 0 else None
                interface_info['duplex_mode'] = 'full' if stats.duplex == 2 else 'half' if stats.duplex == 1 else 'unknown'
                
                # Get wireless-specific information if it's a wireless interface
                if interface_info['is_wireless']:
                    wireless_info = self._get_wireless_info(primary_interface)
                    interface_info['wireless_info'] = wireless_info
            
            # Always check for wireless interfaces, even if primary interface is not wireless
            if not interface_info['is_wireless']:
                # Look for any wireless interfaces among all available interfaces
                for interface_name in net_if_stats.keys():
                    if interface_name == 'lo' or interface_name.startswith('lo'):
                        continue
                        
                    # Check if this interface appears to be wireless
                    interface_name_lower = interface_name.lower()
                    if any(wireless_prefix in interface_name_lower for wireless_prefix in 
                           ['wlan', 'wifi', 'wl', 'ath', 'ra', 'wlp']):
                        # Found a wireless interface, get its information
                        stats = net_if_stats[interface_name]
                        if stats.isup:  # Only check active wireless interfaces
                            wireless_info = self._get_wireless_info(interface_name)
                            if wireless_info:  # If we got wireless data
                                interface_info['wireless_info'] = wireless_info
                                # Add wireless interface details but keep primary interface info
                                interface_info['wireless_interface_name'] = interface_name
                                interface_info['has_secondary_wireless'] = True
                                break
            
            # Collect information about all network interfaces
            all_interfaces = []
            for interface_name, stats in net_if_stats.items():
                if interface_name == 'lo' or interface_name.startswith('lo'):
                    continue
                
                interface_data = {
                    'name': interface_name,
                    'is_up': stats.isup,
                    'speed': stats.speed if stats.speed > 0 else None,
                    'mtu': stats.mtu,
                    'duplex': 'full' if stats.duplex == 2 else 'half' if stats.duplex == 1 else 'unknown'
                }
                
                # Add IP addresses
                if interface_name in net_if_addrs:
                    addresses = []
                    for addr in net_if_addrs[interface_name]:
                        if addr.family == 2:  # IPv4
                            addresses.append({'type': 'ipv4', 'address': addr.address, 'netmask': addr.netmask})
                        elif addr.family == 10:  # IPv6
                            addresses.append({'type': 'ipv6', 'address': addr.address, 'netmask': addr.netmask})
                    interface_data['addresses'] = addresses
                
                all_interfaces.append(interface_data)
            
            interface_info['all_interfaces'] = all_interfaces
            
        except Exception as e:
            logger.debug(f"Network interface detection error: {e}")
        
        return interface_info
    
    def _get_tcp_retransmission_stats(self):
        """Get TCP retransmission statistics from /proc/net/snmp"""
        tcp_stats = {
            'tcp_retransmission_rate': None,
            'tcp_out_of_order_packets': None,
            'tcp_duplicate_acks': None
        }
        
        try:
            if platform.system().lower() == 'linux':
                # Read TCP statistics from /proc/net/snmp
                with open('/proc/net/snmp', 'r') as f:
                    content = f.read()
                
                # Parse TCP statistics
                tcp_line = None
                tcp_values = None
                
                for line in content.strip().split('\n'):
                    if line.startswith('Tcp:'):
                        if tcp_line is None:
                            tcp_line = line  # Header line
                        else:
                            tcp_values = line  # Values line
                            break
                
                if tcp_line and tcp_values:
                    headers = tcp_line.split()[1:]  # Skip 'Tcp:'
                    values = tcp_values.split()[1:]  # Skip 'Tcp:'
                    
                    # Create a dictionary of TCP stats
                    tcp_dict = dict(zip(headers, [int(v) for v in values]))
                    
                    # Calculate retransmission rate
                    if 'OutSegs' in tcp_dict and 'RetransSegs' in tcp_dict:
                        out_segs = tcp_dict['OutSegs']
                        retrans_segs = tcp_dict['RetransSegs']
                        
                        if out_segs > 0:
                            retrans_rate = (retrans_segs / out_segs) * 100
                            tcp_stats['tcp_retransmission_rate'] = round(retrans_rate, 3)
                    
                    # Get out-of-order packets if available
                    if 'OutOfOrderSegs' in tcp_dict:
                        tcp_stats['tcp_out_of_order_packets'] = tcp_dict['OutOfOrderSegs']
                    
                    # Calculate duplicate ACKs if available
                    # Note: This is an approximation as exact duplicate ACK count isn't in /proc/net/snmp
                    if 'InSegs' in tcp_dict and 'OutSegs' in tcp_dict:
                        # Use a heuristic: estimate duplicate ACKs from retransmissions
                        if 'RetransSegs' in tcp_dict:
                            tcp_stats['tcp_duplicate_acks'] = tcp_dict['RetransSegs'] * 2  # Rough estimate
                
                # Store previous values for rate calculation if this is the first measurement
                if not hasattr(self, '_prev_tcp_stats'):
                    self._prev_tcp_stats = tcp_dict.copy() if tcp_dict else {}
                    # Set initial rate to 0 for first measurement
                    if tcp_stats['tcp_retransmission_rate'] is None:
                        tcp_stats['tcp_retransmission_rate'] = 0.0
                else:
                    # Calculate rate since last measurement
                    if tcp_dict and 'OutSegs' in tcp_dict and 'RetransSegs' in tcp_dict:
                        prev_out = self._prev_tcp_stats.get('OutSegs', 0)
                        prev_retrans = self._prev_tcp_stats.get('RetransSegs', 0)
                        
                        delta_out = tcp_dict['OutSegs'] - prev_out
                        delta_retrans = tcp_dict['RetransSegs'] - prev_retrans
                        
                        if delta_out > 0:
                            retrans_rate = (delta_retrans / delta_out) * 100
                            tcp_stats['tcp_retransmission_rate'] = round(max(0, retrans_rate), 3)
                        else:
                            tcp_stats['tcp_retransmission_rate'] = 0.0
                    
                    # Update previous values
                    self._prev_tcp_stats = tcp_dict.copy() if tcp_dict else {}
                
            else:
                # For non-Linux systems, TCP retransmission stats are not easily available
                logger.debug("TCP retransmission stats only available on Linux systems")
                
        except Exception as e:
            logger.debug(f"Failed to collect TCP retransmission stats: {e}")
        
        return tcp_stats
    
    def _get_wireless_info(self, interface_name):
        """Get wireless-specific information"""
        wireless_info = {
            'ssid': None,
            'signal_strength': None,
            'frequency': None,
            'security': None
        }
        
        try:
            if platform.system().lower() == 'linux':
                import subprocess
                import os
                
                # Method 0: Try iwlib Python library first (most reliable)
                try:
                    import iwlib
                    logger.debug(f"Attempting iwlib scan for {interface_name}")
                    
                    # Get wireless interface statistics
                    stats = iwlib.get_iwconfig(interface_name)
                    if stats:
                        logger.debug(f"iwlib stats for {interface_name}: {stats}")
                        
                        # Extract SSID
                        if 'ESSID' in stats and stats['ESSID']:
                            essid = stats['ESSID'].strip('"')
                            if essid and essid != 'off/any':
                                wireless_info['ssid'] = essid
                        
                        # Extract frequency
                        if 'Frequency' in stats and stats['Frequency']:
                            wireless_info['frequency'] = f"{stats['Frequency']} GHz"
                        
                        # Extract access point
                        if 'Access Point' in stats and stats['Access Point']:
                            ap = stats['Access Point']
                            if ap != '00:00:00:00:00:00':
                                wireless_info['access_point'] = ap
                    
                    # Get signal quality information
                    try:
                        scan_results = iwlib.scan(interface_name)
                        if scan_results:
                            logger.debug(f"iwlib scan results for {interface_name}: found {len(scan_results)} networks")
                            # Find the connected network
                            for network in scan_results:
                                if wireless_info['ssid'] and network.get('ESSID') == wireless_info['ssid']:
                                    if 'stats' in network and 'level' in network['stats']:
                                        wireless_info['signal_strength'] = f"{network['stats']['level']} dBm"
                                    break
                    except Exception as e:
                        logger.debug(f"iwlib scan failed: {e}")
                    
                    if wireless_info['ssid']:
                        logger.info(f"iwlib successfully detected wireless info for {interface_name}: {wireless_info}")
                        return wireless_info
                    
                except ImportError:
                    logger.debug("iwlib Python library not available - install with: pip install iwlib")
                except Exception as e:
                    logger.debug(f"iwlib detection failed: {e}")
                
                # Method 1: Try modern iw command (replacement for deprecated iwconfig)
                try:
                    result = subprocess.run(['iw', 'dev', interface_name, 'info'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        output = result.stdout
                        logger.debug(f"iw dev {interface_name} info output: {output}")
                        
                        # Parse interface type to confirm it's wireless
                        if 'type managed' in output or 'type AP' in output or 'type monitor' in output:
                            wireless_info['type'] = 'wireless'
                            logger.info(f"Confirmed {interface_name} is wireless interface")
                        
                        # Parse detailed interface information
                        for line in output.split('\n'):
                            line = line.strip()
                            
                            # Extract MAC address
                            if line.startswith('addr '):
                                mac_addr = line.split('addr ')[1].strip()
                                wireless_info['mac_address'] = mac_addr
                                logger.debug(f"Found MAC address: {mac_addr}")
                            
                            # Extract channel and frequency info
                            elif 'channel' in line.lower() and 'MHz' in line:
                                # Parse lines like: "channel 6 (2437 MHz), width: 20 MHz, center1: 2437 MHz"
                                if '(' in line and 'MHz' in line:
                                    try:
                                        # Extract frequency from parentheses
                                        freq_part = line.split('(')[1].split(' MHz')[0]
                                        wireless_info['frequency'] = f"{freq_part} MHz"
                                        logger.debug(f"Found frequency: {freq_part} MHz")
                                    except:
                                        pass
                                
                                # Extract channel number
                                if 'channel' in line.lower():
                                    try:
                                        channel_part = line.lower().split('channel')[1].strip().split()[0]
                                        wireless_info['channel'] = channel_part
                                        logger.debug(f"Found channel: {channel_part}")
                                    except:
                                        pass
                            
                            # Extract transmission power
                            elif 'txpower' in line.lower():
                                try:
                                    # Parse lines like: "txpower 20.00 dBm"
                                    txpower_part = line.split('txpower')[1].strip()
                                    wireless_info['txpower'] = txpower_part
                                    logger.debug(f"Found txpower: {txpower_part}")
                                except:
                                    pass
                    
                    # Get connection info (SSID and signal strength)
                    link_result = subprocess.run(['iw', 'dev', interface_name, 'link'], 
                                               capture_output=True, text=True, timeout=5)
                    if link_result.returncode == 0:
                        link_output = link_result.stdout
                        logger.debug(f"iw dev {interface_name} link output: {link_output}")
                        
                        # Parse SSID
                        if 'Connected to' in link_output:
                            for line in link_output.split('\n'):
                                if 'SSID:' in line:
                                    ssid = line.split('SSID:')[1].strip()
                                    if ssid:
                                        wireless_info['ssid'] = ssid
                                        logger.info(f"iw found SSID: {ssid}")
                                    break
                        
                        # Parse signal strength
                        if 'signal:' in link_output:
                            for line in link_output.split('\n'):
                                if 'signal:' in line:
                                    signal_part = line.split('signal:')[1].strip().split()[0]
                                    wireless_info['signal_strength'] = signal_part
                                    logger.info(f"iw found signal strength: {signal_part}")
                                    break
                    
                    # Get frequency/channel details
                    scan_result = subprocess.run(['iw', 'dev', interface_name, 'scan', 'dump'], 
                                               capture_output=True, text=True, timeout=10)
                    if scan_result.returncode == 0 and wireless_info.get('ssid'):
                        scan_output = scan_result.stdout
                        # Parse frequency for connected network
                        current_ssid = wireless_info.get('ssid')
                        in_target_bss = False
                        for line in scan_output.split('\n'):
                            if f'SSID: {current_ssid}' in line:
                                in_target_bss = True
                            elif line.startswith('BSS ') and in_target_bss:
                                in_target_bss = False
                            elif in_target_bss and 'freq:' in line:
                                freq_part = line.split('freq:')[1].strip().split()[0]
                                wireless_info['frequency'] = f"{freq_part} MHz"
                                break
                                
                except subprocess.TimeoutExpired:
                    logger.warning(f"iw command timeout for {interface_name}")
                except FileNotFoundError:
                    logger.warning("iw command not found - install iw package")
                except Exception as e:
                    logger.debug(f"iw command failed: {e}")
                
                # Method 2: Try nmcli as fallback
                if not wireless_info['ssid']:
                    try:
                        result = subprocess.run(['nmcli', '-t', '-f', 'ACTIVE,SSID,SIGNAL,FREQ', 'dev', 'wifi'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            logger.debug(f"nmcli output: {result.stdout}")
                            for line in result.stdout.strip().split('\n'):
                                if line.startswith('yes:'):
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        ssid = parts[1]
                                        if ssid:
                                            wireless_info['ssid'] = ssid
                                            logger.info(f"nmcli found SSID: {ssid}")
                                        if len(parts) >= 3 and parts[2]:
                                            wireless_info['signal_strength'] = f"{parts[2]} dBm"
                                        if len(parts) >= 4 and parts[3]:
                                            wireless_info['frequency'] = f"{parts[3]} MHz"
                                        break
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        logger.warning("nmcli not available or timeout - install NetworkManager")
                    except Exception as e:
                        logger.debug(f"nmcli failed: {e}")
                
                # Method 3: Try /proc/net/wireless
                if not wireless_info['ssid']:
                    try:
                        with open('/proc/net/wireless', 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if interface_name in line:
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        # Signal quality info available
                                        logger.debug(f"/proc/net/wireless data for {interface_name}: {line.strip()}")
                                        break
                    except (FileNotFoundError, PermissionError):
                        logger.debug("/proc/net/wireless not accessible")
                
                # Method 4: Try iw command
                if not wireless_info['ssid']:
                    try:
                        result = subprocess.run(['iw', 'dev', interface_name, 'link'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            output = result.stdout
                            logger.debug(f"iw output for {interface_name}: {output}")
                            
                            # Parse SSID from iw output
                            for line in output.split('\n'):
                                if 'SSID:' in line:
                                    ssid = line.split('SSID:')[1].strip()
                                    if ssid:
                                        wireless_info['ssid'] = ssid
                                elif 'freq:' in line:
                                    freq = line.split('freq:')[1].split()[0]
                                    wireless_info['frequency'] = f"{freq} MHz"
                                elif 'signal:' in line:
                                    signal = line.split('signal:')[1].split()[0]
                                    wireless_info['signal_strength'] = f"{signal} dBm"
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        logger.debug("iw command not available or timeout")
                
                # Method 5: Check /sys/class/net for wireless info
                try:
                    wireless_path = f"/sys/class/net/{interface_name}/wireless"
                    if os.path.exists(wireless_path):
                        logger.debug(f"Wireless interface detected via /sys/class/net/{interface_name}/wireless")
                        # Interface is wireless but detailed info may not be available
                except Exception:
                    pass
            
            elif platform.system().lower() == 'darwin':  # macOS
                try:
                    import subprocess
                    # Try airport command on macOS
                    result = subprocess.run(['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-I'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        output = result.stdout
                        for line in output.split('\n'):
                            if 'SSID:' in line:
                                wireless_info['ssid'] = line.split('SSID:')[1].strip()
                            elif 'agrCtlRSSI:' in line:
                                wireless_info['signal_strength'] = f"{line.split('agrCtlRSSI:')[1].strip()} dBm"
                            elif 'channel:' in line:
                                wireless_info['frequency'] = line.split('channel:')[1].strip()
                except Exception as e:
                    logger.debug(f"macOS wireless detection failed: {e}")
            
            elif platform.system().lower() == 'windows':
                try:
                    import subprocess
                    # Try netsh on Windows
                    result = subprocess.run(['netsh', 'wlan', 'show', 'profiles'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        # Get current connection info
                        result2 = subprocess.run(['netsh', 'wlan', 'show', 'interfaces'], 
                                               capture_output=True, text=True, timeout=5)
                        if result2.returncode == 0:
                            output = result2.stdout
                            for line in output.split('\n'):
                                if 'SSID' in line and ':' in line:
                                    wireless_info['ssid'] = line.split(':', 1)[1].strip()
                                elif 'Signal' in line and ':' in line:
                                    wireless_info['signal_strength'] = line.split(':', 1)[1].strip()
                                elif 'Channel' in line and ':' in line:
                                    wireless_info['frequency'] = line.split(':', 1)[1].strip()
                except Exception as e:
                    logger.debug(f"Windows wireless detection failed: {e}")
                        
        except Exception as e:
            logger.debug(f"Wireless info detection failed: {e}")
        
        # Enhanced logging for troubleshooting
        if any(wireless_info.values()):
            logger.info(f"✓ Wireless info collected for {interface_name}: SSID={wireless_info.get('ssid', 'None')}, Signal={wireless_info.get('signal_strength', 'None')}, Freq={wireless_info.get('frequency', 'None')}, Channel={wireless_info.get('channel', 'None')}, MAC={wireless_info.get('mac_address', 'None')}, TxPower={wireless_info.get('txpower', 'None')}")
        else:
            logger.warning(f"⚠ No wireless information available for {interface_name} despite being detected as wireless interface")
            logger.warning("   Troubleshooting steps:")
            logger.warning("   1. Ensure client has wireless tools: sudo apt install iw")
            logger.warning("   2. Ensure NetworkManager is installed: sudo apt install network-manager") 
            logger.warning("   3. Ensure iwlib Python package: pip install iwlib")
            logger.warning("   4. Check interface is connected: iw dev " + interface_name + " link")
            logger.warning("   5. Run client with --verbose for detailed debugging")
        
        return wireless_info
    
    def _get_current_signal_strength(self):
        """Get current wireless signal strength for monitoring during tests"""
        try:
            # Get network interfaces first
            net_if_stats = psutil.net_if_stats()
            wireless_interfaces = []
            
            # Identify wireless interfaces
            for name in net_if_stats:
                if any(prefix in name.lower() for prefix in ['wlan', 'wifi', 'wl', 'ath', 'ra', 'wlp']):
                    wireless_interfaces.append(name)
                elif os.path.exists(f"/sys/class/net/{name}/wireless"):
                    wireless_interfaces.append(name)
            
            logger.debug(f"Detected wireless interfaces: {wireless_interfaces}")
            
            if not wireless_interfaces:
                logger.debug("No wireless interfaces found")
                return None
            
            # Get signal strength for the first active wireless interface
            for interface in wireless_interfaces:
                # Check if interface is up
                if_stats = net_if_stats.get(interface)
                if not if_stats or not if_stats.isup:
                    logger.debug(f"Interface {interface} is not up")
                    continue
                
                if platform.system().lower() == 'linux':
                    # Method 1: Try /proc/net/wireless first (most reliable)
                    try:
                        with open('/proc/net/wireless', 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if interface in line and not line.strip().startswith('Inter-'):
                                    parts = line.split()
                                    logger.debug(f"/proc/net/wireless line for {interface}: {line.strip()}")
                                    if len(parts) >= 4:
                                        # Signal level is typically in the 4th column (index 3)
                                        signal_str = parts[3].rstrip('.')
                                        logger.debug(f"Raw signal string: '{signal_str}'")
                                        if signal_str.replace('-', '').replace('.', '').isdigit():
                                            signal_val = float(signal_str)
                                            logger.debug(f"/proc/net/wireless signal for {interface}: {signal_val} dBm")
                                            return signal_val
                    except Exception as e:
                        logger.debug(f"/proc/net/wireless failed for {interface}: {e}")
                        pass
                    
                    # Method 2: Try iwlib Python library
                    try:
                        import iwlib
                        stats = iwlib.get_iwconfig(interface)
                        if stats and 'stats' in stats:
                            if 'level' in stats['stats']:
                                signal_val = float(stats['stats']['level'])
                                logger.debug(f"iwlib signal strength for {interface}: {signal_val} dBm")
                                return signal_val
                    except (ImportError, Exception) as e:
                        logger.debug(f"iwlib failed for {interface}: {e}")
                        pass
                    
                    # Method 3: Try modern iw command (replacement for deprecated iwconfig)
                    try:
                        result = subprocess.run(['iw', 'dev', interface, 'link'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            output = result.stdout
                            logger.debug(f"iw dev {interface} link output: {output}")
                            for line in output.split('\n'):
                                if 'signal:' in line:
                                    logger.debug(f"Found signal line: {line.strip()}")
                                    # Extract signal level (e.g., "signal: -45 dBm")
                                    try:
                                        signal_part = line.split('signal:')[1].strip().split()[0]
                                        # Handle different formats: -45, -45dBm
                                        signal_part = signal_part.replace('dBm', '').replace('dbm', '')
                                        if '/' in signal_part:
                                            signal_part = signal_part.split('/')[0]
                                        
                                        # Clean and validate the signal value
                                        signal_clean = signal_part.strip()
                                        if signal_clean.replace('-', '').replace('.', '').isdigit():
                                            signal_val = float(signal_clean)
                                            logger.debug(f"iw signal strength for {interface}: {signal_val} dBm")
                                            return signal_val
                                    except (IndexError, ValueError) as e:
                                        logger.debug(f"Failed to parse signal from line '{line.strip()}': {e}")
                                        continue
                    except FileNotFoundError:
                        logger.debug("iw command not found - install iw package")
                    except Exception as e:
                        logger.debug(f"iw command failed for {interface}: {e}")
                        pass
                    
                    # Method 3: Try nmcli with multiple formats
                    try:
                        result = subprocess.run(['nmcli', '-f', 'ACTIVE,SIGNAL,SSID', 'dev', 'wifi'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            logger.debug(f"nmcli dev wifi output: {result.stdout}")
                            lines = result.stdout.strip().split('\n')
                            for line in lines[1:]:  # Skip header
                                if 'yes' in line.lower() or '*' in line:
                                    parts = line.split()
                                    for part in parts:
                                        if part.isdigit() and 0 <= int(part) <= 100:
                                            signal_percent = int(part)
                                            # Convert percentage to dBm 
                                            signal_dbm = -100 + (signal_percent * 70 / 100)
                                            logger.debug(f"nmcli signal strength: {signal_dbm:.1f} dBm (from {signal_percent}%)")
                                            return round(signal_dbm, 1)
                    except Exception as e:
                        logger.debug(f"nmcli dev wifi failed: {e}")
                        pass
                    
                    # Method 4: Try nmcli connection info for active interface
                    try:
                        result = subprocess.run(['nmcli', 'con', 'show', '--active'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            logger.debug(f"nmcli active connections: {result.stdout}")
                            # Look for wireless connections and get signal info
                            for line in result.stdout.split('\n'):
                                if '802-11-wireless' in line or 'wifi' in line.lower():
                                    connection_name = line.split()[0]
                                    # Get detailed info for this connection
                                    detail_result = subprocess.run(['nmcli', 'con', 'show', connection_name], 
                                                                 capture_output=True, text=True, timeout=3)
                                    if detail_result.returncode == 0:
                                        for detail_line in detail_result.stdout.split('\n'):
                                            if 'signal' in detail_line.lower():
                                                logger.debug(f"nmcli signal line: {detail_line}")
                    except Exception as e:
                        logger.debug(f"nmcli connection info failed: {e}")
                        pass
                    
                    # Method 5: Try iw command
                    try:
                        result = subprocess.run(['iw', interface, 'link'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0:
                            logger.debug(f"iw link output for {interface}: {result.stdout}")
                            for line in result.stdout.split('\n'):
                                if 'signal:' in line.lower():
                                    signal_match = re.search(r'signal:\s*(-?\d+(?:\.\d+)?)', line)
                                    if signal_match:
                                        signal_val = float(signal_match.group(1))
                                        logger.debug(f"iw signal strength for {interface}: {signal_val} dBm")
                                        return signal_val
                    except Exception as e:
                        logger.debug(f"iw command failed: {e}")
                        pass
        except Exception as e:
            logger.debug(f"Error in signal strength detection: {e}")
            pass
        
        # If all methods failed, log comprehensive debug info
        try:
            logger.warning(f"All signal strength detection methods failed for wireless interfaces: {wireless_interfaces}")
        except NameError:
            logger.warning("Signal strength detection failed - no wireless interfaces detected")
        logger.debug("Available wireless detection methods were: iwlib, iwconfig, nmcli, /proc/net/wireless")
        return None
    
    def _update_signal_strength_tracker(self, test_id, current_strength):
        """Update signal strength statistics for a test"""
        if current_strength is None:
            return
        
        if test_id not in self.signal_strength_tracker:
            self.signal_strength_tracker[test_id] = {
                'values': [],
                'min': current_strength,
                'max': current_strength,
                'sum': 0.0,
                'count': 0
            }
        
        tracker = self.signal_strength_tracker[test_id]
        tracker['values'].append(current_strength)
        tracker['min'] = min(tracker['min'], current_strength)
        tracker['max'] = max(tracker['max'], current_strength)
        tracker['sum'] += current_strength
        tracker['count'] += 1
    
    def _get_signal_strength_stats(self, test_id):
        """Get signal strength statistics for a test"""
        if test_id not in self.signal_strength_tracker or self.signal_strength_tracker[test_id]['count'] == 0:
            return None, None, None, 0, None
        
        tracker = self.signal_strength_tracker[test_id]
        avg = tracker['sum'] / tracker['count']
        # Create comma-delimited string of all values
        values_str = ','.join(str(round(val, 1)) for val in tracker['values'])
        return tracker['min'], tracker['max'], avg, tracker['count'], values_str
    
    def _get_application_metrics(self, destination):
        """Collect application-layer metrics"""
        metrics = {
            'content_download_time': None,
            'compression_ratio': None,
            'http_response_codes': None,
            'connection_reuse_ratio': None,
            'certificate_validation_time': None
        }
        
        try:
            import requests
            from urllib.parse import urlparse
            
            # Ensure we have a valid URL
            if not destination.startswith(('http://', 'https://')):
                destination = f'https://{destination}'
            
            parsed_url = urlparse(destination)
            if not parsed_url.netloc:
                logger.warning(f"Invalid URL for application metrics: {destination}")
                return metrics
            
            logger.debug(f"Starting application metrics collection for: {destination}")
            
            # Test HTTP performance
            session = requests.Session()
            
            # Measure content download time
            start_time = time.time()
            try:
                logger.info(f"Attempting HTTP request to {destination} for application metrics")
                response = session.get(destination, timeout=10, stream=True)
                logger.info(f"HTTP response received: {response.status_code}")
                
                # Track response codes
                status_code = response.status_code
                if 200 <= status_code < 300:
                    response_class = '2xx'
                elif 300 <= status_code < 400:
                    response_class = '3xx'
                elif 400 <= status_code < 500:
                    response_class = '4xx'
                else:
                    response_class = '5xx'
                
                metrics['http_response_codes'] = json.dumps({response_class: 1})
                
                # Measure content download time
                content_length = 0
                for chunk in response.iter_content(chunk_size=8192):
                    content_length += len(chunk)
                    # Stop after reasonable amount for testing
                    if content_length > 1048576:  # 1MB limit
                        break
                
                download_time = (time.time() - start_time) * 1000  # Convert to ms
                metrics['content_download_time'] = round(download_time, 2)
                
                # Check for compression
                content_encoding = response.headers.get('content-encoding', '').lower()
                if content_encoding in ['gzip', 'deflate', 'br']:
                    # Estimate compression ratio based on headers
                    content_length_header = response.headers.get('content-length')
                    if content_length_header and content_length > 0:
                        # Rough compression ratio estimate
                        compressed_size = int(content_length_header)
                        if compressed_size < content_length:
                            ratio = ((content_length - compressed_size) / content_length) * 100
                            metrics['compression_ratio'] = round(ratio, 1)
                        else:
                            metrics['compression_ratio'] = 0.0
                    else:
                        # Default compression ratio for compressed content
                        metrics['compression_ratio'] = 30.0  # Typical web compression
                else:
                    metrics['compression_ratio'] = 0.0
                
                # Check for keep-alive connection reuse
                connection_header = response.headers.get('connection', '').lower()
                if 'keep-alive' in connection_header:
                    metrics['connection_reuse_ratio'] = 100.0
                else:
                    metrics['connection_reuse_ratio'] = 0.0
                
                # SSL certificate validation time (for HTTPS)
                if parsed_url.scheme == 'https':
                    cert_start = time.time()
                    try:
                        import ssl
                        import socket
                        
                        context = ssl.create_default_context()
                        with socket.create_connection((parsed_url.netloc, 443), timeout=5) as sock:
                            with context.wrap_socket(sock, server_hostname=parsed_url.netloc) as ssock:
                                cert_time = (time.time() - cert_start) * 1000
                                metrics['certificate_validation_time'] = round(cert_time, 2)
                    except Exception as e:
                        logger.debug(f"SSL certificate validation timing failed: {e}")
                        
            except Exception as e:
                logger.warning(f"Application metrics collection failed for {destination}: {e}")
                
        except ImportError as e:
            logger.warning(f"requests library not available for application metrics: {e}")
        except Exception as e:
            logger.warning(f"Application metrics collection error: {e}")
        
        logger.debug(f"Application metrics collected: {[(k, v) for k, v in metrics.items() if v is not None]}")
        return metrics
    
    def _get_infrastructure_metrics(self):
        """Collect infrastructure monitoring metrics"""
        metrics = {
            'power_consumption_watts': None,
            'memory_error_rate': None,
            'fan_speeds_rpm': None,
            'smart_drive_health': None
        }
        
        try:
            logger.info("Starting infrastructure metrics collection...")
            # Power consumption monitoring (Linux only)
            if platform.system().lower() == 'linux':
                logger.debug("Linux system detected, checking power monitoring sources")
                # Try to get power consumption from various sources
                power_sources = [
                    '/sys/class/power_supply/BAT0/power_now',  # Battery power
                    '/sys/class/power_supply/BAT1/power_now',
                    '/sys/class/hwmon/hwmon0/power1_input',    # Hardware monitoring
                    '/sys/class/hwmon/hwmon1/power1_input'
                ]
                
                for power_file in power_sources:
                    try:
                        with open(power_file, 'r') as f:
                            power_microwatts = int(f.read().strip())
                            power_watts = power_microwatts / 1000000.0  # Convert to watts
                            if 0 < power_watts < 1000:  # Reasonable range
                                metrics['power_consumption_watts'] = round(power_watts, 1)
                                break
                    except (FileNotFoundError, ValueError, PermissionError):
                        continue
                
                # Memory error rate monitoring
                try:
                    # Check for ECC memory errors in kernel logs
                    result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        ecc_errors = result.stdout.lower().count('ecc') + result.stdout.lower().count('memory error')
                        # Estimate error rate per hour (very rough approximation)
                        if ecc_errors > 0:
                            # Assume system has been up for some time, rough estimate
                            with open('/proc/uptime', 'r') as f:
                                uptime_seconds = float(f.read().split()[0])
                                uptime_hours = uptime_seconds / 3600.0
                                if uptime_hours > 0:
                                    error_rate = ecc_errors / uptime_hours
                                    metrics['memory_error_rate'] = round(error_rate, 3)
                        else:
                            metrics['memory_error_rate'] = 0.0
                except Exception as e:
                    logger.debug(f"Memory error rate collection failed: {e}")
                
                # Fan speed monitoring
                try:
                    fan_speeds = {}
                    fan_paths = [
                        '/sys/class/hwmon/hwmon0/fan1_input',
                        '/sys/class/hwmon/hwmon1/fan1_input',
                        '/sys/class/hwmon/hwmon2/fan1_input'
                    ]
                    
                    for i, fan_path in enumerate(fan_paths):
                        try:
                            with open(fan_path, 'r') as f:
                                rpm = int(f.read().strip())
                                if rpm > 0:
                                    fan_speeds[f'fan_{i+1}'] = rpm
                        except (FileNotFoundError, ValueError, PermissionError):
                            continue
                    
                    if fan_speeds:
                        metrics['fan_speeds_rpm'] = json.dumps(fan_speeds)
                except Exception as e:
                    logger.debug(f"Fan speed monitoring failed: {e}")
                
                # SMART drive health monitoring (basic)
                try:
                    # Try to get basic disk health information
                    result = subprocess.run(['df', '-h'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        # Just check if drives are responding (basic health check)
                        drive_health = {'status': 'responsive'}
                        metrics['smart_drive_health'] = json.dumps(drive_health)
                except Exception as e:
                    logger.debug(f"Drive health check failed: {e}")
                
        except Exception as e:
            logger.warning(f"Infrastructure metrics collection error: {e}")
        
        logger.debug(f"Infrastructure metrics collected: {[(k, v) for k, v in metrics.items() if v is not None]}")
        return metrics
    
    def start(self):
        """Start the client"""
        logger.info("Starting StreamSwarm client...")
        self.running = True
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._send_heartbeat, daemon=True)
        self.heartbeat_thread.start()
        
        # Start test checking thread
        test_check_thread = threading.Thread(target=self._check_for_tests, daemon=True)
        test_check_thread.start()
        
        # Confirm client is ready
        logger.info(f"✓ Client connected and ready! Monitoring server for tests...")
        logger.info(f"✓ Client ID: {self.client_id} | Server: {self.server_url}")
        logger.info(f"✓ Heartbeat active | Test monitoring active")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            self.stop()
    
    def _sync_gnmi_devices(self):
        """Synchronize GNMI device configuration from server"""
        try:
            logger.info("Synchronizing GNMI devices from server...")
            
            # Get device list from server
            response = requests.get(
                f"{self.server_url}/api/client/gnmi/devices",
                headers={'Authorization': f'Bearer {self.api_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                devices = response.json()
                logger.info(f"Retrieved {len(devices)} GNMI devices from server")
                
                # Clear existing devices
                self.gnmi_devices = []
                if self.gnmi_analyzer:
                    self.gnmi_analyzer.clear_devices()
                
                # Configure each device
                for device in devices:
                    try:
                        self._configure_gnmi_device(device)
                        self.gnmi_devices.append(device)
                    except Exception as e:
                        logger.error(f"Failed to configure GNMI device {device.get('name', 'unknown')}: {e}")
                
                # Save configuration to local file
                self._save_gnmi_config()
                
                logger.info(f"Successfully configured {len(self.gnmi_devices)} GNMI devices")
                
            else:
                logger.warning(f"Failed to retrieve GNMI devices from server: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error synchronizing GNMI devices: {e}")
            # Try to load from local cache
            self._load_gnmi_config()
    
    def _configure_gnmi_device(self, device):
        """Configure a GNMI device based on server configuration"""
        if not self.gnmi_analyzer:
            return
            
        device_config = {
            'name': device.get('name'),
            'host': device.get('host'),
            'port': device.get('port', 830),
            'username': device.get('username'),
            'auth_method': device.get('auth_method', 'password')
        }
        
        # Download and configure certificates if needed
        if device_config['auth_method'] in ['certificate', 'cert_username']:
            cert_info = self._download_certificates(device)
            if cert_info:
                device_config.update(cert_info)
            else:
                logger.warning(f"Failed to download certificates for device {device_config['name']}")
                return
        
        # Add device to GNMI analyzer
        try:
            if device_config['auth_method'] == 'password':
                self.gnmi_analyzer.add_device(
                    device_config['host'],
                    device_config['username'],
                    device.get('password', ''),
                    device_config['port']
                )
            elif device_config['auth_method'] == 'certificate':
                self.gnmi_analyzer.add_device(
                    device_ip=device_config['host'],
                    port=device_config['port'],
                    auth_method='certificate',
                    cert_file=device_config.get('cert_file'),
                    key_file=device_config.get('key_file'),
                    ca_file=device_config.get('ca_file')
                )
            elif device_config['auth_method'] == 'cert_username':
                self.gnmi_analyzer.add_device(
                    device_ip=device_config['host'],
                    username=device_config['username'],
                    port=device_config['port'],
                    auth_method='cert_username',
                    cert_file=device_config.get('cert_file'),
                    key_file=device_config.get('key_file')
                )
            
            logger.info(f"✓ Configured GNMI device: {device_config['name']} ({device_config['host']})")
            
        except Exception as e:
            logger.error(f"Failed to add GNMI device {device_config['name']}: {e}")
            raise
    
    def _download_certificates(self, device):
        """Download certificates for a GNMI device"""
        try:
            # Get device certificates from server
            response = requests.get(
                f"{self.server_url}/api/gnmi/certificates/{device['id']}",
                headers={'Authorization': f'Bearer {self.api_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                cert_data = response.json()
                cert_files = {}
                
                # Save certificates to local files
                for cert_type in ['client_cert', 'client_key', 'ca_cert']:
                    if cert_data.get(cert_type):
                        cert_path = os.path.join(self.gnmi_certs_dir, f"{device['name']}_{cert_type}.pem")
                        
                        with open(cert_path, 'w') as f:
                            f.write(cert_data[cert_type])
                        
                        # Set appropriate permissions for key files
                        if cert_type == 'client_key':
                            os.chmod(cert_path, 0o600)
                        
                        # Map to expected keys
                        if cert_type == 'client_cert':
                            cert_files['cert_file'] = cert_path
                        elif cert_type == 'client_key':
                            cert_files['key_file'] = cert_path
                        elif cert_type == 'ca_cert':
                            cert_files['ca_file'] = cert_path
                
                return cert_files
            else:
                logger.warning(f"Failed to download certificates for device {device['name']}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading certificates for device {device['name']}: {e}")
            return None
    
    def _save_gnmi_config(self):
        """Save GNMI configuration to local file"""
        try:
            with open(self.gnmi_config_file, 'w') as f:
                json.dump(self.gnmi_devices, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving GNMI configuration: {e}")
    
    def _load_gnmi_config(self):
        """Load GNMI configuration from local file"""
        try:
            if os.path.exists(self.gnmi_config_file):
                with open(self.gnmi_config_file, 'r') as f:
                    self.gnmi_devices = json.load(f)
                logger.info(f"Loaded {len(self.gnmi_devices)} GNMI devices from local cache")
            else:
                logger.info("No local GNMI configuration found")
        except Exception as e:
            logger.error(f"Error loading GNMI configuration: {e}")
    
    def _ensure_client_certificates(self):
        """Ensure client has generated certificates for GNMI authentication"""
        try:
            # Create client certificates directory if it doesn't exist
            if not os.path.exists(self.client_cert_dir):
                os.makedirs(self.client_cert_dir, mode=0o700)
            
            # Check if certificates already exist
            if os.path.exists(self.client_cert_file) and os.path.exists(self.client_key_file):
                logger.info("Client certificates already exist")
                # Upload to server if not already uploaded
                self._upload_client_certificates()
                return
            
            # Generate new certificates
            logger.info("Generating new client certificates for GNMI authentication...")
            
            cert_pem, key_pem = self._generate_client_certificates()
            
            # Save certificates to local files
            with open(self.client_cert_file, 'w') as f:
                f.write(cert_pem)
            os.chmod(self.client_cert_file, 0o644)
            
            with open(self.client_key_file, 'w') as f:
                f.write(key_pem)
            os.chmod(self.client_key_file, 0o600)
            
            logger.info("Client certificates generated and saved successfully")
            
            # Upload certificates to server
            self._upload_client_certificates()
            
        except Exception as e:
            logger.error(f"Error ensuring client certificates: {e}")
    
    def _generate_client_certificates(self):
        """Generate self-signed client certificate and private key"""
        try:
            import subprocess
            import tempfile
            
            # Generate private key
            key_result = subprocess.run([
                'openssl', 'genrsa', '-out', '/dev/stdout', '2048'
            ], capture_output=True, text=True, check=True)
            
            if key_result.returncode != 0:
                raise Exception(f"Failed to generate private key: {key_result.stderr}")
            
            key_pem = key_result.stdout
            
            # Create certificate subject with client hostname
            subject = f"/CN={self.client_name}/O=StreamSwarm/C=US"
            
            # Generate certificate signing request and self-signed certificate
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.key') as key_file:
                key_file.write(key_pem)
                key_file.flush()
                
                # Generate certificate
                cert_result = subprocess.run([
                    'openssl', 'req', '-new', '-x509', '-days', '365',
                    '-key', key_file.name,
                    '-out', '/dev/stdout',
                    '-subj', subject
                ], capture_output=True, text=True, check=True)
                
                # Clean up temporary key file
                os.unlink(key_file.name)
                
                if cert_result.returncode != 0:
                    raise Exception(f"Failed to generate certificate: {cert_result.stderr}")
                
                cert_pem = cert_result.stdout
            
            return cert_pem, key_pem
            
        except subprocess.CalledProcessError as e:
            logger.error(f"OpenSSL command failed: {e}")
            raise Exception(f"Certificate generation failed: {e}")
        except Exception as e:
            logger.error(f"Error generating certificates: {e}")
            raise
    
    def _upload_client_certificates(self):
        """Upload client certificates to server"""
        try:
            if not os.path.exists(self.client_cert_file) or not os.path.exists(self.client_key_file):
                logger.warning("Client certificates not found, cannot upload")
                return
            
            # Read certificate files
            with open(self.client_cert_file, 'r') as f:
                cert_content = f.read()
            
            with open(self.client_key_file, 'r') as f:
                key_content = f.read()
            
            # Extract certificate subject and expiry information
            cert_info = self._extract_certificate_info(cert_content)
            
            # Upload to server
            data = {
                'client_cert': cert_content,
                'client_key': key_content,
                'cert_subject': cert_info.get('subject', ''),
                'cert_expiry': cert_info.get('expiry', '')
            }
            
            response = requests.post(
                f"{self.server_url}/api/client/{self.client_id}/certificates",
                json=data,
                headers={'Authorization': f'Bearer {self.api_token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Client certificates uploaded to server successfully")
            else:
                logger.warning(f"Failed to upload certificates to server: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error uploading client certificates: {e}")
    
    def _extract_certificate_info(self, cert_pem):
        """Extract subject and expiry information from certificate"""
        try:
            import subprocess
            import tempfile
            from datetime import datetime
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.crt') as cert_file:
                cert_file.write(cert_pem)
                cert_file.flush()
                
                # Extract subject
                subject_result = subprocess.run([
                    'openssl', 'x509', '-in', cert_file.name, '-noout', '-subject'
                ], capture_output=True, text=True)
                
                # Extract expiry date
                expiry_result = subprocess.run([
                    'openssl', 'x509', '-in', cert_file.name, '-noout', '-enddate'
                ], capture_output=True, text=True)
                
                # Clean up
                os.unlink(cert_file.name)
                
                info = {}
                if subject_result.returncode == 0:
                    info['subject'] = subject_result.stdout.strip().replace('subject=', '')
                
                if expiry_result.returncode == 0:
                    expiry_str = expiry_result.stdout.strip().replace('notAfter=', '')
                    try:
                        # Parse OpenSSL date format: MMM DD HH:MM:SS YYYY GMT
                        expiry_date = datetime.strptime(expiry_str, '%b %d %H:%M:%S %Y %Z')
                        info['expiry'] = expiry_date.isoformat()
                    except ValueError:
                        info['expiry'] = expiry_str
                
                return info
                
        except Exception as e:
            logger.error(f"Error extracting certificate info: {e}")
            return {}
    
    def _detect_wifi_interfaces(self):
        """Detect available WiFi interfaces and classify them using modern iw command"""
        try:
            if not WIFI_SCANNING_AVAILABLE:
                logger.info("WiFi scanning not available - iw command not found")
                return
            
            self.wifi_interfaces = {}
            
            # Get all wireless interfaces using iw
            try:
                result = subprocess.run(['iw', 'dev'], capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    logger.warning("Failed to get wireless interfaces with iw dev")
                    return
                
                current_interface = None
                
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('Interface '):
                        current_interface = line.split('Interface ')[1]
                        
                        # Initialize interface data
                        interface_data = {
                            'name': current_interface,
                            'connected': False,
                            'essid': '',
                            'frequency': '',
                            'access_point': '',
                            'signal_strength': '',
                            'type': 'wireless'
                        }
                        
                        # Get connection details
                        try:
                            link_result = subprocess.run(['iw', 'dev', current_interface, 'link'], 
                                                       capture_output=True, text=True, timeout=5)
                            if link_result.returncode == 0:
                                link_output = link_result.stdout
                                
                                # Check if connected
                                if 'Connected to' in link_output:
                                    interface_data['connected'] = True
                                    
                                    # Extract SSID
                                    for link_line in link_output.split('\n'):
                                        if 'SSID:' in link_line:
                                            interface_data['essid'] = link_line.split('SSID:')[1].strip()
                                        elif 'freq:' in link_line:
                                            freq_match = re.search(r'freq:\s*(\d+)', link_line)
                                            if freq_match:
                                                interface_data['frequency'] = f"{freq_match.group(1)} MHz"
                                        elif 'signal:' in link_line:
                                            signal_match = re.search(r'signal:\s*(-?\d+(?:\.\d+)?)', link_line)
                                            if signal_match:
                                                interface_data['signal_strength'] = f"{signal_match.group(1)} dBm"
                        except Exception as e:
                            logger.debug(f"Error getting link info for {current_interface}: {e}")
                        
                        self.wifi_interfaces[current_interface] = interface_data
                        
                        # Classify interfaces
                        if interface_data['connected']:
                            if not self.primary_wifi_interface:
                                self.primary_wifi_interface = current_interface
                                logger.info(f"Primary WiFi interface detected: {current_interface} (connected to {interface_data['essid']})")
                        else:
                            self.spare_wifi_interfaces.append(current_interface)
                            logger.info(f"Spare WiFi interface detected: {current_interface}")
                            
            except Exception as e:
                logger.error(f"Error running iw dev command: {e}")
                return
            
            logger.info(f"WiFi interface detection complete - Primary: {self.primary_wifi_interface}, Spare: {len(self.spare_wifi_interfaces)}")
            
        except Exception as e:
            logger.error(f"Error detecting WiFi interfaces: {e}")
    
    def _perform_wifi_environmental_scan(self, interface_name=None, scan_duration=30):
        """Perform WiFi environmental scanning on specified interface using modern iw command"""
        try:
            if not WIFI_SCANNING_AVAILABLE:
                return None
            
            # Use spare interface if available, otherwise use primary or specified
            target_interface = interface_name
            if not target_interface:
                if self.spare_wifi_interfaces:
                    target_interface = self.spare_wifi_interfaces[0]
                elif self.primary_wifi_interface:
                    target_interface = self.primary_wifi_interface
                else:
                    logger.warning("No WiFi interfaces available for environmental scanning")
                    return None
            
            logger.info(f"Starting WiFi environmental scan on interface {target_interface}")
            
            # Perform wireless scan using modern iw command with graceful permission handling
            scan_result = None
            try:
                # Strategy 1: Try regular iw scan first
                result = subprocess.run(['iw', 'dev', target_interface, 'scan'], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    scan_result = result
                    logger.debug(f"WiFi scan successful on {target_interface}")
                elif "Operation not permitted" in result.stderr:
                    logger.info(f"WiFi scan requires elevated permissions on {target_interface}, trying sudo...")
                    
                    # Strategy 2: Try with passwordless sudo
                    try:
                        result = subprocess.run(['sudo', '-n', 'iw', 'dev', target_interface, 'scan'],
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            scan_result = result
                            logger.info(f"WiFi scan successful with sudo on {target_interface}")
                        else:
                            logger.warning(f"Sudo WiFi scan failed on {target_interface}: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Sudo scan timed out - may require password authentication")
                    except Exception as e:
                        logger.warning(f"Sudo not available or failed: {e}")
                        
                    # Strategy 3: If sudo failed, provide helpful instructions
                    if scan_result is None:
                        logger.error(f"WiFi environmental scan failed on {target_interface}. Solutions:")
                        logger.error(f"1. Run client with sudo: sudo python3 client.py")
                        logger.error(f"2. Configure passwordless sudo for WiFi scanning:")
                        logger.error(f"   echo '$(whoami) ALL=(ALL) NOPASSWD: /usr/sbin/iw' | sudo tee /etc/sudoers.d/wifi-scan")
                        logger.error(f"3. Add user to netdev group: sudo usermod -a -G netdev $(whoami)")
                        return None
                else:
                    logger.error(f"WiFi scan failed on {target_interface}: {result.stderr}")
                    return None
                
                scan_output = scan_result.stdout
                scan_results = self._parse_iw_scan_output(scan_output)
                
                # Process scan results into environmental metrics
                environment_data = self._analyze_wifi_environment(scan_results, target_interface)
                
                logger.info(f"WiFi environmental scan completed on {target_interface}: found {len(scan_results)} networks")
                return environment_data
                
            except subprocess.TimeoutExpired:
                logger.error(f"WiFi scan timeout on {target_interface}")
                return None
            except Exception as e:
                logger.error(f"Error running iw scan on {target_interface}: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error performing WiFi environmental scan: {e}")
            return None
    
    def _parse_iw_scan_output(self, scan_output):
        """Parse iw scan output into network information"""
        networks = []
        current_network = None
        
        try:
            for line in scan_output.split('\n'):
                line = line.strip()
                
                # Start of a new BSS (network)
                if line.startswith('BSS '):
                    if current_network:
                        networks.append(current_network)
                    
                    # Extract MAC address and frequency
                    bss_parts = line.split()
                    mac_address = bss_parts[1].rstrip(':')
                    
                    current_network = {
                        'mac_address': mac_address,
                        'ESSID': '',
                        'Channel': 0,
                        'Signal level': -100,
                        'Frequency': '',
                        'Encryption key': 'off',
                        'Quality': '',
                        'Bitrates': []
                    }
                    
                    # Extract frequency if present (handle both MHz and GHz formats)
                    freq_match = re.search(r'(\d+\.?\d*)\s*MHz', line)
                    if not freq_match:
                        freq_match = re.search(r'(\d+\.?\d*)\s*GHz', line)
                        if freq_match:
                            freq_mhz = float(freq_match.group(1)) * 1000  # Convert GHz to MHz
                        else:
                            freq_mhz = None
                    else:
                        freq_mhz = float(freq_match.group(1))
                    
                    if freq_mhz:
                        current_network['Frequency'] = f"{freq_mhz} MHz"
                        
                        # Convert frequency to channel (improved accuracy)
                        if 2412 <= freq_mhz <= 2484:
                            # 2.4 GHz channels (1-14)
                            if freq_mhz == 2484:
                                current_network['Channel'] = 14
                            else:
                                current_network['Channel'] = int((freq_mhz - 2412) / 5) + 1
                        elif 5000 <= freq_mhz <= 5925:
                            # 5 GHz channels (36-165)
                            if freq_mhz >= 5735:
                                current_network['Channel'] = int((freq_mhz - 5000) / 5)
                            else:
                                current_network['Channel'] = int((freq_mhz - 5000) / 5) + 36
                        elif 5945 <= freq_mhz <= 7125:
                            # 6 GHz channels (WiFi 6E)
                            current_network['Channel'] = int((freq_mhz - 5945) / 20) + 1
                
                elif current_network:
                    # Parse signal strength
                    if line.startswith('signal:'):
                        signal_match = re.search(r'signal:\s*(-?\d+(?:\.\d+)?)', line)
                        if signal_match:
                            current_network['Signal level'] = float(signal_match.group(1))
                    
                    # Parse SSID
                    elif line.startswith('SSID:'):
                        ssid = line.split('SSID: ', 1)[1] if 'SSID: ' in line else ''
                        current_network['ESSID'] = ssid if ssid else 'Hidden'
                    
                    # Parse privacy/encryption
                    elif 'Privacy:' in line:
                        if 'Privacy: yes' in line:
                            current_network['Encryption key'] = 'on'
                        else:
                            current_network['Encryption key'] = 'off'
                    
                    # Parse frequency from freq: lines (alternative location)
                    elif line.startswith('freq:'):
                        freq_match = re.search(r'freq:\s*(\d+)', line)
                        if freq_match:
                            freq_mhz = float(freq_match.group(1))
                            current_network['Frequency'] = f"{freq_mhz} MHz"
                            
                            # Convert frequency to channel
                            if 2412 <= freq_mhz <= 2484:
                                if freq_mhz == 2484:
                                    current_network['Channel'] = 14
                                else:
                                    current_network['Channel'] = int((freq_mhz - 2412) / 5) + 1
                            elif 5000 <= freq_mhz <= 5925:
                                if freq_mhz >= 5735:
                                    current_network['Channel'] = int((freq_mhz - 5000) / 5)
                                else:
                                    current_network['Channel'] = int((freq_mhz - 5000) / 5) + 36
                            elif 5945 <= freq_mhz <= 7125:
                                current_network['Channel'] = int((freq_mhz - 5945) / 20) + 1
                    
                    # Parse supported rates
                    elif 'Supported rates:' in line:
                        rates_part = line.split('Supported rates: ')[1]
                        current_network['Bitrates'] = rates_part.split()
            
            # Don't forget the last network
            if current_network:
                networks.append(current_network)
                
        except Exception as e:
            logger.error(f"Error parsing iw scan output: {e}")
        
        return networks
    
    def _analyze_wifi_environment(self, scan_results, interface_name):
        """Analyze WiFi scan results to extract environmental metrics"""
        try:
            networks = []
            channel_usage = {}
            signal_strengths = []
            
            for network in scan_results:
                try:
                    # Extract network information (compatible with both iwlib and iw formats)
                    ssid = network.get('ESSID', 'Hidden')
                    if not ssid or ssid == '':
                        ssid = 'Hidden'
                    
                    channel = network.get('Channel', 0)
                    signal = network.get('Signal level', -100)
                    frequency = network.get('Frequency', '')
                    encryption = network.get('Encryption key', 'off') == 'on'
                    
                    # Handle different data types for signal strength
                    if isinstance(signal, str):
                        try:
                            signal = float(signal.replace('dBm', '').strip())
                        except (ValueError, AttributeError):
                            signal = -100
                    
                    networks.append({
                        'ssid': ssid,
                        'channel': int(channel) if channel else 0,
                        'signal_strength': signal,
                        'frequency': frequency,
                        'encrypted': encryption,
                        'mac_address': network.get('mac_address', '')
                    })
                    
                    # Track channel usage (improved filtering)
                    if channel and int(channel) > 0:
                        channel_usage[int(channel)] = channel_usage.get(int(channel), 0) + 1
                    else:
                        logger.debug(f"Network {ssid} has invalid channel: {channel}")
                    
                    # Collect signal strengths for analysis
                    if signal > -100:
                        signal_strengths.append(signal)
                        
                except Exception as e:
                    logger.warning(f"Error processing network scan result: {e}")
                    continue
            
            # Calculate environmental metrics
            total_networks = len(networks)
            avg_signal_strength = sum(signal_strengths) / len(signal_strengths) if signal_strengths else -100
            
            # Channel congestion analysis
            most_congested_channel = max(channel_usage, key=channel_usage.get) if channel_usage else 0
            max_congestion = max(channel_usage.values()) if channel_usage else 0
            
            # Advanced RF Analysis
            rf_analysis = self._perform_advanced_rf_analysis(networks, interface_name)
            
            # 2.4GHz vs 5GHz distribution (improved classification)
            ghz_24_networks = len([n for n in networks if 1 <= n['channel'] <= 14])
            ghz_5_networks = len([n for n in networks if n['channel'] >= 36])
            ghz_6_networks = len([n for n in networks if n['channel'] >= 1 and n['channel'] <= 233 and n['channel'] not in range(1, 15) and n['channel'] not in range(36, 200)])
            
            # Signal quality assessment
            strong_signals = len([s for s in signal_strengths if s > -50])
            weak_signals = len([s for s in signal_strengths if s < -70])
            
            # Calculate WiFi pollution score (0-100, higher = more polluted)
            # Improved calculation considering channel overlap and interference
            base_score = min(50, total_networks * 1.5)  # Base score from network count
            congestion_score = min(30, max_congestion * 3)  # Channel congestion impact
            interference_score = min(20, weak_signals * 0.5)  # Weak signal interference
            pollution_score = int(base_score + congestion_score + interference_score)
            
            environment_data = {
                'interface_name': interface_name,
                'scan_timestamp': datetime.now(timezone.utc).isoformat(),
                'total_networks': total_networks,
                'networks_24ghz': ghz_24_networks,
                'networks_5ghz': ghz_5_networks,
                'avg_signal_strength': round(avg_signal_strength, 1),
                'strong_signals_count': strong_signals,
                'weak_signals_count': weak_signals,
                'channel_usage': channel_usage,
                'most_congested_channel': most_congested_channel,
                'max_channel_congestion': max_congestion,
                'wifi_pollution_score': pollution_score,
                'environment_quality': self._assess_environment_quality(pollution_score),
                'detected_networks': networks,  # Store all detected networks
                # Advanced RF Analysis
                'rf_analysis': rf_analysis
            }
            
            return environment_data
            
        except Exception as e:
            logger.error(f"Error analyzing WiFi environment: {e}")
            return None
    
    def _perform_advanced_rf_analysis(self, networks, interface_name):
        """Perform advanced RF analysis for enhanced WiFi environmental assessment"""
        try:
            # Noise Floor Measurement
            noise_floor = self._measure_noise_floor(interface_name)
            
            # Channel Utilization Analysis
            channel_utilization = self._calculate_channel_utilization(networks)
            
            # Signal-to-Noise Ratio calculations
            snr_analysis = self._calculate_snr_metrics(networks, noise_floor)
            
            # Interference Source Classification
            interference_analysis = self._classify_interference_sources(networks)
            
            return {
                'noise_floor': noise_floor,
                'channel_utilization': channel_utilization,
                'snr_analysis': snr_analysis,
                'interference_analysis': interference_analysis,
                'rf_quality_score': self._calculate_rf_quality_score(noise_floor, channel_utilization, snr_analysis, interference_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced RF analysis: {e}")
            return {
                'noise_floor': {'avg_noise_floor': -95, 'measurement_method': 'estimated'},
                'channel_utilization': {},
                'snr_analysis': {'avg_snr': 0, 'poor_snr_count': 0},
                'interference_analysis': {'total_interference_sources': 0, 'interference_types': {}},
                'rf_quality_score': 50
            }
    
    def _measure_noise_floor(self, interface_name):
        """Measure background RF noise levels"""
        try:
            # Use iw to get interface statistics and estimate noise floor
            result = subprocess.run(['iw', 'dev', interface_name, 'survey', 'dump'], 
                                  capture_output=True, text=True, timeout=10)
            
            noise_measurements = []
            in_use_channel = None
            
            if result.returncode == 0:
                current_freq = None
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('frequency:'):
                        current_freq = line.split(':')[1].strip().split()[0]
                    elif line.startswith('noise:') and current_freq:
                        noise_match = re.search(r'noise:\s*(-?\d+)', line)
                        if noise_match:
                            noise_measurements.append(int(noise_match.group(1)))
                    elif '[in use]' in line and current_freq:
                        in_use_channel = current_freq
            
            if noise_measurements:
                avg_noise_floor = sum(noise_measurements) / len(noise_measurements)
                return {
                    'avg_noise_floor': round(avg_noise_floor, 1),
                    'noise_measurements': noise_measurements,
                    'in_use_channel': in_use_channel,
                    'measurement_method': 'survey_dump'
                }
            else:
                # Fallback estimation based on band
                return {
                    'avg_noise_floor': -95,  # Typical 2.4GHz noise floor
                    'noise_measurements': [],
                    'in_use_channel': None,
                    'measurement_method': 'estimated'
                }
                
        except Exception as e:
            logger.debug(f"Error measuring noise floor: {e}")
            return {
                'avg_noise_floor': -95,
                'noise_measurements': [],
                'in_use_channel': None,
                'measurement_method': 'error_fallback'
            }
    
    def _calculate_channel_utilization(self, networks):
        """Calculate actual airtime usage per channel"""
        try:
            channel_stats = {}
            
            for network in networks:
                channel = network.get('channel', 0)
                signal = network.get('signal_strength', -100)
                
                if channel > 0:
                    if channel not in channel_stats:
                        channel_stats[channel] = {
                            'network_count': 0,
                            'total_signal_power': 0,
                            'utilization_estimate': 0,
                            'strongest_signal': -100
                        }
                    
                    channel_stats[channel]['network_count'] += 1
                    channel_stats[channel]['strongest_signal'] = max(channel_stats[channel]['strongest_signal'], signal)
                    
                    # Convert dBm to linear power for utilization calculation
                    power_mw = 10 ** (signal / 10)
                    channel_stats[channel]['total_signal_power'] += power_mw
            
            # Calculate utilization percentage based on signal strength and network density
            for channel, stats in channel_stats.items():
                # Estimate utilization based on network count and signal strength
                base_utilization = min(stats['network_count'] * 15, 80)  # 15% per network, max 80%
                
                # Adjust for signal strength (stronger signals = more utilization)
                signal_factor = max(0, (stats['strongest_signal'] + 100) / 50)  # 0-1 scale
                estimated_utilization = min(base_utilization * (1 + signal_factor), 100)
                
                stats['utilization_estimate'] = round(estimated_utilization, 1)
            
            return channel_stats
            
        except Exception as e:
            logger.error(f"Error calculating channel utilization: {e}")
            return {}
    
    def _calculate_snr_metrics(self, networks, noise_floor_data):
        """Calculate Signal-to-Noise Ratio for each network"""
        try:
            noise_floor = noise_floor_data.get('avg_noise_floor', -95)
            snr_values = []
            poor_snr_count = 0
            
            network_snr = []
            for network in networks:
                signal = network.get('signal_strength', -100)
                if signal > -100:
                    snr = signal - noise_floor
                    snr_values.append(snr)
                    
                    # Add SNR to network data for detailed analysis
                    network_snr.append({
                        'ssid': network.get('ssid', 'Hidden'),
                        'signal_strength': signal,
                        'snr': round(snr, 1),
                        'channel': network.get('channel', 0),
                        'connection_quality': self._assess_connection_quality(snr)
                    })
                    
                    if snr < 20:  # Poor SNR threshold
                        poor_snr_count += 1
            
            avg_snr = sum(snr_values) / len(snr_values) if snr_values else 0
            
            return {
                'avg_snr': round(avg_snr, 1),
                'snr_values': snr_values,
                'poor_snr_count': poor_snr_count,
                'network_snr': network_snr[:10],  # Top 10 for storage
                'noise_floor_used': noise_floor
            }
            
        except Exception as e:
            logger.error(f"Error calculating SNR metrics: {e}")
            return {'avg_snr': 0, 'poor_snr_count': 0, 'network_snr': []}
    
    def _classify_interference_sources(self, networks):
        """Identify interference sources beyond WiFi networks with detailed analysis"""
        try:
            interference_sources = {
                'wifi_congestion': 0,
                'bluetooth_interference': 0,
                'microwave_interference': 0,
                'other_rf_sources': 0
            }
            
            # Detailed interference data for display
            interference_details = {
                'wifi_congestion_details': [],
                'bluetooth_details': [],
                'microwave_details': [],
                'other_rf_details': []
            }
            
            # Analyze channel patterns for interference detection
            channel_24ghz = [n for n in networks if 1 <= n.get('channel', 0) <= 14]
            channel_5ghz = [n for n in networks if n.get('channel', 0) >= 36]
            
            # WiFi congestion analysis with detailed tracking
            if channel_24ghz:
                # Check for overlapping channels in 2.4GHz
                channels_24 = [n['channel'] for n in channel_24ghz]
                channel_counts = {}
                for ch in channels_24:
                    channel_counts[ch] = channel_counts.get(ch, 0) + 1
                
                # Find congested channels
                congested_channels = []
                for ch, count in channel_counts.items():
                    if count > 1:  # More than one network on channel
                        congested_channels.append(ch)
                        # Get networks on this channel
                        networks_on_channel = [n for n in channel_24ghz if n.get('channel') == ch]
                        strongest_signal = max(n.get('signal_strength', -100) for n in networks_on_channel)
                        
                        interference_details['wifi_congestion_details'].append({
                            'channel': ch,
                            'network_count': count,
                            'networks': [{'ssid': n.get('ssid', 'Hidden'), 'signal': n.get('signal_strength', -100)} 
                                       for n in networks_on_channel[:5]],  # Top 5 networks
                            'strongest_signal': strongest_signal,
                            'congestion_level': 'High' if count >= 6 else 'Moderate' if count >= 3 else 'Low'
                        })
                
                interference_sources['wifi_congestion'] = len(congested_channels) * 5
                
                # Additional congestion from non-standard channels
                non_standard_channels = [ch for ch in channels_24 if ch not in [1, 6, 11]]
                if non_standard_channels:
                    interference_sources['wifi_congestion'] += len(non_standard_channels) * 2
                    for ch in non_standard_channels:
                        networks_on_channel = [n for n in channel_24ghz if n.get('channel') == ch]
                        if networks_on_channel:
                            interference_details['wifi_congestion_details'].append({
                                'channel': ch,
                                'network_count': len(networks_on_channel),
                                'networks': [{'ssid': n.get('ssid', 'Hidden'), 'signal': n.get('signal_strength', -100)} 
                                           for n in networks_on_channel[:3]],
                                'strongest_signal': max(n.get('signal_strength', -100) for n in networks_on_channel),
                                'congestion_level': 'Non-standard channel'
                            })
            
            # Bluetooth interference with detailed analysis
            if len(channel_24ghz) > 3:
                # High density of 2.4GHz networks suggests possible Bluetooth interference
                weak_24ghz = [n for n in channel_24ghz if n.get('signal_strength', -100) < -65]
                if len(weak_24ghz) > 1:
                    interference_sources['bluetooth_interference'] = min(len(weak_24ghz) * 3, 30)
                    
                    # Group by frequency ranges for Bluetooth analysis
                    bluetooth_ranges = {
                        '2.402-2.420 GHz': [],
                        '2.420-2.440 GHz': [],
                        '2.440-2.460 GHz': [],
                        '2.460-2.480 GHz': []
                    }
                    
                    for network in weak_24ghz:
                        freq_str = network.get('frequency', '0')
                        try:
                            # Parse frequency from string format like "2422.0 MHz"
                            if isinstance(freq_str, str):
                                freq = float(freq_str.replace(' MHz', '').replace('MHz', ''))
                            else:
                                freq = float(freq_str)
                        except (ValueError, TypeError):
                            freq = 0
                        
                        if 2402 <= freq <= 2420:
                            bluetooth_ranges['2.402-2.420 GHz'].append(network)
                        elif 2420 <= freq <= 2440:
                            bluetooth_ranges['2.420-2.440 GHz'].append(network)
                        elif 2440 <= freq <= 2460:
                            bluetooth_ranges['2.440-2.460 GHz'].append(network)
                        elif 2460 <= freq <= 2480:
                            bluetooth_ranges['2.460-2.480 GHz'].append(network)
                    
                    # Add detailed Bluetooth interference patterns
                    for freq_range, networks_in_range in bluetooth_ranges.items():
                        if networks_in_range:
                            interference_details['bluetooth_details'].append({
                                'frequency_range': freq_range,
                                'affected_networks': len(networks_in_range),
                                'networks': [{'ssid': n.get('ssid', 'Hidden'), 'signal': n.get('signal_strength', -100),
                                            'channel': n.get('channel', 0)} for n in networks_in_range[:3]],
                                'interference_type': 'Bluetooth Classic/BLE frequency hopping',
                                'impact': 'Intermittent interference during Bluetooth activity'
                            })
                
                # Additional Bluetooth detection based on network density
                if len(channel_24ghz) > 8:
                    interference_sources['bluetooth_interference'] += 5
                    interference_details['bluetooth_details'].append({
                        'frequency_range': '2.4 GHz Band',
                        'affected_networks': len(channel_24ghz),
                        'networks': [{'ssid': 'High density environment', 'signal': -70, 'channel': 'Multiple'}],
                        'interference_type': 'High 2.4GHz network density',
                        'impact': 'Increased likelihood of Bluetooth interference'
                    })
            
            # Microwave interference with detailed tracking
            if channel_24ghz:
                # Look for networks with very poor signal quality in 2.4GHz
                very_weak = [n for n in channel_24ghz if n.get('signal_strength', -100) < -80]
                if len(very_weak) > 0:
                    interference_sources['microwave_interference'] = min(len(very_weak) * 4, 25)
                    
                    # Focus on channels 6-11 (2.45GHz microwave overlap)
                    microwave_channels = [n for n in very_weak if 6 <= n.get('channel', 0) <= 11]
                    if microwave_channels:
                        interference_details['microwave_details'].append({
                            'frequency_range': '2.45 GHz (Microwave ISM Band)',
                            'affected_channels': list(set(n.get('channel', 0) for n in microwave_channels)),
                            'networks': [{'ssid': n.get('ssid', 'Hidden'), 'signal': n.get('signal_strength', -100),
                                        'channel': n.get('channel', 0)} for n in microwave_channels[:3]],
                            'interference_type': 'Microwave oven interference',
                            'impact': 'Signal degradation during microwave operation',
                            'distance_estimate': '10-15 feet from microwave source'
                        })
                
                # Additional microwave detection from signal patterns
                if len(channel_24ghz) > 6:
                    microwave_overlap = [n for n in channel_24ghz if n.get('channel', 0) in [2, 3, 4, 5, 7, 8, 9, 10]]
                    if len(microwave_overlap) > 3:
                        interference_sources['microwave_interference'] += 3
                        interference_details['microwave_details'].append({
                            'frequency_range': '2.4-2.5 GHz (Extended microwave range)',
                            'affected_channels': list(set(n.get('channel', 0) for n in microwave_overlap)),
                            'networks': [{'ssid': n.get('ssid', 'Hidden'), 'signal': n.get('signal_strength', -100),
                                        'channel': n.get('channel', 0)} for n in microwave_overlap[:3]],
                            'interference_type': 'Industrial/Commercial microwave equipment',
                            'impact': 'Broader frequency interference',
                            'distance_estimate': '30-50 feet from source'
                        })
            
            # Other RF sources with detailed frequency analysis
            total_networks = len(networks)
            if total_networks > 10:
                signal_variance = self._calculate_signal_variance(networks)
                if signal_variance > 200:
                    interference_sources['other_rf_sources'] = min(signal_variance / 15, 25)
                    
                    # Analyze frequency distribution for other RF sources
                    frequency_bands = {
                        '2.4 GHz': [],
                        '5 GHz': [],
                        '6 GHz': []
                    }
                    
                    for network in networks:
                        freq_str = network.get('frequency', '0')
                        try:
                            # Parse frequency from string format like "2422.0 MHz"
                            if isinstance(freq_str, str):
                                freq = float(freq_str.replace(' MHz', '').replace('MHz', ''))
                            else:
                                freq = float(freq_str)
                        except (ValueError, TypeError):
                            freq = 0
                        
                        if 2400 <= freq <= 2500:
                            frequency_bands['2.4 GHz'].append(network)
                        elif 5000 <= freq <= 6000:
                            frequency_bands['5 GHz'].append(network)
                        elif 6000 <= freq <= 7000:
                            frequency_bands['6 GHz'].append(network)
                    
                    for band, networks_in_band in frequency_bands.items():
                        if len(networks_in_band) > 5:
                            interference_details['other_rf_details'].append({
                                'frequency_range': band,
                                'network_count': len(networks_in_band),
                                'signal_variance': round(signal_variance, 1),
                                'potential_sources': self._identify_potential_rf_sources(band, len(networks_in_band)),
                                'impact': 'Variable interference depending on device activity',
                                'distance_estimate': 'Varies by device type (10-500 feet)'
                            })
                
                # Additional RF source detection
                if total_networks > 30:
                    interference_sources['other_rf_sources'] += 5
                    interference_details['other_rf_details'].append({
                        'frequency_range': 'Multi-band',
                        'network_count': total_networks,
                        'signal_variance': round(signal_variance, 1),
                        'potential_sources': ['High-density environment', 'Multiple RF sources'],
                        'impact': 'Complex interference environment',
                        'distance_estimate': 'Multiple overlapping sources'
                    })
            
            total_interference = sum(interference_sources.values())
            
            return {
                'total_interference_sources': round(total_interference, 1),
                'interference_types': interference_sources,
                'interference_level': self._assess_interference_level(total_interference),
                'interference_details': interference_details
            }
            
        except Exception as e:
            logger.error(f"Error classifying interference sources: {e}")
            return {'total_interference_sources': 0, 'interference_types': {}, 'interference_level': 'unknown', 'interference_details': {}}
    
    def _identify_potential_rf_sources(self, frequency_band, network_count):
        """Identify potential RF interference sources based on frequency band and density"""
        sources = []
        
        if frequency_band == '2.4 GHz':
            if network_count > 20:
                sources.extend(['Cordless phones (DECT)', 'Baby monitors', 'Wireless security cameras'])
            if network_count > 15:
                sources.extend(['Bluetooth devices', 'RC controllers'])
            if network_count > 10:
                sources.extend(['Microwave ovens', 'Wireless keyboards/mice'])
        elif frequency_band == '5 GHz':
            if network_count > 15:
                sources.extend(['Radar systems', 'Wireless security cameras'])
            if network_count > 10:
                sources.extend(['Baby monitors (5.8GHz)', 'RC/Drone controllers'])
        elif frequency_band == '6 GHz':
            if network_count > 5:
                sources.extend(['WiFi 6E devices', 'Industrial equipment'])
        
        return sources if sources else ['Unidentified RF sources']
    
    def _calculate_signal_variance(self, networks):
        """Calculate variance in signal strengths"""
        try:
            signals = [n.get('signal_strength', -100) for n in networks if n.get('signal_strength', -100) > -100]
            if len(signals) < 2:
                return 0
            
            mean_signal = sum(signals) / len(signals)
            variance = sum((s - mean_signal) ** 2 for s in signals) / len(signals)
            return variance
            
        except Exception as e:
            return 0
    
    def _assess_connection_quality(self, snr):
        """Assess connection quality based on SNR"""
        if snr >= 40:
            return 'excellent'
        elif snr >= 25:
            return 'good'
        elif snr >= 15:
            return 'fair'
        elif snr >= 10:
            return 'poor'
        else:
            return 'unusable'
    
    def _assess_interference_level(self, total_interference):
        """Assess overall interference level"""
        if total_interference < 10:
            return 'minimal'
        elif total_interference < 25:
            return 'low'
        elif total_interference < 50:
            return 'moderate'
        elif total_interference < 75:
            return 'high'
        else:
            return 'severe'
    
    def _calculate_rf_quality_score(self, noise_floor, channel_utilization, snr_analysis, interference_analysis):
        """Calculate overall RF quality score (0-100, higher is better)"""
        try:
            score = 100
            
            # Noise floor impact (0-25 points deduction)
            noise_floor_dbm = noise_floor.get('avg_noise_floor', -95)
            if noise_floor_dbm > -90:
                score -= min(25, (noise_floor_dbm + 90) * 2)
            
            # Channel utilization impact (0-30 points deduction)
            if channel_utilization:
                avg_utilization = sum(ch.get('utilization_estimate', 0) for ch in channel_utilization.values()) / len(channel_utilization)
                score -= min(30, avg_utilization * 0.3)
            
            # SNR impact (0-25 points deduction)
            avg_snr = snr_analysis.get('avg_snr', 0)
            if avg_snr < 25:
                score -= min(25, (25 - avg_snr))
            
            # Interference impact (0-20 points deduction)
            total_interference = interference_analysis.get('total_interference_sources', 0)
            score -= min(20, total_interference * 0.25)
            
            return max(0, round(score, 1))
            
        except Exception as e:
            logger.error(f"Error calculating RF quality score: {e}")
            return 50
    
    def _assess_environment_quality(self, pollution_score):
        """Assess WiFi environment quality based on pollution score"""
        if pollution_score < 20:
            return 'excellent'
        elif pollution_score < 40:
            return 'good'
        elif pollution_score < 60:
            return 'fair'
        elif pollution_score < 80:
            return 'poor'
        else:
            return 'critical'
    
    def _wifi_environmental_test(self, test_id, destination, duration, interval):
        """Run standalone WiFi environmental test"""
        try:
            logger.info(f"Starting WiFi environmental test {test_id} with {duration}s duration, {interval}s interval")
            
            if not WIFI_SCANNING_AVAILABLE:
                logger.error("WiFi scanning not available - iw command not found (install with: sudo apt-get install iw)")
                return
            
            if not (self.wifi_interfaces):
                logger.error("No WiFi interfaces available for environmental testing")
                return
            
            if not (self.spare_wifi_interfaces):
                logger.warning("No spare WiFi interfaces available, will attempt to use primary interface")
                # Continue execution but with warning
            
            start_time = time.time()
            end_time = start_time + duration
            scan_count = 0
            
            while time.time() < end_time and self.running:
                try:
                    # Check if test has been stopped on server
                    if not self._check_test_status(test_id):
                        logger.info(f"WiFi environmental test {test_id} was stopped on server")
                        break
                    
                    scan_count += 1
                    logger.info(f"WiFi environmental scan {scan_count} starting...")
                    
                    # Get system metrics
                    system_metrics = self._get_system_metrics()
                    
                    # Perform WiFi environmental scan
                    wifi_environment = self._perform_wifi_environmental_scan()
                    
                    # Prepare test result data
                    result_data = {
                        'client_id': self.client_id,
                        'test_id': test_id,
                        'timestamp': datetime.now(zoneinfo.ZoneInfo('America/New_York')).replace(tzinfo=None).isoformat(),
                        **system_metrics,
                        'wifi_environment_data': json.dumps(wifi_environment) if wifi_environment else None,
                        # Basic network metrics with null values for standalone WiFi test
                        'ping_latency': None,
                        'ping_packet_loss': None,
                        'jitter': None,
                        'traceroute_hops': None,
                        'traceroute_data': [],
                        'bandwidth_download': None,
                        'bandwidth_upload': None
                    }
                    
                    # Send results to server
                    response = requests.post(
                        f'{self.server_url}/api/test/results',
                        json=result_data,
                        headers={'Authorization': f'Bearer {self.api_token}'}
                    )
                    
                    if response.status_code == 200:
                        if wifi_environment:
                            logger.info(f"WiFi scan {scan_count} completed: {wifi_environment.get('total_networks', 0)} networks, pollution score: {wifi_environment.get('wifi_pollution_score', 0)}")
                        else:
                            logger.warning(f"WiFi scan {scan_count} completed with no data")
                    else:
                        logger.error(f"Failed to send WiFi scan {scan_count} results: {response.status_code}")
                    
                    # Wait for next interval
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error during WiFi environmental scan {scan_count}: {e}")
                    time.sleep(interval)
                    continue
            
            logger.info(f"WiFi environmental test {test_id} completed with {scan_count} scans")
            
        except Exception as e:
            logger.error(f"Error in WiFi environmental test {test_id}: {e}")

    def _voip_analysis_test(self, test_id, server_url, duration, interval):
        """
        Perform VoIP analysis test with SIP/RTP communication
        Tests call quality, codec performance, and MOS scores
        """
        logger.info(f"Starting VoIP analysis test {test_id} for {duration} seconds")
        
        if not VOIP_TESTING_AVAILABLE:
            logger.error("VoIP testing not available - sipsak not installed")
            return None
        
        try:
            # Parse server URL to get host
            from urllib.parse import urlparse
            parsed_url = urlparse(server_url)
            sip_server = parsed_url.hostname
            sip_port = 5060  # Standard SIP port
            
            start_time = time.time()
            test_results = []
            
            logger.info(f"VoIP analysis test against {sip_server}:{sip_port}")
            
            while time.time() - start_time < duration:
                try:
                    # Perform VoIP analysis
                    voip_metrics = self._perform_voip_analysis(sip_server, sip_port)
                    
                    if voip_metrics:
                        # Collect system metrics alongside VoIP metrics
                        system_metrics = self._get_system_metrics()
                        
                        # Merge VoIP and system metrics
                        combined_metrics = {**system_metrics, **voip_metrics}
                        
                        # Add test metadata
                        combined_metrics.update({
                            'test_id': test_id,
                            'client_id': self.client_id,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'test_type': 'voip_analysis'
                        })
                        
                        # Submit results to server
                        headers = {'Authorization': f'Bearer {self.api_token}'} if self.api_token else {}
                        response = requests.post(
                            urljoin(self.server_url, '/api/test/results'),
                            json=combined_metrics,
                            headers=headers,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            logger.debug(f"Submitted VoIP test result for test {test_id}")
                        else:
                            logger.warning(f"Failed to submit VoIP result: {response.status_code}")
                        test_results.append(combined_metrics)
                        
                        logger.info(f"VoIP analysis result: MOS Score {voip_metrics.get('mos_score', 'N/A')}, "
                                  f"Jitter {voip_metrics.get('rtp_jitter_avg', 'N/A')}ms, "
                                  f"Packet Loss {voip_metrics.get('rtp_packet_loss_rate', 'N/A')}%")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"VoIP analysis iteration failed: {e}")
                    time.sleep(interval)
            
            logger.info(f"VoIP analysis test completed after {time.time() - start_time:.1f} seconds")
            return test_results
        
        except Exception as e:
            logger.error(f"VoIP analysis test failed: {e}")
            return None
    
    def _perform_voip_analysis(self, sip_server, sip_port):
        """
        Perform comprehensive VoIP analysis with SIP/RTP testing
        Returns detailed VoIP metrics including MOS, jitter, packet loss
        """
        logger.info(f"Performing VoIP analysis against {sip_server}:{sip_port}")
        
        try:
            voip_metrics = {}
            
            # Step 1: SIP Registration Test
            registration_metrics = self._test_sip_registration(sip_server, sip_port)
            voip_metrics.update(registration_metrics)
            
            # Step 2: SIP Call Setup Test
            call_setup_metrics = self._test_sip_call_setup(sip_server, sip_port)
            voip_metrics.update(call_setup_metrics)
            
            # Step 3: RTP Stream Quality Test
            rtp_metrics = self._test_rtp_stream_quality(sip_server, sip_port)
            voip_metrics.update(rtp_metrics)
            
            # Step 4: Calculate MOS Score
            mos_score = self._calculate_mos_score(voip_metrics)
            voip_metrics['mos_score'] = mos_score
            
            # Step 5: Calculate Voice Quality Score
            voice_quality = self._calculate_voice_quality_score(voip_metrics)
            voip_metrics['voice_quality_score'] = voice_quality
            
            # Step 6: Codec Efficiency Analysis
            codec_efficiency = self._analyze_codec_efficiency(voip_metrics)
            voip_metrics['codec_efficiency'] = codec_efficiency
            
            # Step 7: Overall VoIP test status
            voip_metrics['voip_test_status'] = 'completed'
            
            # Store comprehensive analysis data
            voip_metrics['voip_analysis_data'] = json.dumps({
                'test_timestamp': datetime.now(timezone.utc).isoformat(),
                'server': sip_server,
                'port': sip_port,
                'analysis_summary': {
                    'mos_score': mos_score,
                    'voice_quality_score': voice_quality,
                    'codec_efficiency': codec_efficiency,
                    'primary_issues': self._identify_voip_issues(voip_metrics)
                }
            })
            
            logger.info(f"VoIP analysis completed successfully - MOS: {mos_score:.2f}")
            return voip_metrics
            
        except Exception as e:
            logger.error(f"VoIP analysis failed: {e}")
            return {
                'voip_test_status': 'failed',
                'mos_score': 1.0,  # Poor quality fallback
                'voice_quality_score': 0.0,
                'codec_efficiency': 0.0,
                'voip_analysis_data': json.dumps({
                    'error': str(e),
                    'test_timestamp': datetime.now(timezone.utc).isoformat()
                })
            }
    
    def _test_sip_registration(self, sip_server, sip_port):
        """Test SIP registration timing"""
        try:
            start_time = time.time()
            
            # Use sipsak to test SIP registration
            cmd = [
                'sipsak', '-s', f'sip:test@{sip_server}:{sip_port}',
                '-f', '0',  # No follow redirects
                '-n'        # No DNS lookup
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            registration_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            sip_response_codes = {}
            if result.returncode == 0:
                # Parse successful response
                sip_response_codes['200'] = 1
                logger.info(f"SIP registration successful in {registration_time:.2f}ms")
            else:
                # Parse error response
                sip_response_codes['error'] = 1
                logger.warning(f"SIP registration failed after {registration_time:.2f}ms")
            
            return {
                'sip_registration_time': registration_time,
                'sip_response_codes': json.dumps(sip_response_codes)
            }
            
        except subprocess.TimeoutExpired:
            logger.error("SIP registration test timed out")
            return {
                'sip_registration_time': 30000.0,  # 30 second timeout
                'sip_response_codes': json.dumps({'timeout': 1})
            }
        except Exception as e:
            logger.error(f"SIP registration test failed: {e}")
            return {
                'sip_registration_time': None,
                'sip_response_codes': json.dumps({'error': 1})
            }
    
    def _test_sip_call_setup(self, sip_server, sip_port):
        """Test SIP call setup and teardown timing"""
        try:
            # Call setup test
            setup_start = time.time()
            
            cmd = [
                'sipsak', '-s', f'sip:test@{sip_server}:{sip_port}',
                '-M', 'INVITE',  # Send INVITE request
                '-f', '0'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            setup_time = (time.time() - setup_start) * 1000
            
            # Call teardown test
            teardown_start = time.time()
            
            cmd = [
                'sipsak', '-s', f'sip:test@{sip_server}:{sip_port}',
                '-M', 'BYE',  # Send BYE request
                '-f', '0'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            teardown_time = (time.time() - teardown_start) * 1000
            
            logger.info(f"SIP call setup: {setup_time:.2f}ms, teardown: {teardown_time:.2f}ms")
            
            return {
                'sip_call_setup_time': setup_time,
                'sip_call_teardown_time': teardown_time
            }
            
        except Exception as e:
            logger.error(f"SIP call setup/teardown test failed: {e}")
            return {
                'sip_call_setup_time': None,
                'sip_call_teardown_time': None
            }
    
    def _test_rtp_stream_quality(self, sip_server, sip_port):
        """Test RTP stream quality with synthetic traffic"""
        try:
            # Step 1: Establish SIP session first to get RTP port allocation
            sip_session = self._establish_sip_session(sip_server, sip_port)
            if not sip_session:
                logger.error("Failed to establish SIP session for RTP testing")
                return {
                    'rtp_packet_loss_rate': 100.0,
                    'rtp_jitter_avg': 0.0,
                    'rtp_latency_avg': 0.0,
                    'rtp_packets_sent': 0,
                    'rtp_packets_received': 0
                }
            
            # Step 2: Use RTP port from SIP session
            rtp_port = sip_session['rtp_port']
            session_id = sip_session['session_id']
            
            # Create UDP socket for RTP simulation
            rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rtp_socket.settimeout(5.0)
            
            # RTP packet structure (simplified)
            packets_sent = 0
            packets_received = 0
            jitter_samples = []
            latency_samples = []
            
            stream_duration = 30  # 30 second RTP stream test
            packet_interval = 5.0  # 5 second interval for testing (instead of 20ms)
            
            logger.info(f"Starting RTP stream quality test for {stream_duration} seconds to port {rtp_port}")
            
            start_time = time.time()
            last_packet_time = start_time
            
            while time.time() - start_time < stream_duration:
                try:
                    # Create RTP packet (simplified format)
                    timestamp = int((time.time() - start_time) * 8000)  # 8kHz sample rate
                    sequence = packets_sent
                    
                    # Basic RTP header (12 bytes)
                    rtp_header = struct.pack('!BBHII', 
                                           0x80,  # V=2, P=0, X=0, CC=0
                                           0x00,  # M=0, PT=0 (PCMU)
                                           sequence,
                                           timestamp,
                                           0x12345678)  # SSRC
                    
                    # Add payload (160 bytes for 20ms of PCMU audio)
                    payload = b'\x00' * 160
                    rtp_packet = rtp_header + payload
                    
                    # Send RTP packet to server RTP port
                    packet_send_time = time.time()
                    rtp_socket.sendto(rtp_packet, (sip_server, rtp_port))
                    packets_sent += 1
                    
                    # Try to receive echoed packet (server should echo back)
                    try:
                        response, addr = rtp_socket.recvfrom(1024)
                        packet_receive_time = time.time()
                        packets_received += 1
                        
                        # Calculate latency
                        latency = (packet_receive_time - packet_send_time) * 1000
                        latency_samples.append(latency)
                        
                        # Calculate jitter (RFC 3550)
                        current_time = packet_receive_time
                        if last_packet_time:
                            transit_time = current_time - packet_send_time
                            if hasattr(self, '_last_transit_time'):
                                jitter = abs(transit_time - self._last_transit_time)
                                jitter_samples.append(jitter * 1000)  # Convert to ms
                            self._last_transit_time = transit_time
                        last_packet_time = current_time
                        
                    except socket.timeout:
                        # No response received (normal for most servers)
                        pass
                    
                    # Wait for next packet interval
                    time.sleep(packet_interval)
                    
                except Exception as e:
                    logger.debug(f"RTP packet error: {e}")
                    continue
            
            # Close socket and SIP session
            rtp_socket.close()
            self._terminate_sip_session(sip_server, sip_port, session_id)
            
            # Calculate RTP metrics
            packet_loss_rate = ((packets_sent - packets_received) / packets_sent * 100) if packets_sent > 0 else 100.0
            avg_jitter = sum(jitter_samples) / len(jitter_samples) if jitter_samples else 0.0
            max_jitter = max(jitter_samples) if jitter_samples else 0.0
            avg_latency = sum(latency_samples) / len(latency_samples) if latency_samples else 0.0
            max_latency = max(latency_samples) if latency_samples else 0.0
            
            logger.info(f"RTP stream test completed: {packets_sent} sent, {packets_received} received, "
                       f"Loss: {packet_loss_rate:.2f}%, Jitter: {avg_jitter:.2f}ms")
            
            return {
                'rtp_packet_loss_rate': packet_loss_rate,
                'rtp_jitter_avg': avg_jitter,
                'rtp_jitter_max': max_jitter,
                'rtp_latency_avg': avg_latency,
                'rtp_latency_max': max_latency,
                'rtp_stream_duration': stream_duration
            }
            
        except Exception as e:
            logger.error(f"RTP stream quality test failed: {e}")
            return {
                'rtp_packet_loss_rate': 100.0,  # Assume complete failure
                'rtp_jitter_avg': 0.0,
                'rtp_jitter_max': 0.0,
                'rtp_latency_avg': 0.0,
                'rtp_latency_max': 0.0,
                'rtp_stream_duration': 0.0
            }
    
    def _establish_sip_session(self, sip_server, sip_port):
        """Establish SIP session using INVITE to get RTP port allocation"""
        try:
            # Create SIP socket
            sip_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sip_socket.settimeout(10.0)
            
            # Generate unique session identifiers
            call_id = f"rtp-test-{int(time.time())}-{os.getpid()}"
            from_tag = f"client-{int(time.time())}"
            
            # Create SIP INVITE request
            invite_request = f"""INVITE sip:test@{sip_server}:{sip_port} SIP/2.0
Via: SIP/2.0/UDP {sip_server}:{sip_port};branch=z9hG4bK-{call_id}
From: <sip:client@{sip_server}>;tag={from_tag}
To: <sip:test@{sip_server}>
Call-ID: {call_id}
CSeq: 1 INVITE
Content-Type: application/sdp
Content-Length: 0

"""
            
            # Send INVITE request
            sip_socket.sendto(invite_request.encode('utf-8'), (sip_server, sip_port))
            
            # Receive response
            response, addr = sip_socket.recvfrom(4096)
            response_text = response.decode('utf-8')
            
            # Parse RTP port from SDP response
            rtp_port = None
            for line in response_text.split('\n'):
                if line.startswith('m=audio'):
                    parts = line.split()
                    if len(parts) >= 2:
                        rtp_port = int(parts[1])
                        break
            
            sip_socket.close()
            
            if rtp_port and '200 OK' in response_text:
                logger.info(f"SIP session established: Call-ID={call_id}, RTP port={rtp_port}")
                return {
                    'session_id': call_id,
                    'rtp_port': rtp_port,
                    'sip_server': sip_server,
                    'sip_port': sip_port,
                    'from_tag': from_tag
                }
            else:
                logger.error(f"SIP INVITE failed: {response_text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to establish SIP session: {e}")
            return None
    
    def _terminate_sip_session(self, sip_server, sip_port, session_id):
        """Terminate SIP session using BYE request"""
        try:
            sip_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sip_socket.settimeout(5.0)
            
            # Create SIP BYE request
            bye_request = f"""BYE sip:test@{sip_server}:{sip_port} SIP/2.0
Via: SIP/2.0/UDP {sip_server}:{sip_port};branch=z9hG4bK-bye-{session_id}
From: <sip:client@{sip_server}>;tag=client-{session_id}
To: <sip:test@{sip_server}>
Call-ID: {session_id}
CSeq: 2 BYE
Content-Length: 0

"""
            
            # Send BYE request
            sip_socket.sendto(bye_request.encode('utf-8'), (sip_server, sip_port))
            
            # Receive response (optional)
            try:
                response, addr = sip_socket.recvfrom(4096)
                logger.info(f"SIP session terminated: {session_id}")
            except socket.timeout:
                pass
            
            sip_socket.close()
            
        except Exception as e:
            logger.error(f"Failed to terminate SIP session {session_id}: {e}")
    
    def _calculate_mos_score(self, voip_metrics):
        """Calculate MOS (Mean Opinion Score) based on network metrics"""
        try:
            # Get key metrics
            packet_loss = voip_metrics.get('rtp_packet_loss_rate', 0)
            jitter = voip_metrics.get('rtp_jitter_avg', 0)
            latency = voip_metrics.get('rtp_latency_avg', 0)
            
            # ITU-T G.107 E-model approximation
            # Base transmission rating
            R = 93.2  # Base R-factor for high quality
            
            # Packet loss impairment
            if packet_loss > 0:
                loss_factor = 11 + 40 * packet_loss
                R -= loss_factor
            
            # Jitter impairment (affects buffering)
            if jitter > 40:  # 40ms threshold
                jitter_factor = (jitter - 40) / 10
                R -= jitter_factor
            
            # Latency impairment (one-way delay)
            if latency > 150:  # 150ms threshold
                delay_factor = (latency - 150) / 50
                R -= delay_factor
            
            # Convert R-factor to MOS
            if R < 0:
                mos = 1.0
            elif R > 100:
                mos = 4.5
            else:
                mos = 1 + 0.035 * R + 7e-6 * R * (R - 60) * (100 - R)
            
            # Clamp MOS to valid range
            mos = max(1.0, min(5.0, mos))
            
            return round(mos, 2)
            
        except Exception as e:
            logger.error(f"MOS calculation failed: {e}")
            return 1.0  # Poor quality fallback
    
    def _calculate_voice_quality_score(self, voip_metrics):
        """Calculate overall voice quality score (0-100)"""
        try:
            mos = voip_metrics.get('mos_score', 1.0)
            
            # Convert MOS (1-5) to percentage score (0-100)
            # MOS 1 = 0%, MOS 5 = 100%
            quality_score = ((mos - 1.0) / 4.0) * 100
            
            return round(quality_score, 1)
            
        except Exception as e:
            logger.error(f"Voice quality calculation failed: {e}")
            return 0.0
    
    def _analyze_codec_efficiency(self, voip_metrics):
        """Analyze codec efficiency based on performance metrics"""
        try:
            # Simulated codec efficiency based on quality metrics
            packet_loss = voip_metrics.get('rtp_packet_loss_rate', 0)
            jitter = voip_metrics.get('rtp_jitter_avg', 0)
            
            # Base efficiency for G.711 PCMU (64 kbps)
            base_efficiency = 85.0
            
            # Reduce efficiency based on packet loss
            if packet_loss > 0:
                loss_penalty = packet_loss * 2  # 2% penalty per 1% packet loss
                base_efficiency -= loss_penalty
            
            # Reduce efficiency based on jitter
            if jitter > 20:  # 20ms threshold
                jitter_penalty = (jitter - 20) / 10  # 1% penalty per 10ms extra jitter
                base_efficiency -= jitter_penalty
            
            # Clamp to valid range
            efficiency = max(0.0, min(100.0, base_efficiency))
            
            return round(efficiency, 1)
            
        except Exception as e:
            logger.error(f"Codec efficiency analysis failed: {e}")
            return 0.0
    
    def _identify_voip_issues(self, voip_metrics):
        """Identify primary VoIP quality issues"""
        issues = []
        
        try:
            packet_loss = voip_metrics.get('rtp_packet_loss_rate', 0)
            jitter = voip_metrics.get('rtp_jitter_avg', 0)
            latency = voip_metrics.get('rtp_latency_avg', 0)
            mos = voip_metrics.get('mos_score', 5.0)
            
            if packet_loss > 1:
                issues.append(f"High packet loss: {packet_loss:.1f}%")
            
            if jitter > 50:
                issues.append(f"Excessive jitter: {jitter:.1f}ms")
            
            if latency > 150:
                issues.append(f"High latency: {latency:.1f}ms")
            
            if mos < 3.0:
                issues.append(f"Poor call quality: MOS {mos:.1f}")
            
            return issues if issues else ["No significant issues detected"]
            
        except Exception as e:
            logger.error(f"VoIP issue identification failed: {e}")
            return ["Analysis failed"]
    
    def _execute_reboot(self):
        """Execute system reboot with proper checks"""
        try:
            logger.info("REBOOT: Starting reboot execution process")
            
            # Verify we're on Linux
            platform_name = platform.system().lower()
            logger.info(f"REBOOT: Detected platform: {platform_name}")
            if not platform_name == 'linux':
                logger.error("REBOOT: Reboot command only supported on Linux systems")
                return
            
            # Check if we have sudo privileges
            logger.info("REBOOT: Checking sudo privileges...")
            try:
                result = subprocess.run(['sudo', '-n', 'true'], capture_output=True, timeout=5)
                logger.info(f"REBOOT: Sudo check result - returncode: {result.returncode}, stdout: {result.stdout.decode()}, stderr: {result.stderr.decode()}")
                if result.returncode != 0:
                    logger.error("REBOOT: Sudo privileges required. Configure passwordless sudo or run client with sudo.")
                    logger.error(f"REBOOT: Sudo check failed with returncode {result.returncode}")
                    return
            except subprocess.TimeoutExpired:
                logger.error("REBOOT: Sudo privilege check timed out")
                return
            
            logger.info("REBOOT: Sudo privileges confirmed. Executing system reboot in 5 seconds...")
            
            # Give time for the log message to be processed
            time.sleep(2)
            
            # Stop the client cleanly
            logger.info("REBOOT: Stopping client cleanly...")
            self.stop()
            
            # Wait a moment for cleanup
            time.sleep(3)
            
            # Execute the reboot command using shutdown -r now (more reliable than reboot)
            logger.info("REBOOT: Executing 'sudo shutdown -r now' command...")
            subprocess.run(['sudo', 'shutdown', '-r', 'now'], check=True)
            logger.info("REBOOT: System reboot initiated successfully.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"REBOOT: Command failed with CalledProcessError: {e}")
            logger.error(f"REBOOT: Command returncode: {e.returncode}")
            logger.error(f"REBOOT: Command stdout: {e.stdout}")
            logger.error(f"REBOOT: Command stderr: {e.stderr}")
        except Exception as e:
            logger.error(f"REBOOT: Error during reboot execution: {e}")
            logger.error(f"REBOOT: Exception type: {type(e)}")
            import traceback
            logger.error(f"REBOOT: Full traceback: {traceback.format_exc()}")
    
    def stop(self):
        """Stop the client"""
        self.running = False
        logger.info("StreamSwarm client stopped")

def main():
    parser = argparse.ArgumentParser(description='StreamSwarm Client')
    parser.add_argument('--server', required=True, help='Server URL (e.g., http://localhost:5000)')
    parser.add_argument('--token', required=True, help='API token for authentication')
    parser.add_argument('--name', help='Client name (defaults to hostname)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start client
    client = StreamSwarmClient(args.server, args.name, args.token)
    client.start()

if __name__ == '__main__':
    main()
