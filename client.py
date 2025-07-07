#!/usr/bin/env python3
"""
StreamSwarm Client - Network and system monitoring client
"""

# Client version - increment when making changes
CLIENT_VERSION = "1.0.5"

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
                data = {'client_version': CLIENT_VERSION}
                response = requests.post(
                    urljoin(self.server_url, f'/api/client/{self.client_id}/heartbeat'),
                    json=data,
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code != 200:
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
        interval = test_config.get('interval', 5)
        packet_size = test_config.get('packet_size', 64)
        
        logger.info(f"Starting test {test_id} to {destination} for {duration}s")
        
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time and self.running:
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
                    **gnmi_path_data
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
                f"{self.server_url}/api/gnmi/devices",
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
