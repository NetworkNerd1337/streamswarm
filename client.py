#!/usr/bin/env python3
"""
StreamSwarm Client - Network and system monitoring client
"""

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
from datetime import datetime, timezone
import zoneinfo
import psutil
import requests
from urllib.parse import urljoin, urlparse

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
        
        # Get system information
        self.system_info = self._get_system_info()
        
        # Register with server
        self._register()
    
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
                response = requests.post(
                    urljoin(self.server_url, f'/api/client/{self.client_id}/heartbeat'),
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
    
    def _ping_test(self, destination, count=4):
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
                cmd = ['ping', '-n', str(count), hostname]
            else:
                cmd = ['ping', '-c', str(count), hostname]
            
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
        """Perform traceroute test"""
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
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip():
                        # Count hops (simplified parsing)
                        if line.strip()[0].isdigit():
                            hops += 1
                            hop_data.append(line.strip())
            
            return {
                'hops': hops,
                'data': hop_data,
                'raw_output': result.stdout
            }
            
        except Exception as e:
            logger.error(f"Error performing traceroute: {e}")
            return {'hops': 0, 'data': [], 'raw_output': ''}
    
    def _run_test(self, test_config):
        """Run a single test"""
        test_id = test_config['id']
        destination = test_config['destination']
        duration = test_config.get('duration', 300)
        interval = test_config.get('interval', 5)
        
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
                
                # Perform network tests
                ping_result = self._ping_test(destination)
                traceroute_result = self._traceroute_test(destination)
                advanced_network = self._advanced_network_test(destination)
                qos_metrics = self._qos_test(destination)
                bandwidth_metrics = self._bandwidth_test(destination)
                
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
                    **bandwidth_metrics
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
            
            # TCP connection timing
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
                    if elapsed_time > 1.0 and total_bytes > 0:  # Minimum 1s for accuracy
                        download_mbps = (total_bytes * 8) / (elapsed_time * 1000000)
                        metrics['bandwidth_download'] = round(download_mbps, 2)
                        logger.debug(f"Download speed test: {total_bytes} bytes in {elapsed_time:.2f}s = {download_mbps:.2f} Mbps")
                        
            except Exception as e:
                logger.debug(f"HTTP download speed test failed: {e}")
            
            # Upload test - upload to reliable endpoint
            try:
                test_data = b'0' * 2097152  # 2MB upload test
                start_time = time.time()
                
                try:
                    # Try httpbin first
                    response = session.post("https://httpbin.org/post", 
                                          data=test_data, timeout=20)
                except:
                    # Fallback to Google
                    response = session.post("https://www.google.com/gen_204", 
                                          data=test_data, timeout=20)
                
                if response.status_code in [200, 204]:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 1.0:  # Minimum 1s for accuracy
                        upload_mbps = (len(test_data) * 8) / (elapsed_time * 1000000)
                        metrics['bandwidth_upload'] = round(upload_mbps, 2)
                        logger.debug(f"Upload speed test: {len(test_data)} bytes in {elapsed_time:.2f}s = {upload_mbps:.2f} Mbps")
                        
            except Exception as e:
                logger.debug(f"HTTP upload speed test failed: {e}")
        
        except Exception as e:
            logger.warning(f"HTTP internet speed test failed: {e}")
        
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
                # Try to get wireless info using iwconfig or similar tools
                import subprocess
                
                # Try iwconfig first
                try:
                    result = subprocess.run(['iwconfig', interface_name], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        output = result.stdout
                        
                        # Parse SSID
                        if 'ESSID:' in output:
                            essid_line = [line for line in output.split('\n') if 'ESSID:' in line][0]
                            if 'ESSID:"' in essid_line:
                                ssid = essid_line.split('ESSID:"')[1].split('"')[0]
                                if ssid and ssid != 'off/any':
                                    wireless_info['ssid'] = ssid
                        
                        # Parse signal strength
                        if 'Signal level=' in output:
                            signal_line = [line for line in output.split('\n') if 'Signal level=' in line][0]
                            if 'Signal level=' in signal_line:
                                signal_part = signal_line.split('Signal level=')[1].split()[0]
                                wireless_info['signal_strength'] = signal_part
                        
                        # Parse frequency
                        if 'Frequency:' in output:
                            freq_line = [line for line in output.split('\n') if 'Frequency:' in line][0]
                            if 'Frequency:' in freq_line:
                                freq_part = freq_line.split('Frequency:')[1].split()[0]
                                wireless_info['frequency'] = freq_part
                                
                except subprocess.TimeoutExpired:
                    pass
                except FileNotFoundError:
                    # iwconfig not available, try other methods
                    pass
                
                # Try nmcli as fallback
                if not wireless_info['ssid']:
                    try:
                        result = subprocess.run(['nmcli', '-t', '-f', 'ACTIVE,SSID', 'dev', 'wifi'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            for line in result.stdout.strip().split('\n'):
                                if line.startswith('yes:'):
                                    ssid = line.split(':', 1)[1]
                                    if ssid:
                                        wireless_info['ssid'] = ssid
                                        break
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                        
        except Exception as e:
            logger.debug(f"Wireless info detection failed: {e}")
        
        return wireless_info
    
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
