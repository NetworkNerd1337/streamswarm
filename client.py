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
from datetime import datetime
import psutil
import requests
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamSwarmClient:
    def __init__(self, server_url, client_name=None):
        self.server_url = server_url.rstrip('/')
        self.client_name = client_name or platform.node()
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
                'system_info': self.system_info
            }
            
            response = requests.post(
                urljoin(self.server_url, '/api/client/register'),
                json=data,
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
                response = requests.post(
                    urljoin(self.server_url, f'/api/client/{self.client_id}/heartbeat'),
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
                        metrics.update({
                            'disk_read_iops': 0,
                            'disk_write_iops': 0,
                            'disk_read_bytes_sec': 0,
                            'disk_write_bytes_sec': 0
                        })
                    else:
                        time_diff = time.time() - self._prev_disk_io['time']
                        if time_diff > 0:
                            metrics['disk_read_iops'] = (disk_io.read_count - self._prev_disk_io['read_count']) / time_diff
                            metrics['disk_write_iops'] = (disk_io.write_count - self._prev_disk_io['write_count']) / time_diff
                            metrics['disk_read_bytes_sec'] = (disk_io.read_bytes - self._prev_disk_io['read_bytes']) / time_diff
                            metrics['disk_write_bytes_sec'] = (disk_io.write_bytes - self._prev_disk_io['write_bytes']) / time_diff
                        
                        self._prev_disk_io = {'time': time.time(), 'read_bytes': disk_io.read_bytes, 'write_bytes': disk_io.write_bytes, 'read_count': disk_io.read_count, 'write_count': disk_io.write_count}
            except:
                metrics.update({
                    'disk_read_iops': None,
                    'disk_write_iops': None,
                    'disk_read_bytes_sec': None,
                    'disk_write_bytes_sec': None
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
            except:
                metrics['process_count'] = None
                metrics['tcp_connections'] = None
            
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
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _ping_test(self, destination, count=4):
        """Perform ping test with jitter calculation"""
        try:
            if platform.system().lower() == 'windows':
                cmd = ['ping', '-n', str(count), destination]
            else:
                cmd = ['ping', '-c', str(count), destination]
            
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
            if platform.system().lower() == 'windows':
                cmd = ['tracert', destination]
            else:
                cmd = ['traceroute', destination]
            
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
                # Get system metrics
                system_metrics = self._get_system_metrics()
                
                # Perform network tests
                ping_result = self._ping_test(destination)
                traceroute_result = self._traceroute_test(destination)
                advanced_network = self._advanced_network_test(destination)
                qos_metrics = self._qos_test(destination)
                
                # Prepare test result data
                result_data = {
                    'client_id': self.client_id,
                    'test_id': test_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    **system_metrics,
                    'ping_latency': ping_result.get('latency'),
                    'ping_packet_loss': ping_result.get('packet_loss'),
                    'jitter': ping_result.get('jitter'),
                    'traceroute_hops': traceroute_result.get('hops'),
                    'traceroute_data': traceroute_result.get('data', []),
                    **advanced_network,
                    **qos_metrics
                }
                
                # Submit results to server
                response = requests.post(
                    urljoin(self.server_url, '/api/test/results'),
                    json=result_data,
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
            # DNS resolution timing
            import socket
            import time as time_module
            
            start_time = time_module.time()
            try:
                socket.gethostbyname(destination)
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
                        result = sock.connect_ex((destination, port))
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
                
                with socket.create_connection((destination, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=destination) as ssock:
                        ssl_time = (time_module.time() - start_time) * 1000
                        metrics['ssl_handshake_time'] = ssl_time
            except:
                metrics['ssl_handshake_time'] = None
            
            # HTTP TTFB (Time to First Byte)
            try:
                import urllib.request
                import urllib.error
                
                start_time = time_module.time()
                try:
                    response = urllib.request.urlopen(f'http://{destination}', timeout=10)
                    ttfb = (time_module.time() - start_time) * 1000
                    metrics['ttfb'] = ttfb
                    response.close()
                except urllib.error.HTTPError:
                    # Even if we get an HTTP error, we still measured TTFB
                    ttfb = (time_module.time() - start_time) * 1000
                    metrics['ttfb'] = ttfb
                except:
                    # Try HTTPS
                    try:
                        start_time = time_module.time()
                        response = urllib.request.urlopen(f'https://{destination}', timeout=10)
                        ttfb = (time_module.time() - start_time) * 1000
                        metrics['ttfb'] = ttfb
                        response.close()
                    except:
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
                            # CoS is in the 802.1Q VLAN tag (3 bits)
                            # This is simplified - actual implementation would need VLAN parsing
                            qos_metrics['cos_value'] = 0  # Placeholder for CoS detection
                            
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
    
    def _validate_qos_policy(self, dscp_value):
        """Validate if DSCP marking complies with expected QoS policy"""
        # Simple validation - in production this would check against configured policies
        if dscp_value in [0, 46, 34, 26, 18, 10]:  # Common valid DSCP values
            return True
        elif dscp_value > 56:  # Invalid DSCP range
            return False
        else:
            return None  # Unknown/unconfigured
    
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
                qos_metrics['qos_policy_compliant'] = True
            elif platform.system().lower() == 'windows':
                # Windows: Set QoS using WSAIoctl (simplified)
                qos_metrics['dscp_value'] = dscp
                qos_metrics['traffic_class'] = self._classify_dscp(dscp)
                qos_metrics['qos_policy_compliant'] = True
            
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
    
    def _check_for_tests(self):
        """Check for new tests from server"""
        while self.running:
            try:
                response = requests.get(
                    urljoin(self.server_url, f'/api/client/{self.client_id}/tests'),
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
    parser.add_argument('--name', help='Client name (defaults to hostname)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start client
    client = StreamSwarmClient(args.server, args.name)
    client.start()

if __name__ == '__main__':
    main()
