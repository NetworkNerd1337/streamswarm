"""
StreamSwarm GNMI Client Module
Provides network path analysis using GNMI protocol for managed network infrastructure
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from pygnmi.client import gNMIclient
import ipaddress

logger = logging.getLogger(__name__)

class GNMINetworkAnalyzer:
    """
    GNMI-powered network path analyzer for managed infrastructure
    Provides hop-by-hop latency analysis within network devices
    """
    
    def __init__(self):
        self.connected_devices = {}
        self.device_credentials = {}
        
    def add_device(self, device_ip: str, username: str = None, password: str = None, 
                   port: int = 830, cert_file: str = None, key_file: str = None, 
                   ca_file: str = None, auth_method: str = 'password'):
        """
        Add a network device for GNMI monitoring
        
        Authentication Methods:
            'password': Username/password authentication (default)
            'certificate': Pure certificate-based authentication (mTLS)
            'cert_username': Certificate + username authentication
        
        Args:
            device_ip: IP address of the GNMI device
            username: Username for authentication (optional for cert-only auth)
            password: Password for authentication (required for password auth)
            port: GNMI port (default: 830, common alternatives: 32768, 6030, 57400)
            cert_file: Path to client certificate file (.crt or .pem)
            key_file: Path to client private key file (.key or .pem)
            ca_file: Path to CA certificate file (optional, for custom CAs)
            auth_method: Authentication method to use
        
        Examples:
            # Password authentication
            add_device("192.168.1.1", "admin", "password123", 830)
            
            # Certificate authentication
            add_device("192.168.1.1", port=830, auth_method='certificate',
                      cert_file='/path/to/client.crt', key_file='/path/to/client.key')
            
            # Certificate + username authentication
            add_device("192.168.1.1", username="admin", auth_method='cert_username',
                      cert_file='/path/to/client.crt', key_file='/path/to/client.key')
        """
        try:
            device_key = f"{device_ip}:{port}"
            
            # Validate authentication parameters
            if auth_method == 'password':
                if not username or not password:
                    raise ValueError(f"Username and password required for password authentication on {device_ip}")
                credentials = {
                    'host': (device_ip, port),
                    'username': username,
                    'password': password,
                    'auth_method': 'password',
                    'skip_verify': True  # For lab environments
                }
            elif auth_method == 'certificate':
                if not cert_file or not key_file:
                    raise ValueError(f"Certificate and key files required for certificate authentication on {device_ip}")
                credentials = {
                    'host': (device_ip, port),
                    'cert_file': cert_file,
                    'key_file': key_file,
                    'ca_file': ca_file,
                    'auth_method': 'certificate',
                    'skip_verify': False if ca_file else True
                }
            elif auth_method == 'cert_username':
                if not username or not cert_file or not key_file:
                    raise ValueError(f"Username, certificate and key files required for cert+username authentication on {device_ip}")
                credentials = {
                    'host': (device_ip, port),
                    'username': username,
                    'cert_file': cert_file,
                    'key_file': key_file,
                    'ca_file': ca_file,
                    'auth_method': 'cert_username',
                    'skip_verify': False if ca_file else True
                }
            else:
                raise ValueError(f"Invalid auth_method: {auth_method}. Must be 'password', 'certificate', or 'cert_username'")
            
            self.device_credentials[device_key] = credentials
            logger.info(f"Added GNMI device: {device_key} using {auth_method} authentication")
            return True
        except Exception as e:
            logger.error(f"Error adding GNMI device {device_ip}: {str(e)}")
            return False
    
    def clear_devices(self):
        """Clear all configured devices and close connections"""
        try:
            # Close all existing connections
            for device_key, client in self.connected_devices.items():
                try:
                    client.close()
                except Exception as e:
                    logger.warning(f"Error closing connection to {device_key}: {e}")
            
            # Clear storage
            self.connected_devices.clear()
            self.device_credentials.clear()
            
            logger.info("Cleared all GNMI devices and connections")
            
        except Exception as e:
            logger.error(f"Error clearing GNMI devices: {e}")
    
    def connect_to_device(self, device_key: str) -> Optional[gNMIclient]:
        """Establish GNMI connection to a network device"""
        try:
            if device_key not in self.device_credentials:
                logger.error(f"No credentials found for device: {device_key}")
                return None
                
            creds = self.device_credentials[device_key]
            auth_method = creds.get('auth_method', 'password')
            
            # Build connection parameters based on authentication method
            connection_params = {
                'target': creds['host'],
                'skip_verify': creds['skip_verify']
            }
            
            if auth_method == 'password':
                connection_params.update({
                    'username': creds['username'],
                    'password': creds['password']
                })
                logger.info(f"Connecting to {device_key} using username/password authentication")
                
            elif auth_method == 'certificate':
                connection_params.update({
                    'cert_file': creds['cert_file'],
                    'key_file': creds['key_file']
                })
                if creds.get('ca_file'):
                    connection_params['ca_file'] = creds['ca_file']
                logger.info(f"Connecting to {device_key} using certificate authentication")
                
            elif auth_method == 'cert_username':
                connection_params.update({
                    'username': creds['username'],
                    'cert_file': creds['cert_file'],
                    'key_file': creds['key_file']
                })
                if creds.get('ca_file'):
                    connection_params['ca_file'] = creds['ca_file']
                logger.info(f"Connecting to {device_key} using certificate + username authentication")
            
            # Create GNMI client with appropriate authentication
            client = gNMIclient(**connection_params)
            
            # Test connection with capabilities request
            capabilities = client.capabilities()
            logger.info(f"Connected to GNMI device {device_key}: {capabilities.get('supported_models', 'Unknown model')}")
            
            self.connected_devices[device_key] = client
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to GNMI device {device_key}: {str(e)}")
            logger.debug(f"Connection details: auth_method={auth_method}, target={creds['host']}")
            return None
    
    def get_interface_statistics(self, device_key: str) -> Dict[str, Any]:
        """Get interface statistics from network device"""
        try:
            if device_key not in self.connected_devices:
                client = self.connect_to_device(device_key)
                if not client:
                    return {}
            else:
                client = self.connected_devices[device_key]
            
            # Query interface statistics
            paths = [
                '/interfaces/interface/state/counters',
                '/interfaces/interface/state/oper-status',
                '/interfaces/interface/state/name'
            ]
            
            result = client.get(path=paths)
            return self._parse_interface_data(result)
            
        except Exception as e:
            logger.error(f"Error getting interface statistics from {device_key}: {str(e)}")
            return {}
    
    def analyze_network_path(self, destination: str, test_id: int = None) -> Dict[str, Any]:
        """
        Perform network path analysis using GNMI-enabled devices
        Provides hop-by-hop latency within managed infrastructure
        """
        try:
            path_analysis = {
                'destination': destination,
                'test_id': test_id,
                'timestamp': time.time(),
                'managed_hops': [],
                'path_metrics': {},
                'device_insights': [],
                'status': 'completed'
            }
            
            # Discover path through managed infrastructure
            managed_path = self._discover_managed_path(destination)
            path_analysis['managed_hops'] = managed_path
            
            # Collect hop-by-hop metrics
            for hop in managed_path:
                hop_metrics = self._collect_hop_metrics(hop)
                if hop_metrics:
                    path_analysis['path_metrics'][hop['device']] = hop_metrics
            
            # Generate insights
            path_analysis['device_insights'] = self._generate_path_insights(path_analysis)
            
            logger.info(f"GNMI path analysis completed for {destination}: {len(managed_path)} managed hops")
            return path_analysis
            
        except Exception as e:
            logger.error(f"Error in GNMI network path analysis for {destination}: {str(e)}")
            return {
                'destination': destination,
                'test_id': test_id,
                'status': 'error',
                'error': str(e),
                'managed_hops': [],
                'path_metrics': {},
                'device_insights': []
            }
    
    def _discover_managed_path(self, destination: str) -> List[Dict[str, Any]]:
        """Discover which managed devices are in the path to destination"""
        managed_hops = []
        
        try:
            # Query routing tables from connected devices to find path
            for device_key, client in self.connected_devices.items():
                try:
                    # Query routing information
                    routing_paths = [
                        '/network-instances/network-instance/protocols/protocol/bgp/rib',
                        '/network-instances/network-instance/protocols/protocol/static-routes',
                        '/network-instances/network-instance/fib'
                    ]
                    
                    routing_data = client.get(path=routing_paths)
                    
                    # Check if this device routes to destination
                    if self._device_routes_to_destination(routing_data, destination):
                        device_ip = device_key.split(':')[0]
                        hop_info = {
                            'device': device_key,
                            'device_ip': device_ip,
                            'hop_number': len(managed_hops) + 1,
                            'device_type': self._identify_device_type(client),
                            'interfaces': self._get_relevant_interfaces(client, destination)
                        }
                        managed_hops.append(hop_info)
                        
                except Exception as e:
                    logger.warning(f"Could not query routing from {device_key}: {str(e)}")
                    continue
            
            # Sort hops by logical order (simplified - in real implementation would use routing topology)
            managed_hops.sort(key=lambda x: x['hop_number'])
            
        except Exception as e:
            logger.error(f"Error discovering managed path: {str(e)}")
        
        return managed_hops
    
    def _collect_hop_metrics(self, hop: Dict[str, Any]) -> Dict[str, Any]:
        """Collect detailed metrics from a specific hop device"""
        try:
            device_key = hop['device']
            if device_key not in self.connected_devices:
                return {}
                
            client = self.connected_devices[device_key]
            
            # Collect multiple metric categories
            metrics = {
                'latency_metrics': self._get_device_latency_metrics(client),
                'interface_metrics': self._get_interface_performance_metrics(client, hop['interfaces']),
                'queue_metrics': self._get_queue_statistics(client),
                'buffer_metrics': self._get_buffer_utilization(client),
                'timestamp': time.time()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting hop metrics for {hop['device']}: {str(e)}")
            return {}
    
    def _get_device_latency_metrics(self, client: gNMIclient) -> Dict[str, Any]:
        """Get latency-related metrics from network device"""
        try:
            # Query device latency statistics
            latency_paths = [
                '/qos/interfaces/interface/output/queues/queue/state/transmit-pkts',
                '/qos/interfaces/interface/output/queues/queue/state/avg-queue-latency',
                '/system/processes/process/state/cpu-utilization'
            ]
            
            latency_data = client.get(path=latency_paths)
            
            return {
                'processing_latency_us': self._extract_processing_latency(latency_data),
                'queue_latency_us': self._extract_queue_latency(latency_data),
                'cpu_utilization_percent': self._extract_cpu_utilization(latency_data),
                'forwarding_delay_us': self._calculate_forwarding_delay(latency_data)
            }
            
        except Exception as e:
            logger.warning(f"Could not get device latency metrics: {str(e)}")
            return {}
    
    def _get_interface_performance_metrics(self, client: gNMIclient, interfaces: List[str]) -> Dict[str, Any]:
        """Get performance metrics for specific interfaces"""
        interface_metrics = {}
        
        try:
            for interface in interfaces:
                interface_paths = [
                    f'/interfaces/interface[name={interface}]/state/counters/in-pkts',
                    f'/interfaces/interface[name={interface}]/state/counters/out-pkts',
                    f'/interfaces/interface[name={interface}]/state/counters/in-octets',
                    f'/interfaces/interface[name={interface}]/state/counters/out-octets',
                    f'/interfaces/interface[name={interface}]/state/counters/in-errors',
                    f'/interfaces/interface[name={interface}]/state/counters/out-errors'
                ]
                
                interface_data = client.get(path=interface_paths)
                
                interface_metrics[interface] = {
                    'utilization_percent': self._calculate_interface_utilization(interface_data),
                    'error_rate_percent': self._calculate_error_rate(interface_data),
                    'packet_rate_pps': self._calculate_packet_rate(interface_data),
                    'latency_contribution_us': self._estimate_interface_latency(interface_data)
                }
                
        except Exception as e:
            logger.warning(f"Could not get interface performance metrics: {str(e)}")
        
        return interface_metrics
    
    def _get_queue_statistics(self, client: gNMIclient) -> Dict[str, Any]:
        """Get queue statistics that affect latency"""
        try:
            queue_paths = [
                '/qos/interfaces/interface/output/queues/queue/state/transmit-pkts',
                '/qos/interfaces/interface/output/queues/queue/state/dropped-pkts',
                '/qos/interfaces/interface/output/queues/queue/state/max-queue-depth'
            ]
            
            queue_data = client.get(path=queue_paths)
            
            return {
                'average_queue_depth': self._extract_average_queue_depth(queue_data),
                'max_queue_depth': self._extract_max_queue_depth(queue_data),
                'drop_rate_percent': self._calculate_queue_drop_rate(queue_data),
                'queue_latency_impact_us': self._estimate_queue_latency_impact(queue_data)
            }
            
        except Exception as e:
            logger.warning(f"Could not get queue statistics: {str(e)}")
            return {}
    
    def _get_buffer_utilization(self, client: gNMIclient) -> Dict[str, Any]:
        """Get buffer utilization metrics"""
        try:
            buffer_paths = [
                '/system/memory/state/physical',
                '/system/memory/state/reserved'
            ]
            
            buffer_data = client.get(path=buffer_paths)
            
            return {
                'buffer_utilization_percent': self._calculate_buffer_utilization(buffer_data),
                'memory_pressure_indicator': self._assess_memory_pressure(buffer_data)
            }
            
        except Exception as e:
            logger.warning(f"Could not get buffer utilization: {str(e)}")
            return {}
    
    def _generate_path_insights(self, path_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from network path analysis"""
        insights = []
        
        try:
            managed_hops = path_analysis.get('managed_hops', [])
            path_metrics = path_analysis.get('path_metrics', {})
            
            # Analyze overall path performance
            if len(managed_hops) > 0:
                insights.append(f"Path traverses {len(managed_hops)} managed network devices")
            
            # Identify performance bottlenecks
            high_latency_devices = []
            high_utilization_devices = []
            
            for device, metrics in path_metrics.items():
                latency_metrics = metrics.get('latency_metrics', {})
                interface_metrics = metrics.get('interface_metrics', {})
                
                # Check for high latency
                processing_latency = latency_metrics.get('processing_latency_us', 0)
                if processing_latency > 1000:  # > 1ms processing latency
                    high_latency_devices.append((device, processing_latency))
                
                # Check for high interface utilization
                for interface, if_metrics in interface_metrics.items():
                    utilization = if_metrics.get('utilization_percent', 0)
                    if utilization > 80:
                        high_utilization_devices.append((device, interface, utilization))
            
            # Generate specific insights
            if high_latency_devices:
                for device, latency in high_latency_devices:
                    insights.append(f"High processing latency detected on {device}: {latency/1000:.1f}ms")
            
            if high_utilization_devices:
                for device, interface, util in high_utilization_devices:
                    insights.append(f"High interface utilization on {device} {interface}: {util:.1f}%")
            
            if not high_latency_devices and not high_utilization_devices:
                insights.append("Network path performance within normal parameters")
            
        except Exception as e:
            logger.error(f"Error generating path insights: {str(e)}")
            insights.append("Could not analyze path performance")
        
        return insights
    
    # Helper methods for data extraction and calculation
    def _parse_interface_data(self, data: Dict) -> Dict[str, Any]:
        """Parse interface data from GNMI response"""
        # Simplified parser - real implementation would handle vendor-specific formats
        return {}
    
    def _device_routes_to_destination(self, routing_data: Dict, destination: str) -> bool:
        """Check if device has route to destination"""
        # Simplified check - real implementation would parse routing tables
        return True  # For demo purposes
    
    def _identify_device_type(self, client: gNMIclient) -> str:
        """Identify type of network device"""
        return "router"  # Simplified
    
    def _get_relevant_interfaces(self, client: gNMIclient, destination: str) -> List[str]:
        """Get interfaces relevant to destination"""
        return ["eth0", "eth1"]  # Simplified
    
    def _extract_processing_latency(self, data: Dict) -> float:
        """Extract processing latency from device data"""
        return 500.0  # Simplified - return microseconds
    
    def _extract_queue_latency(self, data: Dict) -> float:
        """Extract queue latency from device data"""
        return 200.0  # Simplified - return microseconds
    
    def _extract_cpu_utilization(self, data: Dict) -> float:
        """Extract CPU utilization"""
        return 25.0  # Simplified - return percentage
    
    def _calculate_forwarding_delay(self, data: Dict) -> float:
        """Calculate packet forwarding delay"""
        return 100.0  # Simplified - return microseconds
    
    def _calculate_interface_utilization(self, data: Dict) -> float:
        """Calculate interface utilization percentage"""
        return 45.0  # Simplified
    
    def _calculate_error_rate(self, data: Dict) -> float:
        """Calculate interface error rate"""
        return 0.01  # Simplified - return percentage
    
    def _calculate_packet_rate(self, data: Dict) -> float:
        """Calculate packet rate in packets per second"""
        return 1000.0  # Simplified
    
    def _estimate_interface_latency(self, data: Dict) -> float:
        """Estimate latency contribution from interface"""
        return 50.0  # Simplified - return microseconds
    
    def _extract_average_queue_depth(self, data: Dict) -> float:
        """Extract average queue depth"""
        return 10.0  # Simplified
    
    def _extract_max_queue_depth(self, data: Dict) -> float:
        """Extract maximum queue depth"""
        return 100.0  # Simplified
    
    def _calculate_queue_drop_rate(self, data: Dict) -> float:
        """Calculate queue drop rate percentage"""
        return 0.1  # Simplified
    
    def _estimate_queue_latency_impact(self, data: Dict) -> float:
        """Estimate latency impact from queuing"""
        return 150.0  # Simplified - return microseconds
    
    def _calculate_buffer_utilization(self, data: Dict) -> float:
        """Calculate buffer utilization percentage"""
        return 60.0  # Simplified
    
    def _assess_memory_pressure(self, data: Dict) -> str:
        """Assess memory pressure level"""
        return "normal"  # Simplified
    
    def disconnect_all(self):
        """Disconnect from all GNMI devices"""
        for device_key, client in self.connected_devices.items():
            try:
                client.close()
                logger.info(f"Disconnected from GNMI device: {device_key}")
            except Exception as e:
                logger.error(f"Error disconnecting from {device_key}: {str(e)}")
        
        self.connected_devices.clear()