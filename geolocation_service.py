"""
StreamSwarm Geolocation Path Analysis Service
Provides IP geolocation, path visualization, and network route analysis
"""

import json
import logging
import os
import re
import subprocess
import time
import ipaddress
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from typing import Dict, List, Optional, Tuple
import requests
import folium
from folium import plugins

logger = logging.getLogger(__name__)

class GeolocationService:
    """
    Service for IP geolocation lookup and network path analysis
    Uses multiple fallback methods for reliable geolocation
    """
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.cache_timeout = 3600  # 1 hour cache timeout
        
        # Free geolocation APIs (no API key required)
        self.apis = [
            'http://ip-api.com/json/{}',  # Free, 45 requests/minute
            'https://ipapi.co/{}/json/',  # Free, 1000 requests/day
            'https://freegeoip.app/json/{}',  # Free alternative
        ]
        
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private/local"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
        except ValueError:
            return False
    
    def _extract_ip_from_traceroute_line(self, line: str) -> Optional[str]:
        """Extract IP address from traceroute line"""
        # Match IP addresses in various traceroute formats
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, line)
        
        for ip in ips:
            if not self._is_private_ip(ip) and ip != '0.0.0.0':
                return ip
        return None
    
    def _extract_latency_from_traceroute_line(self, line: str) -> List[float]:
        """Extract latency values from traceroute line"""
        # Match latency patterns like "12.345 ms" or "12.345ms"
        latency_pattern = r'(\d+\.?\d*)\s*ms'
        matches = re.findall(latency_pattern, line)
        return [float(match) for match in matches if float(match) > 0]
    
    def get_ip_geolocation(self, ip: str) -> Optional[Dict]:
        """
        Get geolocation data for an IP address using free APIs
        Returns dict with lat, lon, city, country, etc.
        """
        if not ip or self._is_private_ip(ip):
            return None
            
        # Check cache first
        cache_key = f"geo_{ip}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        # Try each API until one works
        for api_url in self.apis:
            try:
                url = api_url.format(ip)
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Normalize response format (different APIs use different fields)
                    if api_url.startswith('http://ip-api.com'):
                        if data.get('status') == 'success':
                            geo_data = {
                                'ip': ip,
                                'latitude': data.get('lat'),
                                'longitude': data.get('lon'),
                                'city': data.get('city'),
                                'region': data.get('regionName'),
                                'country': data.get('country'),
                                'country_code': data.get('countryCode'),
                                'isp': data.get('isp'),
                                'org': data.get('org'),
                                'timezone': data.get('timezone')
                            }
                        else:
                            continue
                    elif api_url.startswith('https://ipapi.co'):
                        if data.get('error') is None:
                            geo_data = {
                                'ip': ip,
                                'latitude': data.get('latitude'),
                                'longitude': data.get('longitude'),
                                'city': data.get('city'),
                                'region': data.get('region'),
                                'country': data.get('country_name'),
                                'country_code': data.get('country_code'),
                                'isp': data.get('org'),
                                'org': data.get('org'),
                                'timezone': data.get('timezone')
                            }
                        else:
                            continue
                    else:
                        # Generic format
                        geo_data = {
                            'ip': ip,
                            'latitude': data.get('latitude') or data.get('lat'),
                            'longitude': data.get('longitude') or data.get('lon'),
                            'city': data.get('city'),
                            'region': data.get('region') or data.get('regionName'),
                            'country': data.get('country') or data.get('country_name'),
                            'country_code': data.get('country_code') or data.get('countryCode'),
                            'isp': data.get('isp') or data.get('org'),
                            'org': data.get('org'),
                            'timezone': data.get('timezone')
                        }
                    
                    # Validate that we got coordinates
                    if geo_data.get('latitude') is not None and geo_data.get('longitude') is not None:
                        # Cache the result
                        self.cache[cache_key] = (geo_data, time.time())
                        logger.debug(f"Geolocation found for {ip}: {geo_data.get('city')}, {geo_data.get('country')}")
                        return geo_data
                        
            except Exception as e:
                logger.debug(f"Geolocation API {api_url} failed for {ip}: {e}")
                continue
        
        logger.warning(f"No geolocation data found for IP {ip}")
        return None
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth
        Returns distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    
    def perform_mtr_analysis(self, destination: str, count: int = 10) -> Dict:
        """
        Perform MTR (My Traceroute) analysis for detailed hop-by-hop statistics
        """
        try:
            # Try mtr first, fallback to traceroute with multiple probes
            try:
                cmd = ['mtr', '--report', '--report-cycles', str(count), '--json', destination]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    return json.loads(result.stdout)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            # Fallback: multiple traceroute runs for statistical analysis
            hop_stats = {}
            
            for i in range(count):
                cmd = ['traceroute', '-n', destination]  # -n for no DNS resolution
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if line.strip() and line.strip()[0].isdigit():
                            hop_num = int(line.strip().split()[0])
                            ip = self._extract_ip_from_traceroute_line(line)
                            latencies = self._extract_latency_from_traceroute_line(line)
                            
                            if ip and latencies:
                                if hop_num not in hop_stats:
                                    hop_stats[hop_num] = {'ip': ip, 'latencies': []}
                                hop_stats[hop_num]['latencies'].extend(latencies)
            
            # Calculate statistics for each hop
            mtr_data = {'hubs': []}
            for hop_num in sorted(hop_stats.keys()):
                hop_data = hop_stats[hop_num]
                latencies = hop_data['latencies']
                
                if latencies:
                    mtr_data['hubs'].append({
                        'count': hop_num,
                        'host': hop_data['ip'],
                        'Loss%': 0,  # Would need packet loss detection
                        'Snt': len(latencies),
                        'Last': latencies[-1],
                        'Avg': sum(latencies) / len(latencies),
                        'Best': min(latencies),
                        'Wrst': max(latencies),
                        'StDev': self._calculate_std_dev(latencies)
                    })
            
            return mtr_data
            
        except Exception as e:
            logger.error(f"MTR analysis failed: {e}")
            return {'hubs': []}
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return sqrt(variance)
    
    def analyze_traceroute_path(self, traceroute_data: List[str], destination: str) -> Dict:
        """
        Analyze traceroute data and create geolocation path analysis
        Returns comprehensive path analysis with geolocation data
        """
        if not traceroute_data or not isinstance(traceroute_data, list):
            return {}
        
        path_analysis = {
            'hops': [],
            'total_hops': 0,
            'total_distance_km': 0,
            'geographic_efficiency': 100,
            'countries_traversed': set(),
            'isps_traversed': set(),
            'map_data': None,
            'performance_insights': []
        }
        
        try:
            # Skip MTR analysis to avoid timeouts - we already have traceroute data
            mtr_hops = {}
            
            # Parse each traceroute line
            prev_lat, prev_lon = None, None
            total_distance = 0
            
            for i, line in enumerate(traceroute_data):
                if not line.strip() or not line.strip()[0].isdigit():
                    continue
                
                hop_num = i + 1
                ip = self._extract_ip_from_traceroute_line(line)
                latencies = self._extract_latency_from_traceroute_line(line)
                
                if not ip:
                    continue
                
                # Get geolocation for this hop
                geo_data = self.get_ip_geolocation(ip)
                
                # Get MTR statistics for this hop
                mtr_stats = mtr_hops.get(hop_num, {})
                
                hop_info = {
                    'hop_number': hop_num,
                    'ip': ip,
                    'hostname': None,  # Could add reverse DNS lookup
                    'latency_ms': latencies[0] if latencies else None,
                    'latency_min': mtr_stats.get('Best'),
                    'latency_max': mtr_stats.get('Wrst'),
                    'latency_avg': mtr_stats.get('Avg'),
                    'latency_std_dev': mtr_stats.get('StDev'),
                    'packet_loss_percent': mtr_stats.get('Loss%', 0),
                    'geolocation': geo_data,
                    'distance_from_prev_km': 0,
                    'cumulative_distance_km': total_distance
                }
                
                # Calculate distance if we have geolocation
                if geo_data and geo_data.get('latitude') and geo_data.get('longitude'):
                    current_lat = geo_data['latitude']
                    current_lon = geo_data['longitude']
                    
                    if prev_lat is not None and prev_lon is not None:
                        distance = self.calculate_distance(prev_lat, prev_lon, current_lat, current_lon)
                        hop_info['distance_from_prev_km'] = distance
                        total_distance += distance
                        hop_info['cumulative_distance_km'] = total_distance
                    
                    prev_lat, prev_lon = current_lat, current_lon
                    
                    # Track countries and ISPs
                    if geo_data.get('country'):
                        path_analysis['countries_traversed'].add(geo_data['country'])
                    if geo_data.get('isp'):
                        path_analysis['isps_traversed'].add(geo_data['isp'])
                
                path_analysis['hops'].append(hop_info)
            
            path_analysis['total_hops'] = len(path_analysis['hops'])
            path_analysis['total_distance_km'] = total_distance
            
            # Calculate geographic efficiency
            if len(path_analysis['hops']) >= 2:
                first_hop = path_analysis['hops'][0]
                last_hop = path_analysis['hops'][-1]
                
                if (first_hop.get('geolocation') and last_hop.get('geolocation') and
                    first_hop['geolocation'].get('latitude') and last_hop['geolocation'].get('latitude')):
                    
                    direct_distance = self.calculate_distance(
                        first_hop['geolocation']['latitude'],
                        first_hop['geolocation']['longitude'],
                        last_hop['geolocation']['latitude'],
                        last_hop['geolocation']['longitude']
                    )
                    
                    if direct_distance > 0:
                        path_analysis['geographic_efficiency'] = (direct_distance / total_distance) * 100
            
            # Generate performance insights
            path_analysis['performance_insights'] = self._generate_performance_insights(path_analysis)
            
            # Convert sets to lists for JSON serialization
            path_analysis['countries_traversed'] = list(path_analysis['countries_traversed'])
            path_analysis['isps_traversed'] = list(path_analysis['isps_traversed'])
            
        except Exception as e:
            logger.error(f"Error analyzing traceroute path: {e}")
        
        return path_analysis
    
    def _generate_performance_insights(self, path_analysis: Dict) -> List[str]:
        """Generate actionable performance insights from path analysis"""
        insights = []
        
        hops = path_analysis.get('hops', [])
        if not hops:
            return insights
        
        # Analyze latency jumps
        prev_latency = 0
        for hop in hops:
            latency = hop.get('latency_avg') or hop.get('latency_ms')
            if latency and prev_latency:
                latency_jump = latency - prev_latency
                if latency_jump > 50:  # Significant latency increase
                    geo = hop.get('geolocation', {})
                    location = f"{geo.get('city', 'Unknown')}, {geo.get('country', 'Unknown')}"
                    insights.append(f"High latency increase (+{latency_jump:.1f}ms) at hop {hop['hop_number']} in {location}")
            prev_latency = latency or prev_latency
        
        # Check geographic efficiency
        efficiency = path_analysis.get('geographic_efficiency', 100)
        if efficiency < 70:
            insights.append(f"Network path is geographically inefficient ({efficiency:.1f}% efficiency)")
        
        # Check for high packet loss
        for hop in hops:
            loss = hop.get('packet_loss_percent', 0)
            if loss > 5:
                geo = hop.get('geolocation', {})
                location = f"{geo.get('city', 'Unknown')}, {geo.get('country', 'Unknown')}"
                insights.append(f"High packet loss ({loss:.1f}%) detected at hop {hop['hop_number']} in {location}")
        
        # Check for many countries traversed
        countries = len(path_analysis.get('countries_traversed', []))
        if countries > 5:
            insights.append(f"Traffic routes through {countries} countries, consider CDN or closer servers")
        
        return insights
    
    def create_path_map(self, path_analysis: Dict, destination: str) -> str:
        """
        Create an interactive world map showing the network path
        Returns HTML string of the map
        """
        if not path_analysis.get('hops'):
            return ""
        
        try:
            # Calculate map center and zoom
            lats = []
            lons = []
            
            for hop in path_analysis['hops']:
                geo = hop.get('geolocation')
                if geo and geo.get('latitude') and geo.get('longitude'):
                    lats.append(geo['latitude'])
                    lons.append(geo['longitude'])
            
            if not lats:
                return ""
            
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=2,
                tiles='OpenStreetMap',
                width='100%',
                height='500px'
            )
            
            # Add markers and path
            path_coordinates = []
            
            for i, hop in enumerate(path_analysis['hops']):
                geo = hop.get('geolocation')
                if not geo or not geo.get('latitude'):
                    continue
                
                lat, lon = geo['latitude'], geo['longitude']
                path_coordinates.append([lat, lon])
                
                # Create popup text
                popup_text = f"""
                <b>Hop {hop['hop_number']}</b><br>
                IP: {hop['ip']}<br>
                Location: {geo.get('city', 'Unknown')}, {geo.get('country', 'Unknown')}<br>
                ISP: {geo.get('isp', 'Unknown')}<br>
                Latency: {hop.get('latency_avg', hop.get('latency_ms', 'N/A'))} ms<br>
                Distance: {hop.get('cumulative_distance_km', 0):.1f} km from start
                """
                
                # Color code based on latency
                latency = hop.get('latency_avg') or hop.get('latency_ms') or 0
                if latency < 50:
                    color = 'green'
                elif latency < 150:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Add marker with hop number and popup
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='black',
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
                
                # Add hop number as a clickable marker with popup
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12px; font-weight: bold; color: white; text-shadow: 1px 1px 1px black; cursor: pointer; text-align: center; line-height: 20px; width: 20px; height: 20px; border-radius: 50%; background-color: rgba(0,0,0,0.3);">{hop["hop_number"]}</div>',
                        class_name="custom-div-icon",
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    )
                ).add_to(m)
            
            # Add path line
            if len(path_coordinates) > 1:
                folium.PolyLine(
                    path_coordinates,
                    color='blue',
                    weight=3,
                    opacity=0.7
                ).add_to(m)
            
            # Add summary information with color-coded efficiency
            efficiency = path_analysis.get('geographic_efficiency', 0)
            
            # Determine efficiency color based on percentage
            if efficiency >= 75:
                efficiency_color = '#28a745'  # Green
                efficiency_status = 'Excellent'
            elif efficiency >= 50:
                efficiency_color = '#ffc107'  # Yellow
                efficiency_status = 'Moderate'
            else:
                efficiency_color = '#dc3545'  # Red
                efficiency_status = 'Poor'
            
            summary_html = f"""
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 280px; height: 170px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:12px; padding: 10px">
                <h4>Path Summary</h4>
                <b>Destination:</b> {destination}<br>
                <b>Total Hops:</b> {path_analysis.get('total_hops', 0)}<br>
                <b>Total Distance:</b> {path_analysis.get('total_distance_km', 0):.1f} km<br>
                <b>Countries:</b> {len(path_analysis.get('countries_traversed', []))}<br>
                <b>Efficiency:</b> <span style="color: {efficiency_color}; font-weight: bold;">{efficiency:.1f}% ({efficiency_status})</span>
            </div>
            """
            
            m.get_root().html.add_child(folium.Element(summary_html))
            
            # Generate HTML
            return m._repr_html_()
            
        except Exception as e:
            logger.error(f"Error creating path map: {e}")
            return ""