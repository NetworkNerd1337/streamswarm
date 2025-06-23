"""
AI-Powered Diagnostic Engine for StreamSwarm
Provides intelligent analysis and troubleshooting suggestions for network performance issues
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import statistics
from dataclasses import dataclass

from openai import OpenAI
from models import TestResult, Test, Client, db
from sqlalchemy import and_, desc

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@dataclass
class DiagnosticResult:
    """Structure for AI diagnostic results"""
    confidence: float  # 0-100
    primary_issue: str
    evidence: List[str]
    recommendations: List[str]
    severity: str  # low, medium, high, critical
    category: str  # network, system, application, infrastructure
    similar_incidents: List[Dict]

class NetworkDiagnosticEngine:
    """AI-powered network diagnostic engine"""
    
    def __init__(self):
        self.thresholds = {
            'ping_latency': {'good': 50, 'warning': 100, 'critical': 200},
            'packet_loss': {'good': 0.1, 'warning': 1.0, 'critical': 5.0},
            'bandwidth_download': {'good': 50, 'warning': 25, 'critical': 10},
            'cpu_percent': {'good': 70, 'warning': 85, 'critical': 95},
            'memory_percent': {'good': 80, 'warning': 90, 'critical': 95},
            'disk_percent': {'good': 80, 'warning': 90, 'critical': 95},
            'jitter': {'good': 10, 'warning': 25, 'critical': 50}
        }
    
    def analyze_test_results(self, test_id: int, result_id: Optional[int] = None) -> DiagnosticResult:
        """
        Perform comprehensive AI analysis of test results
        """
        if not openai_client:
            return self._fallback_rule_based_analysis(test_id, result_id)
        
        # Gather data for analysis
        test_data = self._gather_test_data(test_id, result_id)
        
        if not test_data['results']:
            return DiagnosticResult(
                confidence=0,
                primary_issue="Insufficient data for analysis",
                evidence=["No test results available"],
                recommendations=["Run additional tests to gather data"],
                severity="low",
                category="system",
                similar_incidents=[]
            )
        
        # Generate AI analysis
        ai_analysis = self._generate_ai_analysis(test_data)
        
        # Combine with rule-based analysis
        rule_analysis = self._rule_based_analysis(test_data)
        
        # Merge results
        return self._merge_analyses(ai_analysis, rule_analysis, test_data)
    
    def _gather_test_data(self, test_id: int, result_id: Optional[int] = None) -> Dict:
        """Gather comprehensive data for analysis"""
        test = Test.query.get(test_id)
        
        if result_id:
            # Analyze specific result
            result = TestResult.query.get(result_id)
            results = [result] if result else []
            # Get surrounding results for context
            context_results = TestResult.query.filter(
                and_(
                    TestResult.test_id == test_id,
                    TestResult.timestamp.between(
                        result.timestamp - timedelta(minutes=30),
                        result.timestamp + timedelta(minutes=30)
                    )
                )
            ).order_by(TestResult.timestamp).all()
        else:
            # Analyze recent results
            results = TestResult.query.filter(
                TestResult.test_id == test_id
            ).order_by(desc(TestResult.timestamp)).limit(50).all()
            context_results = results
        
        # Calculate statistics
        metrics = self._calculate_metrics_statistics(results)
        trends = self._calculate_trends(context_results)
        anomalies = self._detect_anomalies(results)
        
        return {
            'test': test,
            'results': results,
            'context_results': context_results,
            'metrics': metrics,
            'trends': trends,
            'anomalies': anomalies,
            'client_info': self._get_client_info(test_id)
        }
    
    def _calculate_metrics_statistics(self, results: List[TestResult]) -> Dict:
        """Calculate statistical measures for all metrics"""
        if not results:
            return {}
        
        metrics = {}
        metric_fields = [
            'ping_latency', 'ping_packet_loss', 'bandwidth_download', 'bandwidth_upload',
            'cpu_percent', 'memory_percent', 'disk_percent', 'jitter',
            'dns_resolution_time', 'tcp_connect_time', 'ssl_handshake_time', 'ttfb'
        ]
        
        for field in metric_fields:
            values = [getattr(r, field) for r in results if getattr(r, field) is not None]
            if values:
                metrics[field] = {
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'count': len(values),
                    'latest': values[0] if results else None
                }
        
        return metrics
    
    def _calculate_trends(self, results: List[TestResult]) -> Dict:
        """Calculate trend analysis"""
        if len(results) < 3:
            return {}
        
        trends = {}
        metric_fields = ['ping_latency', 'bandwidth_download', 'cpu_percent', 'memory_percent']
        
        for field in metric_fields:
            values = [getattr(r, field) for r in results if getattr(r, field) is not None]
            if len(values) >= 3:
                # Simple trend calculation
                recent_avg = statistics.mean(values[:len(values)//3])
                older_avg = statistics.mean(values[len(values)//3:])
                
                if recent_avg > older_avg * 1.1:
                    trends[field] = 'increasing'
                elif recent_avg < older_avg * 0.9:
                    trends[field] = 'decreasing'
                else:
                    trends[field] = 'stable'
        
        return trends
    
    def _detect_anomalies(self, results: List[TestResult]) -> List[Dict]:
        """Detect anomalous values in the data"""
        anomalies = []
        
        if len(results) < 5:
            return anomalies
        
        metric_fields = ['ping_latency', 'bandwidth_download', 'cpu_percent', 'memory_percent']
        
        for field in metric_fields:
            values = [getattr(r, field) for r in results if getattr(r, field) is not None]
            if len(values) >= 5:
                mean_val = statistics.mean(values)
                stdev_val = statistics.stdev(values)
                
                for i, value in enumerate(values[:10]):  # Check recent values
                    if abs(value - mean_val) > 2 * stdev_val:  # 2 sigma rule
                        anomalies.append({
                            'metric': field,
                            'value': value,
                            'expected_range': f"{mean_val - 2*stdev_val:.2f} - {mean_val + 2*stdev_val:.2f}",
                            'severity': 'high' if abs(value - mean_val) > 3 * stdev_val else 'medium'
                        })
        
        return anomalies
    
    def _get_client_info(self, test_id: int) -> Dict:
        """Get client information for context"""
        test = Test.query.get(test_id)
        clients = Client.query.join(TestResult).filter(TestResult.test_id == test_id).distinct().all()
        
        return {
            'client_count': len(clients),
            'clients': [{'hostname': c.hostname, 'ip_address': c.ip_address} for c in clients]
        }
    
    def _generate_ai_analysis(self, test_data: Dict) -> Dict:
        """Generate AI-powered analysis using OpenAI"""
        try:
            # Prepare data summary for AI
            summary = self._prepare_ai_summary(test_data)
            
            system_prompt = """You are an expert network engineer and system administrator with 20+ years of experience in network troubleshooting, performance optimization, and infrastructure analysis. 

Analyze the provided network monitoring data and provide:
1. Primary issue identification with confidence level (0-100)
2. Evidence supporting your diagnosis
3. Specific technical recommendations
4. Severity assessment (low/medium/high/critical)
5. Category (network/system/application/infrastructure)

Focus on actionable insights based on the metrics provided. Consider correlation between different metrics and provide root cause analysis."""

            user_prompt = f"""Analyze this network monitoring data:

Test Configuration:
- Destination: {test_data['test'].destination}
- Duration: {test_data['test'].duration} seconds
- Test Type: Network performance monitoring

Current Metrics Summary:
{json.dumps(summary, indent=2)}

Provide your analysis in the following JSON format:
{{
    "confidence": <0-100>,
    "primary_issue": "<concise issue description>",
    "evidence": ["<evidence point 1>", "<evidence point 2>"],
    "recommendations": ["<recommendation 1>", "<recommendation 2>"],
    "severity": "<low/medium/high/critical>",
    "category": "<network/system/application/infrastructure>"
}}"""

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"AI analysis error: {e}")
            return {}
    
    def _prepare_ai_summary(self, test_data: Dict) -> Dict:
        """Prepare concise summary for AI analysis"""
        metrics = test_data['metrics']
        trends = test_data['trends']
        anomalies = test_data['anomalies']
        
        summary = {
            'metrics_summary': {},
            'trends': trends,
            'anomalies': len(anomalies),
            'anomaly_details': anomalies[:5],  # Top 5 anomalies
            'client_count': test_data['client_info']['client_count']
        }
        
        # Include key metrics
        key_metrics = ['ping_latency', 'bandwidth_download', 'cpu_percent', 'memory_percent', 'jitter']
        for metric in key_metrics:
            if metric in metrics:
                data = metrics[metric]
                summary['metrics_summary'][metric] = {
                    'current': data['latest'],
                    'average': round(data['avg'], 2),
                    'max': round(data['max'], 2),
                    'variation': round(data['stdev'], 2)
                }
        
        return summary
    
    def _rule_based_analysis(self, test_data: Dict) -> Dict:
        """Rule-based analysis for deterministic issues"""
        issues = []
        evidence = []
        recommendations = []
        severity = "low"
        
        metrics = test_data['metrics']
        
        # Check critical thresholds
        for metric, thresholds in self.thresholds.items():
            if metric in metrics:
                current_value = metrics[metric]['latest']
                if current_value is not None:
                    if metric in ['ping_latency', 'packet_loss', 'cpu_percent', 'memory_percent', 'disk_percent', 'jitter']:
                        if current_value > thresholds['critical']:
                            issues.append(f"Critical {metric.replace('_', ' ')} detected")
                            evidence.append(f"{metric}: {current_value} (threshold: {thresholds['critical']})")
                            severity = "critical"
                        elif current_value > thresholds['warning']:
                            issues.append(f"Elevated {metric.replace('_', ' ')}")
                            evidence.append(f"{metric}: {current_value} (threshold: {thresholds['warning']})")
                            if severity not in ["critical"]:
                                severity = "high"
        
        # Network-specific rules
        if 'ping_latency' in metrics and 'jitter' in metrics:
            latency = metrics['ping_latency']['latest']
            jitter = metrics['jitter']['latest']
            if latency and jitter and jitter > latency * 0.3:
                issues.append("High network instability detected")
                evidence.append(f"Jitter ({jitter}ms) is >30% of latency ({latency}ms)")
                recommendations.append("Check for network congestion or routing instability")
        
        # System resource correlation
        if 'cpu_percent' in metrics and 'memory_percent' in metrics:
            cpu = metrics['cpu_percent']['latest']
            memory = metrics['memory_percent']['latest']
            if cpu and memory and cpu > 90 and memory > 90:
                issues.append("System resource exhaustion")
                evidence.append(f"CPU: {cpu}%, Memory: {memory}%")
                recommendations.append("Investigate resource-intensive processes")
        
        return {
            'issues': issues,
            'evidence': evidence,
            'recommendations': recommendations,
            'severity': severity
        }
    
    def _merge_analyses(self, ai_analysis: Dict, rule_analysis: Dict, test_data: Dict) -> DiagnosticResult:
        """Merge AI and rule-based analyses"""
        
        if ai_analysis:
            # Use AI analysis as primary
            confidence = ai_analysis.get('confidence', 50)
            primary_issue = ai_analysis.get('primary_issue', 'Performance issues detected')
            evidence = ai_analysis.get('evidence', [])
            recommendations = ai_analysis.get('recommendations', [])
            severity = ai_analysis.get('severity', 'medium')
            category = ai_analysis.get('category', 'network')
        else:
            # Fallback to rule-based
            confidence = 70 if rule_analysis['issues'] else 30
            primary_issue = rule_analysis['issues'][0] if rule_analysis['issues'] else 'No significant issues detected'
            evidence = rule_analysis['evidence']
            recommendations = rule_analysis['recommendations']
            severity = rule_analysis['severity']
            category = 'system'
        
        # Add rule-based evidence
        evidence.extend(rule_analysis['evidence'])
        recommendations.extend(rule_analysis['recommendations'])
        
        # Remove duplicates
        evidence = list(dict.fromkeys(evidence))
        recommendations = list(dict.fromkeys(recommendations))
        
        # Find similar incidents
        similar_incidents = self._find_similar_incidents(test_data)
        
        return DiagnosticResult(
            confidence=confidence,
            primary_issue=primary_issue,
            evidence=evidence[:10],  # Limit to top 10
            recommendations=recommendations[:8],  # Limit to top 8
            severity=severity,
            category=category,
            similar_incidents=similar_incidents
        )
    
    def _find_similar_incidents(self, test_data: Dict) -> List[Dict]:
        """Find historically similar incidents"""
        # Simple implementation - could be enhanced with ML similarity matching
        similar = []
        
        current_metrics = test_data['metrics']
        if not current_metrics:
            return similar
        
        # Look for tests with similar performance patterns
        recent_tests = Test.query.filter(
            Test.id != test_data['test'].id
        ).order_by(desc(Test.created_at)).limit(20).all()
        
        for test in recent_tests:
            results = TestResult.query.filter(
                TestResult.test_id == test.id
            ).limit(10).all()
            
            if results:
                # Simple similarity based on latency patterns
                test_metrics = self._calculate_metrics_statistics(results)
                
                if 'ping_latency' in current_metrics and 'ping_latency' in test_metrics:
                    current_latency = current_metrics['ping_latency']['avg']
                    test_latency = test_metrics['ping_latency']['avg']
                    
                    if abs(current_latency - test_latency) / max(current_latency, test_latency) < 0.2:
                        similar.append({
                            'test_id': test.id,
                            'test_name': test.name,
                            'destination': test.destination,
                            'created_at': test.created_at.isoformat(),
                            'similarity_score': 85  # Placeholder score
                        })
        
        return similar[:3]  # Return top 3 similar incidents
    
    def _fallback_rule_based_analysis(self, test_id: int, result_id: Optional[int] = None) -> DiagnosticResult:
        """Fallback analysis when AI is not available"""
        test_data = self._gather_test_data(test_id, result_id)
        rule_analysis = self._rule_based_analysis(test_data)
        
        confidence = 60 if rule_analysis['issues'] else 20
        primary_issue = rule_analysis['issues'][0] if rule_analysis['issues'] else 'Basic analysis - no critical issues detected'
        
        return DiagnosticResult(
            confidence=confidence,
            primary_issue=primary_issue,
            evidence=rule_analysis['evidence'],
            recommendations=rule_analysis['recommendations'] + ["Enable AI analysis for deeper insights"],
            severity=rule_analysis['severity'],
            category='system',
            similar_incidents=[]
        )

# Global diagnostic engine instance
diagnostic_engine = NetworkDiagnosticEngine()

def diagnose_test_performance(test_id: int, result_id: Optional[int] = None) -> DiagnosticResult:
    """
    Main function to diagnose test performance issues
    """
    return diagnostic_engine.analyze_test_results(test_id, result_id)

def diagnose_client_health(client_id: int) -> DiagnosticResult:
    """
    Diagnose overall client health across all tests
    """
    # Get recent test results for this client
    recent_results = TestResult.query.filter(
        and_(
            TestResult.client_id == client_id,
            TestResult.timestamp >= datetime.now() - timedelta(hours=24)
        )
    ).order_by(desc(TestResult.timestamp)).limit(50).all()
    
    if not recent_results:
        return DiagnosticResult(
            confidence=0,
            primary_issue="No recent data available for client health analysis",
            evidence=["No test results in the last 24 hours"],
            recommendations=["Run tests to establish baseline health metrics"],
            severity="low",
            category="system",
            similar_incidents=[]
        )
    
    # Group by test_id for analysis
    test_groups = {}
    for result in recent_results:
        if result.test_id not in test_groups:
            test_groups[result.test_id] = []
        test_groups[result.test_id].append(result)
    
    # Analyze each test group and aggregate results
    issues = []
    evidence = []
    recommendations = []
    max_severity = "low"
    
    for test_id, results in test_groups.items():
        test_data = {
            'test': Test.query.get(test_id),
            'results': results,
            'metrics': diagnostic_engine._calculate_metrics_statistics(results),
            'trends': {},
            'anomalies': [],
            'client_info': {'client_count': 1, 'clients': []}
        }
        
        rule_analysis = diagnostic_engine._rule_based_analysis(test_data)
        
        if rule_analysis['issues']:
            issues.extend(rule_analysis['issues'])
            evidence.extend(rule_analysis['evidence'])
            recommendations.extend(rule_analysis['recommendations'])
            
            if rule_analysis['severity'] == "critical":
                max_severity = "critical"
            elif rule_analysis['severity'] == "high" and max_severity != "critical":
                max_severity = "high"
            elif rule_analysis['severity'] == "medium" and max_severity in ["low"]:
                max_severity = "medium"
    
    # Remove duplicates
    issues = list(dict.fromkeys(issues))
    evidence = list(dict.fromkeys(evidence))
    recommendations = list(dict.fromkeys(recommendations))
    
    primary_issue = issues[0] if issues else "Client health appears normal"
    confidence = 70 if issues else 40
    
    return DiagnosticResult(
        confidence=confidence,
        primary_issue=primary_issue,
        evidence=evidence[:10],
        recommendations=recommendations[:8],
        severity=max_severity,
        category="system",
        similar_incidents=[]
    )