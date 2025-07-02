"""
StreamSwarm ML Diagnostics System
Local machine learning models for network performance analysis and troubleshooting
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import zoneinfo
from typing import Dict, List, Tuple, Optional, Any

# Scikit-learn imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
import joblib

from models import TestResult, Test, Client, db

logger = logging.getLogger(__name__)

class NetworkDiagnosticEngine:
    """
    ML-powered network diagnostic and troubleshooting engine
    """
    
    def __init__(self, models_dir='ml_models'):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models from disk"""
        model_files = {
            'anomaly_detector': 'anomaly_detector.joblib',
            'health_classifier': 'health_classifier.joblib',
            'performance_predictor': 'performance_predictor.joblib',
            'scaler': 'feature_scaler.joblib',
            'performance_scaler': 'performance_scaler.joblib',
            'health_encoder': 'health_encoder.joblib'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                try:
                    if model_name == 'scaler':
                        self.scalers['main'] = joblib.load(filepath)
                    elif model_name == 'performance_scaler':
                        self.scalers['performance'] = joblib.load(filepath)
                    elif model_name == 'health_encoder':
                        self.encoders['health'] = joblib.load(filepath)
                    else:
                        self.models[model_name] = joblib.load(filepath)
                    logger.info(f"Loaded {model_name} from {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
        
        # Load feature columns used during training
        feature_columns_path = os.path.join(self.models_dir, "feature_columns.joblib")
        if os.path.exists(feature_columns_path):
            try:
                self.feature_columns = joblib.load(feature_columns_path)
                logger.info(f"Loaded feature columns: {len(self.feature_columns)} features")
            except Exception as e:
                logger.warning(f"Failed to load feature columns: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                filepath = os.path.join(self.models_dir, f"{model_name}.joblib")
                joblib.dump(model, filepath)
                
            for scaler_name, scaler in self.scalers.items():
                if scaler_name == 'main':
                    filepath = os.path.join(self.models_dir, "feature_scaler.joblib")
                else:
                    filepath = os.path.join(self.models_dir, f"{scaler_name}_scaler.joblib")
                joblib.dump(scaler, filepath)
                
            for encoder_name, encoder in self.encoders.items():
                filepath = os.path.join(self.models_dir, f"{encoder_name}_encoder.joblib")
                joblib.dump(encoder, filepath)
            
            # Save feature columns used during training
            if hasattr(self, 'feature_columns'):
                filepath = os.path.join(self.models_dir, "feature_columns.joblib")
                joblib.dump(self.feature_columns, filepath)
                
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def extract_features(self, test_results: List[TestResult]) -> pd.DataFrame:
        """
        Extract ML features from test results
        """
        if not test_results:
            return pd.DataFrame()
            
        features_list = []
        
        for result in test_results:
            features = {
                # Network Performance Features
                'ping_latency': result.ping_latency or 0,
                'ping_packet_loss': result.ping_packet_loss or 0,
                'jitter': result.jitter or 0,
                'bandwidth_download': result.bandwidth_download or 0,
                'bandwidth_upload': result.bandwidth_upload or 0,
                'dns_resolution_time': result.dns_resolution_time or 0,
                'tcp_connect_time': result.tcp_connect_time or 0,
                'ssl_handshake_time': result.ssl_handshake_time or 0,
                'ttfb': result.ttfb or 0,
                
                # System Performance Features
                'cpu_percent': result.cpu_percent or 0,
                'memory_percent': result.memory_percent or 0,
                'disk_percent': result.disk_percent or 0,
                'cpu_load_1min': result.cpu_load_1min or 0,
                'cpu_load_5min': result.cpu_load_5min or 0,
                'cpu_load_15min': result.cpu_load_15min or 0,
                
                # Network Interface Features
                'network_errors_in': result.network_errors_in or 0,
                'network_errors_out': result.network_errors_out or 0,
                'network_drops_in': result.network_drops_in or 0,
                'network_drops_out': result.network_drops_out or 0,
                'tcp_retransmission_rate': result.tcp_retransmission_rate or 0,
                
                # Application & Infrastructure Features
                'content_download_time': result.content_download_time or 0,
                'compression_ratio': result.compression_ratio or 0,
                'power_consumption_watts': result.power_consumption_watts or 0,
                'memory_error_rate': result.memory_error_rate or 0,
                
                # TCP Handshake Timing Features
                'tcp_handshake_syn_time': result.tcp_handshake_syn_time or 0,
                'tcp_handshake_synack_time': result.tcp_handshake_synack_time or 0,
                'tcp_handshake_ack_time': result.tcp_handshake_ack_time or 0,
                'tcp_handshake_total_time': result.tcp_handshake_total_time or 0,
                'tcp_handshake_network_delay': result.tcp_handshake_network_delay or 0,
                'tcp_handshake_server_processing': result.tcp_handshake_server_processing or 0,
                
                # Signal Strength Features (for wireless)
                'signal_strength_avg': result.signal_strength_avg or 0,
                'signal_strength_min': result.signal_strength_min or 0,
                'signal_strength_max': result.signal_strength_max or 0,
                
                # Derived Features
                'latency_jitter_ratio': (
                    (result.jitter or 0) / (result.ping_latency or 1) 
                    if result.ping_latency and result.ping_latency > 0 and result.jitter is not None 
                    else 0
                ),
                'bandwidth_ratio': (
                    (result.bandwidth_upload or 0) / (result.bandwidth_download or 1) 
                    if result.bandwidth_download and result.bandwidth_download > 0 and result.bandwidth_upload is not None 
                    else 0
                ),
                'system_load_avg': (
                    ((result.cpu_load_1min or 0) + (result.cpu_load_5min or 0) + (result.cpu_load_15min or 0)) / 3 
                    if any([result.cpu_load_1min, result.cpu_load_5min, result.cpu_load_15min]) else 0
                ),
                'total_network_errors': (result.network_errors_in or 0) + (result.network_errors_out or 0),
                'total_network_drops': (result.network_drops_in or 0) + (result.network_drops_out or 0),
                
                # TCP Handshake Derived Features
                'handshake_efficiency': (
                    (result.tcp_handshake_network_delay or 0) / (result.tcp_handshake_total_time or 1) 
                    if result.tcp_handshake_total_time and result.tcp_handshake_total_time > 0 
                    else 0
                ),
                'server_processing_ratio': (
                    (result.tcp_handshake_server_processing or 0) / (result.tcp_handshake_total_time or 1) 
                    if result.tcp_handshake_total_time and result.tcp_handshake_total_time > 0 
                    else 0
                ),
                'handshake_overhead': (
                    (result.tcp_handshake_total_time or 0) - (result.tcp_handshake_network_delay or 0) - (result.tcp_handshake_server_processing or 0)
                    if all([result.tcp_handshake_total_time, result.tcp_handshake_network_delay, result.tcp_handshake_server_processing])
                    else 0
                ),
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def calculate_health_score(self, features: pd.DataFrame) -> float:
        """
        Calculate overall health score based on features
        """
        if features.empty:
            return 0.0
            
        # Define weights for different metric categories
        weights = {
            'network': 0.3,
            'system': 0.25,
            'reliability': 0.2,
            'efficiency': 0.1,
            'handshake': 0.15  # New category for TCP handshake performance
        }
        
        # Network health (lower latency, packet loss, jitter is better)
        network_score = 100
        if features['ping_latency'].mean() > 0:
            network_score -= min(features['ping_latency'].mean() / 2, 50)  # Penalize high latency
        if features['ping_packet_loss'].mean() > 0:
            network_score -= min(features['ping_packet_loss'].mean() * 10, 40)  # Penalize packet loss
        if features['jitter'].mean() > 0:
            network_score -= min(features['jitter'].mean() / 2, 10)  # Penalize high jitter
        
        # TCP Handshake Performance (lower times and better efficiency is better)
        handshake_score = 100
        if 'tcp_handshake_total_time' in features.columns and features['tcp_handshake_total_time'].mean() > 0:
            avg_handshake_time = features['tcp_handshake_total_time'].mean()
            if avg_handshake_time > 200:  # Very slow handshakes
                handshake_score -= 40
            elif avg_handshake_time > 100:  # Slow handshakes
                handshake_score -= 25
            elif avg_handshake_time > 50:  # Moderate handshakes
                handshake_score -= 10
            
            # Penalize poor handshake efficiency (high server processing ratio)
            if 'server_processing_ratio' in features.columns:
                avg_server_ratio = features['server_processing_ratio'].mean()
                if avg_server_ratio > 0.7:  # Server processing dominates
                    handshake_score -= 20
                elif avg_server_ratio > 0.5:
                    handshake_score -= 10
            
            # Penalize high handshake overhead
            if 'handshake_overhead' in features.columns:
                avg_overhead = features['handshake_overhead'].mean()
                if avg_overhead > 20:  # High unexplained overhead
                    handshake_score -= 15
                elif avg_overhead > 10:
                    handshake_score -= 8
        
        # System health (lower CPU, memory usage is better)
        system_score = 100
        if features['cpu_percent'].mean() > 80:
            system_score -= 30
        elif features['cpu_percent'].mean() > 60:
            system_score -= 15
        if features['memory_percent'].mean() > 90:
            system_score -= 20
        elif features['memory_percent'].mean() > 75:
            system_score -= 10
        
        # Reliability (lower error rates, retransmissions is better)
        reliability_score = 100
        if features['total_network_errors'].mean() > 0:
            reliability_score -= min(features['total_network_errors'].mean(), 30)
        if features['tcp_retransmission_rate'].mean() > 5:
            reliability_score -= 20
        elif features['tcp_retransmission_rate'].mean() > 2:
            reliability_score -= 10
        
        # Efficiency (good bandwidth utilization, compression)
        efficiency_score = 100
        if features['bandwidth_download'].mean() < 100:  # Low bandwidth
            efficiency_score -= 20
        if features['compression_ratio'].mean() < 20:  # Poor compression
            efficiency_score -= 10
        
        # Calculate weighted average
        total_score = (
            network_score * weights['network'] +
            system_score * weights['system'] +
            reliability_score * weights['reliability'] +
            efficiency_score * weights['efficiency'] +
            handshake_score * weights['handshake']
        )
        
        return max(0, min(100, total_score))
    
    def train_models(self, min_samples=50):
        """
        Train ML models using available test result data
        """
        try:
            logger.info("Starting model training...")
            
            # Get all test results from database
            results = TestResult.query.all()
            
            if len(results) < min_samples:
                logger.warning(f"Not enough data for training. Need at least {min_samples} samples, have {len(results)}")
                return False
            
            # Extract features
            features_df = self.extract_features(results)
            
            if features_df.empty:
                logger.error("No features extracted from test results")
                return False
            
            # Prepare training data - handle NaN and infinite values
            X = features_df.replace([float('inf'), float('-inf')], 0).fillna(0)
            
            # Create health labels based on calculated scores
            health_scores = []
            for i in range(len(X)):
                row = X.iloc[i]
                
                # Calculate health score for individual row
                network_score = 100
                if row['ping_latency'] > 0:
                    network_score -= min(row['ping_latency'] / 2, 50)
                if row['ping_packet_loss'] > 0:
                    network_score -= min(row['ping_packet_loss'] * 10, 40)
                if row['jitter'] > 0:
                    network_score -= min(row['jitter'] / 2, 10)
                
                system_score = 100
                if row['cpu_percent'] > 80:
                    system_score -= 30
                elif row['cpu_percent'] > 60:
                    system_score -= 15
                if row['memory_percent'] > 90:
                    system_score -= 20
                elif row['memory_percent'] > 75:
                    system_score -= 10
                
                reliability_score = 100
                if row['total_network_errors'] > 0:
                    reliability_score -= min(row['total_network_errors'], 30)
                if row['tcp_retransmission_rate'] > 5:
                    reliability_score -= 20
                elif row['tcp_retransmission_rate'] > 2:
                    reliability_score -= 10
                
                efficiency_score = 100
                if row['bandwidth_download'] < 100:
                    efficiency_score -= 20
                if row['compression_ratio'] < 20:
                    efficiency_score -= 10
                
                # Calculate weighted average
                weights = {'network': 0.4, 'system': 0.3, 'reliability': 0.2, 'efficiency': 0.1}
                score = (
                    network_score * weights['network'] +
                    system_score * weights['system'] +
                    reliability_score * weights['reliability'] +
                    efficiency_score * weights['efficiency']
                )
                score = max(0, min(100, score))
                
                if score >= 80:
                    health_scores.append('healthy')
                elif score >= 60:
                    health_scores.append('warning')
                else:
                    health_scores.append('critical')
            
            # Train scaler on the main feature set (including ping_latency)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['main'] = scaler
            
            # Train anomaly detection model (fixed contamination parameter)
            anomaly_model = IsolationForest(contamination='auto', random_state=42)
            anomaly_model.fit(X_scaled)
            self.models['anomaly_detector'] = anomaly_model
            
            # Train health classification model
            if len(set(health_scores)) > 1:  # Only train if we have multiple classes
                health_encoder = LabelEncoder()
                y_encoded = health_encoder.fit_transform(health_scores)
                self.encoders['health'] = health_encoder
                
                # Split data for training
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
                
                health_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                health_classifier.fit(X_train, y_train)
                self.models['health_classifier'] = health_classifier
                
                # Evaluate model
                y_pred = health_classifier.predict(X_test)
                logger.info("Health classification report:")
                logger.info(classification_report(y_test, y_pred))
            
            # Store the feature columns used for training
            self.feature_columns = list(X.columns)
            
            # Train performance predictor (predict latency based on other metrics)
            if 'ping_latency' in features_df.columns:
                # Use other features to predict latency
                feature_cols = [col for col in features_df.columns if col != 'ping_latency']
                X_perf = X[feature_cols].fillna(0)
                y_perf = X['ping_latency'].fillna(0)
                
                if len(X_perf) > 10:  # Minimum samples for regression
                    # Use a separate scaler for performance prediction
                    perf_scaler = StandardScaler()
                    X_perf_scaled = perf_scaler.fit_transform(X_perf)
                    performance_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    performance_predictor.fit(X_perf_scaled, y_perf)
                    self.models['performance_predictor'] = performance_predictor
                    self.scalers['performance'] = perf_scaler
                    # Store the feature columns used for performance prediction
                    # Note: Using self.feature_columns which is already loaded from saved model
            
            # Save trained models
            self._save_models()
            
            logger.info(f"Model training completed with {len(results)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def diagnose_test(self, test_id: int) -> Dict[str, Any]:
        """
        Perform ML-powered diagnosis of test results
        """
        # Get test and results
        test = Test.query.get(test_id)
        if not test:
            return {'error': 'Test not found'}
        
        results = TestResult.query.filter_by(test_id=test_id).all()
        if not results:
            return {'error': 'No results found for this test'}
        
        # Extract features
        features_df = self.extract_features(results)
        if features_df.empty:
            return {'error': 'Could not extract features from test results'}
        
        X = features_df.fillna(0)
        
        # Calculate basic health score
        health_score = self.calculate_health_score(features_df)
        
        diagnosis = {
            'test_id': test_id,
            'test_name': test.name,
            'health_score': round(health_score, 1),
            'health_status': self._get_health_status(health_score),
            'total_measurements': len(results),
            'analysis_timestamp': datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat(),
            'issues_detected': [],
            'recommendations': [],
            'feature_importance': {},
            'anomalies_detected': 0,
            'model_confidence': 0.0
        }
        
        # Run ML analysis if models are available
        if 'main' in self.scalers and X is not None:
            try:
                # Use the same feature columns that were used during training
                if hasattr(self, 'feature_columns') and self.feature_columns:
                    # Reorder columns to match training order and select only training features
                    logger.info(f"Using saved feature columns: {len(self.feature_columns)} features")
                    # Ensure we only use features that exist in both sets
                    available_features = [col for col in self.feature_columns if col in X.columns]
                    X_model = X[available_features]
                    logger.info(f"Filtered to {len(available_features)} available features")
                else:
                    logger.warning("No feature columns found, using all features")
                    X_model = X
                
                X_scaled = self.scalers['main'].transform(X_model)
                
                # Anomaly detection
                if 'anomaly_detector' in self.models:
                    anomaly_scores = self.models['anomaly_detector'].decision_function(X_scaled)
                    anomalies = self.models['anomaly_detector'].predict(X_scaled)
                    anomaly_count = np.sum(anomalies == -1)
                    diagnosis['anomalies_detected'] = int(anomaly_count)
                    
                    # Detailed anomaly analysis
                    anomaly_details = self._analyze_specific_anomalies(
                        anomalies, anomaly_scores, X_model, results, features_df
                    )
                    diagnosis['anomaly_details'] = anomaly_details
                    
                    if anomaly_count > 0:
                        diagnosis['issues_detected'].append({
                            'type': 'anomaly',
                            'severity': 'high' if anomaly_count > len(results) * 0.2 else 'medium',
                            'description': f'Detected {anomaly_count} anomalous measurements with specific issues identified',
                            'recommendation': 'Review detailed anomaly breakdown for targeted troubleshooting'
                        })
                
                # Health classification
                if 'health_classifier' in self.models and 'health' in self.encoders:
                    health_probs = self.models['health_classifier'].predict_proba(X_scaled)
                    health_pred = self.models['health_classifier'].predict(X_scaled)
                    health_labels = self.encoders['health'].inverse_transform(health_pred)
                    
                    # Calculate confidence as average probability of predicted class
                    confidence = np.mean([health_probs[i][pred] for i, pred in enumerate(health_pred)])
                    diagnosis['model_confidence'] = round(confidence * 100, 1)
                    
                    # Feature importance
                    if hasattr(self.models['health_classifier'], 'feature_importances_'):
                        # Use the feature columns that were used during training
                        feature_names = self.feature_columns if hasattr(self, 'feature_columns') else X_model.columns
                        feature_importance = dict(zip(
                            feature_names,
                            self.models['health_classifier'].feature_importances_
                        ))
                        # Sort by importance and get top 5
                        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        diagnosis['feature_importance'] = dict(sorted_importance)
                
            except Exception as e:
                logger.error(f"ML analysis failed: {e}")
                diagnosis['ml_analysis_error'] = str(e)
        
        # Rule-based analysis
        diagnosis.update(self._rule_based_analysis(features_df, results))
        
        return diagnosis
    
    def _get_health_status(self, score: float) -> str:
        """Convert numeric health score to status"""
        if score >= 80:
            return 'healthy'
        elif score >= 60:
            return 'warning'
        else:
            return 'critical'
    
    def _rule_based_analysis(self, features_df: pd.DataFrame, results: List[TestResult]) -> Dict[str, Any]:
        """
        Enhanced rule-based diagnostic analysis with comprehensive root cause analysis
        """
        issues = []
        recommendations = []
        root_cause_analysis = {}
        
        # Core network performance analysis
        avg_latency = features_df['ping_latency'].mean()
        avg_packet_loss = features_df['ping_packet_loss'].mean()
        avg_jitter = features_df['jitter'].mean()
        
        # 1. GEOLOCATION PATH ANALYSIS CORRELATION
        geolocation_insights = self._analyze_geolocation_correlation(results, avg_latency, avg_packet_loss)
        if geolocation_insights:
            issues.extend(geolocation_insights['issues'])
            recommendations.extend(geolocation_insights['recommendations'])
            root_cause_analysis['geolocation_analysis'] = geolocation_insights['analysis']
        
        # 2. GNMI DEVICE PERFORMANCE CORRELATION
        gnmi_insights = self._analyze_gnmi_correlation(results, avg_latency)
        if gnmi_insights:
            issues.extend(gnmi_insights['issues'])
            recommendations.extend(gnmi_insights['recommendations'])
            root_cause_analysis['gnmi_analysis'] = gnmi_insights['analysis']
        
        # 3. APPLICATION-LAYER TIMING BREAKDOWN
        app_layer_insights = self._analyze_application_layer_timing(features_df, results)
        if app_layer_insights:
            issues.extend(app_layer_insights['issues'])
            recommendations.extend(app_layer_insights['recommendations'])
            root_cause_analysis['application_layer'] = app_layer_insights['analysis']
        
        # 4. INFRASTRUCTURE CORRELATION ANALYSIS
        infra_insights = self._analyze_infrastructure_correlation(features_df, avg_latency, avg_packet_loss)
        if infra_insights:
            issues.extend(infra_insights['issues'])
            recommendations.extend(infra_insights['recommendations'])
            root_cause_analysis['infrastructure'] = infra_insights['analysis']
        
        # 5. TEMPORAL PATTERN ANALYSIS
        temporal_insights = self._analyze_temporal_patterns(results, features_df)
        if temporal_insights:
            issues.extend(temporal_insights['issues'])
            recommendations.extend(temporal_insights['recommendations'])
            root_cause_analysis['temporal_patterns'] = temporal_insights['analysis']
        
        # Original network performance rules (enhanced with root cause context)
        if avg_latency > 100:
            severity = 'high' if avg_latency > 200 else 'medium'
            # Determine root cause based on analysis
            root_causes = []
            if geolocation_insights and 'inefficient_routing' in geolocation_insights['analysis']:
                root_causes.append('inefficient geographic routing')
            if gnmi_insights and 'device_bottleneck' in gnmi_insights['analysis']:
                root_causes.append('managed device bottleneck')
            if infra_insights and 'system_correlation' in infra_insights['analysis']:
                root_causes.append('system resource constraints')
            
            description = f'High average latency: {avg_latency:.1f}ms'
            if root_causes:
                description += f' (Root causes: {", ".join(root_causes)})'
            
            issues.append({
                'type': 'high_latency',
                'severity': severity,
                'description': description,
                'recommendation': 'Check network routing and bandwidth utilization',
                'root_causes': root_causes
            })
        
        if avg_packet_loss > 1:
            issues.append({
                'type': 'packet_loss',
                'severity': 'high' if avg_packet_loss > 5 else 'medium',
                'description': f'Packet loss detected: {avg_packet_loss:.1f}%',
                'recommendation': 'Investigate network congestion and equipment health'
            })
        
        if avg_jitter > 20:
            issues.append({
                'type': 'high_jitter',
                'severity': 'medium',
                'description': f'High network jitter: {avg_jitter:.1f}ms',
                'recommendation': 'Consider QoS configuration for time-sensitive applications'
            })
        
        # System performance rules (enhanced with correlation)
        avg_cpu = features_df['cpu_percent'].mean()
        avg_memory = features_df['memory_percent'].mean()
        
        if avg_cpu > 80:
            issues.append({
                'type': 'high_cpu',
                'severity': 'high' if avg_cpu > 90 else 'medium',
                'description': f'High CPU utilization: {avg_cpu:.1f}%',
                'recommendation': 'Review running processes and consider system optimization'
            })
        
        if avg_memory > 85:
            issues.append({
                'type': 'high_memory',
                'severity': 'high' if avg_memory > 95 else 'medium',
                'description': f'High memory utilization: {avg_memory:.1f}%',
                'recommendation': 'Monitor memory usage and consider adding RAM'
            })
        
        # TCP performance rules
        avg_retrans = features_df['tcp_retransmission_rate'].mean()
        if avg_retrans > 3:
            issues.append({
                'type': 'tcp_retransmissions',
                'severity': 'medium',
                'description': f'Elevated TCP retransmission rate: {avg_retrans:.1f}%',
                'recommendation': 'Check network quality and TCP window scaling settings'
            })
        
        # Application performance rules
        avg_download_time = features_df['content_download_time'].mean()
        if avg_download_time > 5000:  # 5 seconds
            issues.append({
                'type': 'slow_downloads',
                'severity': 'medium',
                'description': f'Slow content download times: {avg_download_time:.0f}ms',
                'recommendation': 'Investigate bandwidth limitations and content delivery'
            })
        
        # TCP Handshake performance rules
        if 'tcp_handshake_total_time' in features_df.columns and features_df['tcp_handshake_total_time'].count() > 0:
            avg_handshake_time = features_df['tcp_handshake_total_time'].mean()
            avg_server_processing = features_df['server_processing_ratio'].mean() if 'server_processing_ratio' in features_df.columns else 0
            avg_handshake_overhead = features_df['handshake_overhead'].mean() if 'handshake_overhead' in features_df.columns else 0
            
            if avg_handshake_time > 150:  # Slow handshakes
                issues.append({
                    'type': 'slow_handshake',
                    'severity': 'high' if avg_handshake_time > 300 else 'medium',
                    'description': f'Slow TCP handshake times: {avg_handshake_time:.1f}ms average',
                    'recommendation': 'Check server responsiveness and network path optimization'
                })
            
            if avg_server_processing > 0.6:  # Server processing dominates
                issues.append({
                    'type': 'server_bottleneck',
                    'severity': 'medium',
                    'description': f'Server processing consumes {avg_server_processing*100:.1f}% of handshake time',
                    'recommendation': 'Investigate server load, CPU usage, and application response times'
                })
            
            if avg_handshake_overhead > 15:  # High unexplained overhead
                issues.append({
                    'type': 'handshake_overhead',
                    'severity': 'medium',
                    'description': f'High handshake overhead: {avg_handshake_overhead:.1f}ms unexplained delay',
                    'recommendation': 'Review network configuration, firewall rules, and intermediate proxies'
                })
            
            # Handshake efficiency insights
            if avg_handshake_time > 0:
                efficiency = (features_df['tcp_handshake_network_delay'].mean() if 'tcp_handshake_network_delay' in features_df.columns else 0) / avg_handshake_time
                if efficiency < 0.4:  # Poor handshake efficiency
                    issues.append({
                        'type': 'poor_handshake_efficiency',
                        'severity': 'low',
                        'description': f'Low handshake efficiency: {efficiency*100:.1f}% network vs total time',
                        'recommendation': 'Optimize server configuration and review network path for unnecessary delays'
                    })
        
        # Generate general recommendations
        if not issues:
            recommendations.append('Network performance appears healthy')
        else:
            recommendations.append('Review identified issues for performance optimization')
            if any(issue['severity'] == 'high' for issue in issues):
                recommendations.append('High severity issues detected - immediate attention recommended')
        
        return {
            'issues_detected': issues,
            'recommendations': recommendations,
            'root_cause_analysis': root_cause_analysis
        }

    def _analyze_geolocation_correlation(self, results: List[TestResult], avg_latency: float, avg_packet_loss: float) -> Dict[str, Any]:
        """
        Analyze geolocation path data to identify routing inefficiencies and geographic bottlenecks
        """
        import json
        
        issues = []
        recommendations = []
        analysis = {}
        
        try:
            # Extract geolocation data from test results
            path_analyses = []
            for result in results:
                if hasattr(result, 'path_analysis') and result.path_analysis:
                    try:
                        if isinstance(result.path_analysis, str):
                            path_data = json.loads(result.path_analysis)
                        else:
                            path_data = result.path_analysis
                        path_analyses.append(path_data)
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            if not path_analyses:
                return None
            
            # Analyze geographic efficiency
            total_distance = 0
            total_hops = 0
            inefficient_routes = 0
            countries_traversed = set()
            
            for path_data in path_analyses:
                if 'total_distance_km' in path_data and path_data['total_distance_km']:
                    total_distance += path_data['total_distance_km']
                if 'total_hops' in path_data and path_data['total_hops']:
                    total_hops += path_data['total_hops']
                if 'geographic_efficiency' in path_data and path_data['geographic_efficiency'] < 70:
                    inefficient_routes += 1
                if 'countries_traversed' in path_data:
                    countries_traversed.update(path_data['countries_traversed'])
            
            avg_distance = total_distance / len(path_analyses) if path_analyses else 0
            avg_hops = total_hops / len(path_analyses) if path_analyses else 0
            
            analysis.update({
                'avg_geographic_distance_km': avg_distance,
                'avg_network_hops': avg_hops,
                'countries_traversed': len(countries_traversed),
                'inefficient_routes_percent': (inefficient_routes / len(path_analyses)) * 100 if path_analyses else 0
            })
            
            # Identify inefficient routing patterns
            if inefficient_routes > len(path_analyses) * 0.3:  # 30% of routes are inefficient
                analysis['inefficient_routing'] = True
                issues.append({
                    'type': 'inefficient_routing',
                    'severity': 'medium',
                    'description': f'{inefficient_routes} of {len(path_analyses)} routes show geographic inefficiency',
                    'recommendation': 'Review BGP routing policies and consider traffic engineering'
                })
            
            # Long-distance routing detection
            if avg_distance > 5000:  # > 5000km average
                issues.append({
                    'type': 'long_distance_routing',
                    'severity': 'medium',
                    'description': f'Average routing distance of {avg_distance:.0f}km indicates international paths',
                    'recommendation': 'Consider regional CDN deployment or local caching solutions'
                })
            
            # Excessive hop count
            if avg_hops > 15:
                issues.append({
                    'type': 'excessive_hops',
                    'severity': 'medium',
                    'description': f'High average hop count of {avg_hops:.1f} may indicate suboptimal routing',
                    'recommendation': 'Investigate peering agreements and direct interconnects'
                })
            
            # Multi-country routing
            if len(countries_traversed) > 3:
                issues.append({
                    'type': 'multi_country_routing',
                    'severity': 'low',
                    'description': f'Traffic traverses {len(countries_traversed)} countries, increasing regulatory risk',
                    'recommendation': 'Consider data sovereignty requirements and direct peering'
                })
            
            return {
                'issues': issues,
                'recommendations': recommendations,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Geolocation correlation analysis failed: {e}")
            return None

    def _analyze_gnmi_correlation(self, results: List[TestResult], avg_latency: float) -> Dict[str, Any]:
        """
        Analyze GNMI device performance data to identify managed infrastructure bottlenecks
        """
        import json
        
        issues = []
        recommendations = []
        analysis = {}
        
        try:
            # Extract GNMI analysis data from test results
            gnmi_analyses = []
            for result in results:
                if hasattr(result, 'gnmi_path_analysis') and result.gnmi_path_analysis:
                    try:
                        if isinstance(result.gnmi_path_analysis, str):
                            gnmi_data = json.loads(result.gnmi_path_analysis)
                        else:
                            gnmi_data = result.gnmi_path_analysis
                        gnmi_analyses.append(gnmi_data)
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            if not gnmi_analyses:
                return None
            
            # Analyze managed device performance
            device_metrics = {}
            total_processing_latency = 0
            total_queue_latency = 0
            high_cpu_devices = 0
            high_utilization_devices = 0
            
            for gnmi_data in gnmi_analyses:
                if 'managed_hops' in gnmi_data:
                    for hop in gnmi_data['managed_hops']:
                        device_ip = hop.get('device_ip', 'unknown')
                        
                        if device_ip not in device_metrics:
                            device_metrics[device_ip] = {
                                'processing_latencies': [],
                                'queue_latencies': [],
                                'cpu_utilizations': [],
                                'interface_utilizations': []
                            }
                        
                        # Collect metrics
                        if 'processing_latency_ms' in hop:
                            device_metrics[device_ip]['processing_latencies'].append(hop['processing_latency_ms'])
                            total_processing_latency += hop['processing_latency_ms']
                        
                        if 'queue_latency_ms' in hop:
                            device_metrics[device_ip]['queue_latencies'].append(hop['queue_latency_ms'])
                            total_queue_latency += hop['queue_latency_ms']
                        
                        if 'cpu_utilization_percent' in hop and hop['cpu_utilization_percent'] > 80:
                            high_cpu_devices += 1
                        
                        if 'interface_utilization_percent' in hop and hop['interface_utilization_percent'] > 85:
                            high_utilization_devices += 1
            
            total_devices = len(device_metrics)
            avg_processing_latency = total_processing_latency / total_devices if total_devices > 0 else 0
            avg_queue_latency = total_queue_latency / total_devices if total_devices > 0 else 0
            
            analysis.update({
                'managed_devices_analyzed': total_devices,
                'avg_processing_latency_ms': avg_processing_latency,
                'avg_queue_latency_ms': avg_queue_latency,
                'high_cpu_devices': high_cpu_devices,
                'high_utilization_devices': high_utilization_devices
            })
            
            # Identify device bottlenecks
            if avg_processing_latency > 10:  # > 10ms processing latency
                analysis['device_bottleneck'] = True
                issues.append({
                    'type': 'device_processing_bottleneck',
                    'severity': 'high' if avg_processing_latency > 25 else 'medium',
                    'description': f'High average device processing latency: {avg_processing_latency:.1f}ms',
                    'recommendation': 'Review device CPU utilization and optimize forwarding pipeline'
                })
            
            # Queue bottlenecks
            if avg_queue_latency > 5:  # > 5ms queue latency
                issues.append({
                    'type': 'queue_bottleneck',
                    'severity': 'medium',
                    'description': f'Elevated queue latency: {avg_queue_latency:.1f}ms average',
                    'recommendation': 'Review QoS policies and buffer management configuration'
                })
            
            # Resource utilization issues
            if high_cpu_devices > 0:
                issues.append({
                    'type': 'high_device_cpu',
                    'severity': 'high',
                    'description': f'{high_cpu_devices} managed devices showing high CPU utilization',
                    'recommendation': 'Investigate device workload and consider load balancing'
                })
            
            if high_utilization_devices > 0:
                issues.append({
                    'type': 'high_interface_utilization',
                    'severity': 'medium',
                    'description': f'{high_utilization_devices} interfaces showing high utilization',
                    'recommendation': 'Consider interface upgrades or traffic load balancing'
                })
            
            return {
                'issues': issues,
                'recommendations': recommendations,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"GNMI correlation analysis failed: {e}")
            return None

    def _analyze_application_layer_timing(self, features_df: pd.DataFrame, results: List[TestResult]) -> Dict[str, Any]:
        """
        Analyze application-layer timing breakdown (DNS, SSL handshake, TTFB) for bottleneck identification
        """
        issues = []
        recommendations = []
        analysis = {}
        
        try:
            # DNS resolution timing analysis
            if 'dns_resolution_time' in features_df.columns:
                avg_dns_time = features_df['dns_resolution_time'].mean()
                max_dns_time = features_df['dns_resolution_time'].max()
                
                analysis['avg_dns_resolution_ms'] = avg_dns_time
                analysis['max_dns_resolution_ms'] = max_dns_time
                
                if avg_dns_time > 100:  # > 100ms DNS resolution
                    issues.append({
                        'type': 'slow_dns_resolution',
                        'severity': 'high' if avg_dns_time > 500 else 'medium',
                        'description': f'Slow DNS resolution: {avg_dns_time:.1f}ms average',
                        'recommendation': 'Consider local DNS caching or alternative DNS providers'
                    })
            
            # SSL/TLS handshake timing analysis
            if 'ssl_handshake_time' in features_df.columns:
                avg_ssl_time = features_df['ssl_handshake_time'].mean()
                max_ssl_time = features_df['ssl_handshake_time'].max()
                
                analysis['avg_ssl_handshake_ms'] = avg_ssl_time
                analysis['max_ssl_handshake_ms'] = max_ssl_time
                
                if avg_ssl_time > 200:  # > 200ms SSL handshake
                    issues.append({
                        'type': 'slow_ssl_handshake',
                        'severity': 'high' if avg_ssl_time > 500 else 'medium',
                        'description': f'Slow SSL handshake: {avg_ssl_time:.1f}ms average',
                        'recommendation': 'Review certificate chain length and consider session resumption'
                    })
            
            # Time to First Byte (TTFB) analysis
            if 'ttfb' in features_df.columns:
                avg_ttfb = features_df['ttfb'].mean()
                max_ttfb = features_df['ttfb'].max()
                
                analysis['avg_ttfb_ms'] = avg_ttfb
                analysis['max_ttfb_ms'] = max_ttfb
                
                if avg_ttfb > 500:  # > 500ms TTFB
                    issues.append({
                        'type': 'slow_ttfb',
                        'severity': 'high' if avg_ttfb > 1000 else 'medium',
                        'description': f'Slow time to first byte: {avg_ttfb:.1f}ms average',
                        'recommendation': 'Investigate server response time and database query optimization'
                    })
            
            # TCP connection timing analysis
            if 'tcp_connect_time' in features_df.columns:
                avg_tcp_connect = features_df['tcp_connect_time'].mean()
                
                analysis['avg_tcp_connect_ms'] = avg_tcp_connect
                
                if avg_tcp_connect > 100:  # > 100ms TCP connect
                    issues.append({
                        'type': 'slow_tcp_connect',
                        'severity': 'medium',
                        'description': f'Slow TCP connection: {avg_tcp_connect:.1f}ms average',
                        'recommendation': 'Review network routing and server TCP configuration'
                    })
            
            # Calculate timing breakdown percentages
            total_app_time = 0
            if 'dns_resolution_time' in analysis:
                total_app_time += analysis['avg_dns_resolution_ms']
            if 'ssl_handshake_time' in analysis:
                total_app_time += analysis['avg_ssl_handshake_ms']
            if 'tcp_connect_time' in analysis:
                total_app_time += analysis['avg_tcp_connect_ms']
            
            if total_app_time > 0:
                analysis['application_layer_overhead_ms'] = total_app_time
                
                # Identify dominant component
                if 'avg_dns_resolution_ms' in analysis and analysis['avg_dns_resolution_ms'] / total_app_time > 0.5:
                    analysis['primary_bottleneck'] = 'dns_resolution'
                elif 'avg_ssl_handshake_ms' in analysis and analysis['avg_ssl_handshake_ms'] / total_app_time > 0.5:
                    analysis['primary_bottleneck'] = 'ssl_handshake'
                elif 'avg_tcp_connect_ms' in analysis and analysis['avg_tcp_connect_ms'] / total_app_time > 0.5:
                    analysis['primary_bottleneck'] = 'tcp_connection'
            
            return {
                'issues': issues,
                'recommendations': recommendations,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Application layer timing analysis failed: {e}")
            return None

    def _analyze_infrastructure_correlation(self, features_df: pd.DataFrame, avg_latency: float, avg_packet_loss: float) -> Dict[str, Any]:
        """
        Analyze correlation between infrastructure metrics and network performance
        """
        issues = []
        recommendations = []
        analysis = {}
        
        try:
            # CPU correlation analysis
            if 'cpu_percent' in features_df.columns:
                cpu_corr_latency = features_df['cpu_percent'].corr(features_df['ping_latency'])
                cpu_corr_loss = features_df['cpu_percent'].corr(features_df['ping_packet_loss'])
                
                analysis['cpu_latency_correlation'] = cpu_corr_latency
                analysis['cpu_packet_loss_correlation'] = cpu_corr_loss
                
                if abs(cpu_corr_latency) > 0.6:  # Strong correlation
                    analysis['system_correlation'] = True
                    issues.append({
                        'type': 'cpu_network_correlation',
                        'severity': 'medium',
                        'description': f'Strong correlation ({cpu_corr_latency:.2f}) between CPU usage and latency',
                        'recommendation': 'Investigate system load impact on network stack performance'
                    })
            
            # Memory correlation analysis
            if 'memory_percent' in features_df.columns:
                mem_corr_latency = features_df['memory_percent'].corr(features_df['ping_latency'])
                mem_corr_loss = features_df['memory_percent'].corr(features_df['ping_packet_loss'])
                
                analysis['memory_latency_correlation'] = mem_corr_latency
                analysis['memory_packet_loss_correlation'] = mem_corr_loss
                
                if abs(mem_corr_latency) > 0.6:  # Strong correlation
                    issues.append({
                        'type': 'memory_network_correlation',
                        'severity': 'medium',
                        'description': f'Strong correlation ({mem_corr_latency:.2f}) between memory usage and latency',
                        'recommendation': 'Review memory pressure impact on networking subsystem'
                    })
            
            # Disk I/O correlation analysis
            if 'disk_percent' in features_df.columns:
                disk_corr_latency = features_df['disk_percent'].corr(features_df['ping_latency'])
                
                analysis['disk_latency_correlation'] = disk_corr_latency
                
                if abs(disk_corr_latency) > 0.5:  # Moderate correlation
                    issues.append({
                        'type': 'disk_network_correlation',
                        'severity': 'low',
                        'description': f'Correlation ({disk_corr_latency:.2f}) between disk usage and latency',
                        'recommendation': 'Consider I/O scheduling impact on network performance'
                    })
            
            # Network interface correlation
            if 'total_network_errors' in features_df.columns:
                avg_errors = features_df['total_network_errors'].mean()
                
                analysis['avg_network_errors'] = avg_errors
                
                if avg_errors > 100:  # High error rate
                    issues.append({
                        'type': 'network_interface_errors',
                        'severity': 'high' if avg_errors > 1000 else 'medium',
                        'description': f'High network interface error rate: {avg_errors:.0f} errors/measurement',
                        'recommendation': 'Investigate network driver issues and hardware health'
                    })
            
            # Cross-metric analysis for system bottlenecks
            high_resource_periods = 0
            total_measurements = len(features_df)
            
            for _, row in features_df.iterrows():
                resource_pressure = 0
                if row.get('cpu_percent', 0) > 80:
                    resource_pressure += 1
                if row.get('memory_percent', 0) > 85:
                    resource_pressure += 1
                if row.get('disk_percent', 0) > 90:
                    resource_pressure += 1
                
                if resource_pressure >= 2:  # Multiple resources under pressure
                    high_resource_periods += 1
            
            if high_resource_periods > total_measurements * 0.2:  # > 20% of time
                analysis['system_bottleneck_periods'] = high_resource_periods
                issues.append({
                    'type': 'systemic_resource_pressure',
                    'severity': 'high',
                    'description': f'System under resource pressure {high_resource_periods}/{total_measurements} measurements',
                    'recommendation': 'Consider system upgrade or workload optimization'
                })
            
            return {
                'issues': issues,
                'recommendations': recommendations,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Infrastructure correlation analysis failed: {e}")
            return None

    def _analyze_temporal_patterns(self, results: List[TestResult], features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns in network performance to identify time-based issues
        """
        issues = []
        recommendations = []
        analysis = {}
        
        try:
            from datetime import datetime, timedelta
            
            # Group results by time periods
            hourly_performance = {}
            daily_performance = {}
            
            for result in results:
                if not result.timestamp:
                    continue
                
                hour = result.timestamp.hour
                day = result.timestamp.strftime('%A')
                
                if hour not in hourly_performance:
                    hourly_performance[hour] = {'latencies': [], 'packet_losses': []}
                if day not in daily_performance:
                    daily_performance[day] = {'latencies': [], 'packet_losses': []}
                
                if hasattr(result, 'ping_latency') and result.ping_latency:
                    hourly_performance[hour]['latencies'].append(result.ping_latency)
                    daily_performance[day]['latencies'].append(result.ping_latency)
                
                if hasattr(result, 'ping_packet_loss') and result.ping_packet_loss:
                    hourly_performance[hour]['packet_losses'].append(result.ping_packet_loss)
                    daily_performance[day]['packet_losses'].append(result.ping_packet_loss)
            
            # Analyze hourly patterns
            peak_hours = []
            off_peak_hours = []
            
            for hour, data in hourly_performance.items():
                if data['latencies']:
                    avg_latency = sum(data['latencies']) / len(data['latencies'])
                    
                    if avg_latency > features_df['ping_latency'].mean() * 1.5:  # 50% above average
                        peak_hours.append(hour)
                    elif avg_latency < features_df['ping_latency'].mean() * 0.7:  # 30% below average
                        off_peak_hours.append(hour)
            
            analysis['peak_performance_hours'] = sorted(off_peak_hours)
            analysis['poor_performance_hours'] = sorted(peak_hours)
            
            if peak_hours:
                issues.append({
                    'type': 'peak_hour_degradation',
                    'severity': 'medium',
                    'description': f'Performance degradation during hours: {sorted(peak_hours)}',
                    'recommendation': 'Investigate traffic patterns and consider load balancing during peak hours'
                })
            
            # Analyze weekly patterns
            weekend_days = ['Saturday', 'Sunday']
            weekday_latencies = []
            weekend_latencies = []
            
            for day, data in daily_performance.items():
                if data['latencies']:
                    avg_latency = sum(data['latencies']) / len(data['latencies'])
                    
                    if day in weekend_days:
                        weekend_latencies.append(avg_latency)
                    else:
                        weekday_latencies.append(avg_latency)
            
            if weekday_latencies and weekend_latencies:
                weekday_avg = sum(weekday_latencies) / len(weekday_latencies)
                weekend_avg = sum(weekend_latencies) / len(weekend_latencies)
                
                analysis['weekday_avg_latency'] = weekday_avg
                analysis['weekend_avg_latency'] = weekend_avg
                
                if weekday_avg > weekend_avg * 1.3:  # 30% higher on weekdays
                    issues.append({
                        'type': 'weekday_performance_impact',
                        'severity': 'low',
                        'description': f'Weekday performance {(weekday_avg/weekend_avg-1)*100:.1f}% worse than weekends',
                        'recommendation': 'Consider business hour traffic management and capacity planning'
                    })
            
            # Analyze performance trends over time
            if len(results) >= 10:
                # Split into early and late periods
                sorted_results = sorted(results, key=lambda x: x.timestamp if x.timestamp else datetime.min)
                mid_point = len(sorted_results) // 2
                
                early_latencies = []
                late_latencies = []
                
                for i, result in enumerate(sorted_results):
                    if hasattr(result, 'ping_latency') and result.ping_latency:
                        if i < mid_point:
                            early_latencies.append(result.ping_latency)
                        else:
                            late_latencies.append(result.ping_latency)
                
                if early_latencies and late_latencies:
                    early_avg = sum(early_latencies) / len(early_latencies)
                    late_avg = sum(late_latencies) / len(late_latencies)
                    
                    trend_change = (late_avg - early_avg) / early_avg * 100
                    analysis['performance_trend_percent'] = trend_change
                    
                    if abs(trend_change) > 20:  # > 20% change
                        severity = 'high' if abs(trend_change) > 50 else 'medium'
                        direction = 'degraded' if trend_change > 0 else 'improved'
                        
                        issues.append({
                            'type': 'performance_trend',
                            'severity': severity,
                            'description': f'Performance has {direction} by {abs(trend_change):.1f}% over time',
                            'recommendation': 'Investigate long-term performance trends and capacity planning'
                        })
            
            return {
                'issues': issues,
                'recommendations': recommendations,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return None
    
    def _analyze_specific_anomalies(self, anomalies, anomaly_scores, features_df, results, original_features_df):
        """
        Analyze specific anomalies to provide detailed, actionable insights
        """
        anomaly_details = []
        
        # Find anomalous measurements (where prediction == -1)
        anomalous_indices = np.where(anomalies == -1)[0]
        
        if len(anomalous_indices) == 0:
            return []
        
        # Calculate feature statistics for comparison
        feature_stats = {}
        for col in features_df.columns:
            if col in original_features_df.columns:
                feature_stats[col] = {
                    'mean': original_features_df[col].mean(),
                    'std': original_features_df[col].std(),
                    'median': original_features_df[col].median(),
                    'p95': original_features_df[col].quantile(0.95)
                }
        
        # Key metrics to focus on for anomaly identification
        critical_metrics = {
            'ping_packet_loss': {'threshold': 1.0, 'unit': '%', 'type': 'network'},
            'ping_latency': {'threshold': 100.0, 'unit': 'ms', 'type': 'network'},
            'jitter': {'threshold': 20.0, 'unit': 'ms', 'type': 'network'},
            'cpu_percent': {'threshold': 80.0, 'unit': '%', 'type': 'system'},
            'memory_percent': {'threshold': 85.0, 'unit': '%', 'type': 'system'},
            'tcp_handshake_total_time': {'threshold': 150.0, 'unit': 'ms', 'type': 'tcp'},
            'tcp_retransmission_rate': {'threshold': 3.0, 'unit': '%', 'type': 'tcp'},
            'bandwidth_download': {'threshold': 100.0, 'unit': 'Mbps', 'type': 'bandwidth', 'direction': 'low'},
            'content_download_time': {'threshold': 5000.0, 'unit': 'ms', 'type': 'application'}
        }
        
        # Analyze each anomalous measurement
        for idx in anomalous_indices:
            if idx >= len(results):
                continue
                
            result = results[idx]
            anomaly_score = anomaly_scores[idx]
            timestamp = result.timestamp.strftime('%H:%M:%S') if result.timestamp else f"Measurement {idx+1}"
            
            # Find which metrics are problematic in this measurement
            issues_found = []
            
            for metric, config in critical_metrics.items():
                if hasattr(result, metric):
                    value = getattr(result, metric)
                    if value is not None:
                        # Check if value exceeds threshold
                        is_anomalous = False
                        severity = 'medium'
                        
                        if config.get('direction') == 'low':
                            # For bandwidth, low values are bad
                            if value < config['threshold']:
                                is_anomalous = True
                                severity = 'high' if value < config['threshold'] * 0.5 else 'medium'
                        else:
                            # For most metrics, high values are bad
                            if value > config['threshold']:
                                is_anomalous = True
                                severity = 'high' if value > config['threshold'] * 2 else 'medium'
                        
                        # Also check against statistical deviation
                        if metric in feature_stats and not is_anomalous:
                            stats = feature_stats[metric]
                            if not pd.isna(stats['mean']) and not pd.isna(stats['std']):
                                # More than 3 standard deviations from mean
                                if abs(value - stats['mean']) > 3 * stats['std']:
                                    is_anomalous = True
                                    severity = 'medium'
                        
                        if is_anomalous:
                            # Format value appropriately
                            if metric in ['ping_packet_loss', 'cpu_percent', 'memory_percent', 'tcp_retransmission_rate']:
                                formatted_value = f"{value:.1f}{config['unit']}"
                            elif metric in ['bandwidth_download', 'bandwidth_upload']:
                                formatted_value = f"{value:.0f} {config['unit']}"
                            else:
                                formatted_value = f"{value:.1f}{config['unit']}"
                            
                            issues_found.append({
                                'metric': metric,
                                'value': formatted_value,
                                'type': config['type'],
                                'severity': severity,
                                'raw_value': value
                            })
            
            # Special checks for missing data
            if not hasattr(result, 'bandwidth_download') or result.bandwidth_download is None:
                issues_found.append({
                    'metric': 'bandwidth_download',
                    'value': 'Missing',
                    'type': 'connectivity',
                    'severity': 'medium',
                    'raw_value': None
                })
            
            # Create anomaly detail entry if issues found
            if issues_found:
                # Sort by severity and type
                issues_found.sort(key=lambda x: (x['severity'] == 'high', x['type']))
                
                # Create description based on most severe issues
                primary_issues = [issue for issue in issues_found if issue['severity'] == 'high']
                if not primary_issues:
                    primary_issues = issues_found[:2]  # Take top 2 if no high severity
                
                descriptions = []
                for issue in primary_issues:
                    metric_name = issue['metric'].replace('_', ' ').title()
                    if issue['raw_value'] is None:
                        descriptions.append(f"{metric_name}: {issue['value']}")
                    else:
                        descriptions.append(f"{metric_name}: {issue['value']}")
                
                anomaly_details.append({
                    'timestamp': timestamp,
                    'measurement_id': result.id if hasattr(result, 'id') else None,
                    'anomaly_score': round(abs(anomaly_score), 3),
                    'severity': 'high' if any(i['severity'] == 'high' for i in issues_found) else 'medium',
                    'description': '; '.join(descriptions),
                    'issues': issues_found,
                    'recommendations': self._get_anomaly_recommendations(issues_found)
                })
        
        # Sort by severity and anomaly score
        anomaly_details.sort(key=lambda x: (x['severity'] == 'high', x['anomaly_score']), reverse=True)
        
        # Limit to top 15 most significant anomalies to avoid overwhelming the UI
        return anomaly_details[:15]
    
    def _get_anomaly_recommendations(self, issues):
        """Generate specific recommendations based on anomaly types"""
        recommendations = []
        issue_types = set(issue['type'] for issue in issues)
        
        if 'network' in issue_types:
            recommendations.append("Check network routing and bandwidth utilization")
        if 'system' in issue_types:
            recommendations.append("Review system resource usage and running processes")
        if 'tcp' in issue_types:
            recommendations.append("Investigate TCP configuration and server responsiveness")
        if 'bandwidth' in issue_types:
            recommendations.append("Verify internet connection and ISP performance")
        if 'connectivity' in issue_types:
            recommendations.append("Check network connectivity and firewall rules")
        if 'application' in issue_types:
            recommendations.append("Review application server performance and content delivery")
        
        return recommendations
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current status of ML models
        """
        status = {
            'models_available': list(self.models.keys()),
            'scalers_available': list(self.scalers.keys()),
            'encoders_available': list(self.encoders.keys()),
            'models_trained': len(self.models) > 0,
            'total_training_samples': 0,
            'last_training': None
        }
        
        # Get training data count
        total_results = TestResult.query.count()
        status['total_training_samples'] = total_results
        
        # Check for model files and get timestamps
        model_files = ['anomaly_model.joblib', 'health_classifier.joblib', 'performance_predictor.joblib']
        for filename in model_files:
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                mtime = os.path.getmtime(filepath)
                last_modified = datetime.fromtimestamp(mtime)
                # Compare with existing last_training timestamp
                should_update = status['last_training'] is None
                if not should_update and status['last_training']:
                    try:
                        existing_training = datetime.fromisoformat(status['last_training'])
                        should_update = last_modified > existing_training
                    except (ValueError, TypeError):
                        should_update = True
                
                if should_update:
                    status['last_training'] = last_modified.isoformat()
        
        return status

    def predict_performance(self, test_config: Dict[str, Any], current_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict network performance for a given configuration before running tests
        """
        try:
            if 'performance_predictor' not in self.models:
                logger.warning("Performance prediction model not available, falling back to rule-based prediction")
                return self._rule_based_prediction(test_config, current_conditions)
            
            # Create feature vector for prediction
            prediction_features = self._create_prediction_features(test_config, current_conditions)
            logger.info(f"Created feature vector with {len(prediction_features) if prediction_features else 0} features for config: {test_config}")
            
            if prediction_features is None:
                logger.warning("Unable to create feature vector, falling back to rule-based prediction")
                return self._rule_based_prediction(test_config, current_conditions)
            
            # Scale features using the performance scaler
            if 'performance' not in self.scalers:
                logger.warning("Performance scaler not available, falling back to rule-based prediction")
                return self._rule_based_prediction(test_config, current_conditions)
            
            try:
                features_scaled = self.scalers['performance'].transform([prediction_features])
                
                # Make prediction using ML model
                ml_predicted_latency = self.models['performance_predictor'].predict(features_scaled)[0]
                logger.info(f"ML model raw prediction: {ml_predicted_latency:.2f}ms")
                
                # Enhance ML prediction with parameter-based adjustments for better accuracy
                enhanced_prediction = self._enhance_ml_prediction(ml_predicted_latency, test_config, current_conditions)
                predicted_latency = enhanced_prediction
                
                logger.info(f"Enhanced ML prediction: {predicted_latency:.2f}ms for {test_config.get('destination', 'unknown')}")
                
            except Exception as e:
                logger.warning(f"ML model prediction failed ({str(e)}), falling back to rule-based prediction")
                return self._rule_based_prediction(test_config, current_conditions)
            
            # Get feature importance for explanation
            feature_importance = {}
            if hasattr(self.models['performance_predictor'], 'feature_importances_'):
                importance_values = self.models['performance_predictor'].feature_importances_
                for i, col in enumerate(self.feature_columns):
                    if i < len(importance_values):
                        feature_importance[col] = float(importance_values[i])
            
            # Calculate confidence based on historical performance
            confidence = self._calculate_prediction_confidence(prediction_features, predicted_latency)
            
            # Generate performance insights
            insights = self._generate_performance_insights(predicted_latency, test_config, current_conditions)
            
            return {
                'predicted_latency_ms': float(predicted_latency),
                'confidence_score': confidence,
                'performance_category': self._categorize_predicted_performance(predicted_latency),
                'insights': insights,
                'feature_importance': feature_importance,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error predicting performance: {str(e)}")
            return {
                'error': str(e),
                'status': 'prediction_failed'
            }
    
    def _rule_based_prediction(self, test_config: Dict[str, Any], current_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fallback rule-based prediction when ML model is not available or fails
        """
        try:
            # Calculate baseline latency based on inputs
            destination = test_config.get('destination', 'google.com').lower()
            packet_size = test_config.get('packet_size', 64)
            test_type = test_config.get('test_type', 'Basic Latency')
            test_count = test_config.get('test_count', 10)
            
            # Start with a reasonable baseline
            baseline_latency = 20.0
            
            # Destination-based adjustment
            if 'google.com' in destination or 'cloudflare.com' in destination:
                baseline_latency *= 0.8  # 16ms
                insights_text = "Excellent connectivity to major CDN"
            elif 'github.com' in destination or 'microsoft.com' in destination:
                baseline_latency *= 0.9  # 18ms
                insights_text = "Good connectivity to major tech services"
            elif 'spiegel.de' in destination or any(tld in destination for tld in ['.de', '.eu', '.uk']):
                baseline_latency *= 1.4  # 28ms
                insights_text = "International destination - expect higher latency"
            elif any(tld in destination for tld in ['.asia', '.jp', '.cn', '.au']):
                baseline_latency *= 2.0  # 40ms
                insights_text = "Very distant destination - significant latency expected"
            else:
                baseline_latency *= 1.1  # 22ms
                insights_text = "Standard latency for unknown destination"
            
            # Test type adjustment
            if test_type == 'Bandwidth Focus':
                baseline_latency *= 1.2
                insights_text += " - Bandwidth testing may show higher latency due to congestion"
            elif test_type == 'Comprehensive':
                baseline_latency *= 1.1
                insights_text += " - Comprehensive testing provides thorough analysis"
            
            # Packet size adjustment
            if packet_size > 64:
                size_increase = (packet_size - 64) * 0.02
                baseline_latency += size_increase
                insights_text += f" - Larger packet size ({packet_size}B) adds ~{size_increase:.1f}ms"
            
            # Test count adjustment (more tests = slightly more accurate but longer)
            if test_count > 10:
                insights_text += f" - {test_count} tests provide higher accuracy"
            
            # Determine performance category
            if baseline_latency <= 20:
                category = "Excellent"
                confidence = 75
            elif baseline_latency <= 35:
                category = "Good"
                confidence = 70
            elif baseline_latency <= 60:
                category = "Fair"
                confidence = 65
            else:
                category = "Poor"
                confidence = 60
            
            logger.info(f"Rule-based prediction: {baseline_latency:.1f}ms for {destination}, {test_type}, {packet_size}B")
            
            return {
                'predicted_latency_ms': round(baseline_latency, 1),
                'confidence_score': confidence,
                'performance_category': category,
                'insights': [insights_text],
                'feature_importance': {},
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based prediction: {str(e)}")
            return {
                'error': str(e),
                'status': 'prediction_failed'
            }
    
    def _enhance_ml_prediction(self, ml_prediction: float, test_config: Dict[str, Any], current_conditions: Dict[str, Any] = None) -> float:
        """
        Enhance ML model prediction with parameter-based adjustments for better accuracy
        """
        try:
            # If ML prediction seems unrealistic (like the constant 6ms issue), apply corrections
            if ml_prediction < 5 or ml_prediction > 1000:
                logger.warning(f"ML prediction {ml_prediction:.2f}ms seems unrealistic, applying corrections")
                # Use rule-based calculation as baseline and blend with ML
                rule_result = self._rule_based_prediction(test_config, current_conditions)
                rule_latency = rule_result.get('predicted_latency_ms', 20.0)
                # Blend 70% rule-based, 30% ML if ML seems off
                enhanced_prediction = (rule_latency * 0.7) + (ml_prediction * 0.3)
            else:
                # ML prediction seems reasonable, apply parameter-based adjustments
                enhanced_prediction = ml_prediction
                
                destination = test_config.get('destination', 'google.com').lower()
                packet_size = test_config.get('packet_size', 64)
                test_type = test_config.get('test_type', 'Basic Latency')
                
                # Apply destination-based adjustment factor
                if 'google.com' in destination or 'cloudflare.com' in destination:
                    enhanced_prediction *= 0.9  # CDNs are typically faster
                elif 'spiegel.de' in destination or any(tld in destination for tld in ['.de', '.eu', '.uk']):
                    enhanced_prediction *= 1.2  # International destinations
                elif any(tld in destination for tld in ['.asia', '.jp', '.cn', '.au']):
                    enhanced_prediction *= 1.5  # Very distant destinations
                
                # Apply test type adjustment
                if test_type == 'Bandwidth Focus':
                    enhanced_prediction *= 1.1  # Bandwidth tests may show higher latency
                
                # Apply packet size adjustment
                if packet_size > 64:
                    size_factor = 1.0 + ((packet_size - 64) / 1000)
                    enhanced_prediction *= size_factor
            
            return max(enhanced_prediction, 1.0)  # Ensure minimum 1ms latency
            
        except Exception as e:
            logger.error(f"Error enhancing ML prediction: {str(e)}")
            return ml_prediction  # Return original prediction if enhancement fails
    
    def _generate_synthetic_training_data(self, num_samples: int = 500) -> List:
        """
        Generate synthetic training data to improve model coverage of different scenarios
        """
        import random
        from datetime import datetime, timedelta
        
        synthetic_results = []
        destinations = ['google.com', 'cloudflare.com', 'github.com', 'microsoft.com', 'spiegel.de', 
                       'bbc.co.uk', 'amazon.com', 'facebook.com', 'twitter.com', 'youtube.com',
                       'reddit.com', 'wikipedia.org', 'example.com']
        
        for i in range(num_samples):
            # Create a mock TestResult object with realistic values
            class MockTestResult:
                def __init__(self):
                    destination = random.choice(destinations)
                    
                    # Base latency varies by destination
                    if destination in ['google.com', 'cloudflare.com']:
                        base_latency = random.uniform(8, 20)
                    elif destination in ['github.com', 'microsoft.com', 'amazon.com']:
                        base_latency = random.uniform(15, 35)
                    elif destination in ['spiegel.de', 'bbc.co.uk']:
                        base_latency = random.uniform(25, 50)
                    else:
                        base_latency = random.uniform(18, 40)
                    
                    # Add some noise and packet size impact
                    packet_size = random.choice([64, 128, 256, 512, 1024])
                    size_factor = 1.0 + ((packet_size - 64) / 1000)
                    
                    self.destination = destination
                    self.ping_latency = base_latency * size_factor
                    self.ping_packet_loss = random.uniform(0, 3)  # 0-3% loss
                    self.jitter = random.uniform(0.5, self.ping_latency * 0.2)
                    self.bandwidth_download = random.uniform(50, 500)
                    self.bandwidth_upload = random.uniform(20, 100)
                    self.dns_resolution_time = random.uniform(1, 15)
                    self.tcp_connect_time = random.uniform(2, 25)
                    self.ssl_handshake_time = random.uniform(5, 50)
                    self.ttfb = random.uniform(10, 100)
                    
                    # System metrics
                    self.cpu_percent = random.uniform(10, 80)
                    self.memory_percent = random.uniform(30, 90)
                    self.disk_percent = random.uniform(20, 85)
                    self.network_bytes_sent = random.randint(1000, 100000)
                    self.network_bytes_recv = random.randint(1000, 100000)
                    
                    # QoS metrics
                    self.dscp = random.choice([0, 8, 16, 24, 32, 40, 46])
                    self.cos = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
                    
                    # Test configuration
                    self.test_type = random.choice(['ping', 'comprehensive', 'bandwidth'])
                    self.packet_size = packet_size
                    self.test_count = random.choice([10, 20, 50, 100])
                    
                    # Network interface
                    self.interface_name = random.choice(['eth0', 'wlan0', 'enp0s3'])
                    self.mtu = random.choice([1500, 9000])
                    
                    # Default other fields to avoid attribute errors
                    for attr in ['signal_strength', 'connection_quality', 'tcp_retransmissions',
                               'compression_ratio', 'error_rate', 'timestamp']:
                        if not hasattr(self, attr):
                            if attr == 'timestamp':
                                setattr(self, attr, datetime.now() - timedelta(days=random.randint(0, 30)))
                            elif attr in ['signal_strength', 'connection_quality']:
                                setattr(self, attr, random.uniform(-80, -20) if attr == 'signal_strength' else random.uniform(50, 100))
                            else:
                                setattr(self, attr, random.uniform(0, 10))
            
            synthetic_results.append(MockTestResult())
        
        return synthetic_results
    
    def _create_performance_features(self, X):
        """
        Create enhanced features for performance prediction
        """
        # Start with original features excluding ping_latency
        feature_cols = [col for col in X.columns if col != 'ping_latency']
        X_perf = X[feature_cols].copy()
        
        # Add engineered features that could influence latency
        if 'bandwidth_download' in X_perf.columns and 'bandwidth_upload' in X_perf.columns:
            X_perf['bandwidth_ratio'] = X_perf['bandwidth_download'] / (X_perf['bandwidth_upload'] + 1)
        
        if 'cpu_percent' in X_perf.columns and 'memory_percent' in X_perf.columns:
            X_perf['system_load'] = (X_perf['cpu_percent'] + X_perf['memory_percent']) / 2
        
        if 'packet_size' in X_perf.columns:
            X_perf['packet_size_normalized'] = X_perf['packet_size'] / 1500  # Normalize by MTU
        
        if 'dns_resolution_time' in X_perf.columns and 'tcp_connect_time' in X_perf.columns:
            X_perf['connection_overhead'] = X_perf['dns_resolution_time'] + X_perf['tcp_connect_time']
        
        return X_perf.fillna(0)
    
    def _create_prediction_features(self, test_config: Dict[str, Any], current_conditions: Dict[str, Any] = None) -> List[float]:
        """
        Create feature vector for performance prediction based on test configuration and current conditions
        """
        try:
            # Get recent test results for baseline features
            recent_results = TestResult.query.order_by(TestResult.timestamp.desc()).limit(10).all()
            
            if not recent_results:
                return None
            
            # Extract baseline features from recent results
            baseline_features = self.extract_features(recent_results)
            if baseline_features.empty:
                return None
            
            # Use mean of recent results as baseline
            feature_vector = baseline_features.mean().to_dict()
            
            # Override with current conditions if provided
            if current_conditions:
                for key, value in current_conditions.items():
                    if key in feature_vector:
                        feature_vector[key] = value
            
            # Override with test-specific parameters and map test type to expected performance
            if 'packet_size' in test_config:
                feature_vector['packet_size'] = test_config['packet_size']
            
            if 'test_count' in test_config:
                feature_vector['ping_count'] = test_config['test_count']
            
            # Apply destination-based adjustments
            destination = test_config.get('destination', 'google.com').lower()
            base_latency_adjustment = 1.0
            
            # Estimate latency based on destination
            if 'google.com' in destination or 'cloudflare.com' in destination:
                base_latency_adjustment = 0.8  # Lower latency for major CDNs
            elif 'github.com' in destination or 'microsoft.com' in destination:
                base_latency_adjustment = 0.9  # Good connectivity
            elif 'spiegel.de' in destination or any(tld in destination for tld in ['.de', '.eu', '.uk']):
                base_latency_adjustment = 1.3  # International destinations typically higher
            elif any(tld in destination for tld in ['.asia', '.jp', '.cn', '.au']):
                base_latency_adjustment = 1.8  # Very distant destinations
            else:
                base_latency_adjustment = 1.1  # Default slight increase for unknown destinations
            
            # Apply destination adjustment to baseline latency
            current_latency = feature_vector.get('ping_latency', 25)
            feature_vector['ping_latency'] = current_latency * base_latency_adjustment
            
            # Apply test type influence on prediction features
            test_type = test_config.get('test_type', 'Basic Latency')
            if test_type == 'Bandwidth Focus':
                # Bandwidth tests typically show different latency characteristics
                feature_vector['bandwidth_download'] = feature_vector.get('bandwidth_download', 100) * 1.2
                feature_vector['bandwidth_upload'] = feature_vector.get('bandwidth_upload', 50) * 1.2
                # Bandwidth tests might have slightly higher latency due to congestion
                feature_vector['ping_latency'] = feature_vector.get('ping_latency', 20) * 1.15
            elif test_type == 'Comprehensive':
                # Comprehensive tests are more thorough, might show more realistic latency
                feature_vector['ping_latency'] = feature_vector.get('ping_latency', 20) * 1.08
            
            # Apply packet size influence
            packet_size = test_config.get('packet_size', 64)
            if packet_size > 64:
                # Larger packets typically increase latency slightly
                size_factor = 1.0 + ((packet_size - 64) / 1000)  # Small increase per extra byte
                feature_vector['ping_latency'] = feature_vector.get('ping_latency', 20) * size_factor
            
            # Create ordered feature list matching training features
            if not hasattr(self, 'feature_columns') or not self.feature_columns:
                logger.warning("Feature columns not available for prediction")
                # Instead of returning all zeros, create a more realistic baseline
                destination = test_config.get('destination', 'google.com').lower()
                packet_size = test_config.get('packet_size', 64)
                test_type = test_config.get('test_type', 'Basic Latency')
                
                # Calculate baseline latency based on inputs
                baseline_latency = 20.0  # Start with reasonable baseline
                
                # Destination-based adjustment
                if 'google.com' in destination or 'cloudflare.com' in destination:
                    baseline_latency *= 0.8
                elif 'spiegel.de' in destination or any(tld in destination for tld in ['.de', '.eu', '.uk']):
                    baseline_latency *= 1.4
                elif any(tld in destination for tld in ['.asia', '.jp', '.cn', '.au']):
                    baseline_latency *= 2.0
                
                # Test type adjustment
                if test_type == 'Bandwidth Focus':
                    baseline_latency *= 1.2
                elif test_type == 'Comprehensive':
                    baseline_latency *= 1.1
                
                # Packet size adjustment
                baseline_latency += (packet_size - 64) * 0.05
                
                logger.info(f"Using fallback prediction: {baseline_latency:.1f}ms for {destination}, {test_type}, {packet_size}B")
                return [baseline_latency] + [0.0] * 39
            
            ordered_features = []
            for col in self.feature_columns:
                if col in feature_vector:
                    ordered_features.append(feature_vector[col])
                else:
                    ordered_features.append(0.0)  # Default value for missing features
            
            # Ensure feature vector has exactly 40 features (expected by trained model)
            if len(ordered_features) > 40:
                ordered_features = ordered_features[:40]  # Truncate to 40
            elif len(ordered_features) < 40:
                ordered_features.extend([0.0] * (40 - len(ordered_features)))  # Pad to 40
            
            return ordered_features
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {str(e)}")
            return None
    
    def _calculate_prediction_confidence(self, features: List[float], predicted_value: float) -> float:
        """
        Calculate confidence score for the prediction based on historical accuracy
        """
        try:
            # Get recent results for validation
            recent_results = TestResult.query.order_by(TestResult.timestamp.desc()).limit(50).all()
            
            if len(recent_results) < 10:
                return 0.5  # Low confidence with insufficient data
            
            # Calculate historical prediction accuracy
            recent_features = self.extract_features(recent_results)
            if recent_features.empty or 'ping_latency' not in recent_features.columns:
                return 0.5
            
            actual_latencies = recent_features['ping_latency'].dropna()
            
            if len(actual_latencies) < 5:
                return 0.5
            
            # Compare predicted value to recent actual values
            recent_mean = actual_latencies.mean()
            recent_std = actual_latencies.std()
            
            if recent_std == 0:
                return 0.7  # Stable performance
            
            # Calculate confidence based on how close prediction is to recent patterns
            z_score = abs(predicted_value - recent_mean) / recent_std
            
            if z_score < 1:
                confidence = 0.9
            elif z_score < 2:
                confidence = 0.7
            elif z_score < 3:
                confidence = 0.5
            else:
                confidence = 0.3
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5
    
    def _categorize_predicted_performance(self, predicted_latency: float) -> str:
        """
        Categorize predicted performance into human-readable categories
        """
        if predicted_latency < 20:
            return 'excellent'
        elif predicted_latency < 50:
            return 'good'
        elif predicted_latency < 100:
            return 'fair'
        elif predicted_latency < 200:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_performance_insights(self, predicted_latency: float, test_config: Dict[str, Any], current_conditions: Dict[str, Any] = None) -> List[str]:
        """
        Generate actionable insights based on predicted performance
        """
        insights = []
        
        # Latency-based insights
        if predicted_latency > 200:
            insights.append("High latency predicted - consider running tests during off-peak hours")
            insights.append("Network congestion may be affecting performance")
        elif predicted_latency > 100:
            insights.append("Moderate latency expected - monitor for performance variations")
        elif predicted_latency < 20:
            insights.append("Excellent performance predicted - ideal time for comprehensive testing")
        
        # Configuration-based insights
        if test_config.get('packet_size', 64) > 1000:
            insights.append("Large packet size may increase latency - consider testing with standard sizes first")
        
        # Current conditions insights
        if current_conditions:
            if current_conditions.get('cpu_percent', 0) > 80:
                insights.append("High CPU usage detected - may impact test accuracy")
            
            if current_conditions.get('memory_percent', 0) > 85:
                insights.append("High memory usage detected - consider freeing resources before testing")
            
            if current_conditions.get('bandwidth_download', 0) < 100:
                insights.append("Limited bandwidth detected - network capacity may be constrained")
        
        return insights
    
    def analyze_capacity_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze network capacity trends for capacity planning
        """
        try:
            # Get test results from the specified time period
            cutoff_date = datetime.now(zoneinfo.ZoneInfo("UTC")) - timedelta(days=days_back)
            results = TestResult.query.filter(TestResult.timestamp >= cutoff_date).all()
            
            if len(results) < 10:
                return {
                    'error': 'Insufficient data for trend analysis',
                    'status': 'insufficient_data'
                }
            
            # Extract features for analysis
            features_df = self.extract_features(results)
            if features_df.empty:
                return {
                    'error': 'No features extracted for trend analysis',
                    'status': 'feature_extraction_failed'
                }
            
            # Analyze key performance trends
            trends = {}
            
            # Latency trends
            if 'ping_latency' in features_df.columns:
                latency_trend = self._calculate_trend(features_df['ping_latency'].dropna())
                trends['latency'] = latency_trend
            
            # Bandwidth trends
            if 'bandwidth_download' in features_df.columns:
                bandwidth_trend = self._calculate_trend(features_df['bandwidth_download'].dropna())
                trends['bandwidth'] = bandwidth_trend
            
            # Packet loss trends
            if 'ping_packet_loss' in features_df.columns:
                loss_trend = self._calculate_trend(features_df['ping_packet_loss'].dropna())
                trends['packet_loss'] = loss_trend
            
            # Generate capacity planning recommendations
            recommendations = self._generate_capacity_recommendations(trends, features_df)
            
            # Calculate overall network health trend
            health_trend = self._calculate_overall_health_trend(features_df)
            
            return {
                'trends': trends,
                'recommendations': recommendations,
                'health_trend': health_trend,
                'analysis_period_days': days_back,
                'sample_count': len(results),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing capacity trends: {str(e)}")
            return {
                'error': str(e),
                'status': 'analysis_failed'
            }
    
    def _calculate_trend(self, data_series: pd.Series) -> Dict[str, Any]:
        """
        Calculate trend statistics for a data series
        """
        if len(data_series) < 2:
            return {'trend': 'insufficient_data', 'change_percent': 0}
        
        # Simple linear trend calculation
        x = np.arange(len(data_series))
        y = data_series.values
        
        # Calculate slope using least squares
        slope = np.polyfit(x, y, 1)[0]
        
        # Calculate percentage change
        first_value = data_series.iloc[0]
        last_value = data_series.iloc[-1]
        
        if first_value != 0:
            change_percent = ((last_value - first_value) / abs(first_value)) * 100
        else:
            change_percent = 0
        
        # Determine trend direction
        if abs(change_percent) < 5:
            trend = 'stable'
        elif change_percent > 0:
            trend = 'improving' if data_series.name in ['bandwidth_download'] else 'degrading'
        else:
            trend = 'degrading' if data_series.name in ['bandwidth_download'] else 'improving'
        
        return {
            'trend': trend,
            'change_percent': round(change_percent, 2),
            'slope': slope,
            'current_avg': round(data_series.tail(5).mean(), 2),
            'baseline_avg': round(data_series.head(5).mean(), 2)
        }
    
    def _generate_capacity_recommendations(self, trends: Dict[str, Any], features_df: pd.DataFrame) -> List[str]:
        """
        Generate capacity planning recommendations based on trends
        """
        recommendations = []
        
        # Latency recommendations
        if 'latency' in trends:
            latency_trend = trends['latency']
            if latency_trend['trend'] == 'degrading' and latency_trend['change_percent'] > 20:
                recommendations.append("Network latency is increasing significantly - investigate network path optimization")
            elif latency_trend['current_avg'] > 100:
                recommendations.append("Current latency levels are concerning - consider network infrastructure upgrades")
        
        # Bandwidth recommendations
        if 'bandwidth' in trends:
            bandwidth_trend = trends['bandwidth']
            if bandwidth_trend['trend'] == 'degrading' and bandwidth_trend['change_percent'] < -15:
                recommendations.append("Bandwidth capacity is declining - plan for capacity expansion")
            elif bandwidth_trend['current_avg'] < 100:
                recommendations.append("Current bandwidth levels are limiting - upgrade connection capacity")
        
        # Packet loss recommendations
        if 'packet_loss' in trends:
            loss_trend = trends['packet_loss']
            if loss_trend['trend'] == 'degrading' and loss_trend['current_avg'] > 2:
                recommendations.append("Packet loss is increasing - investigate network quality issues")
        
        # General recommendations based on overall performance
        if 'cpu_percent' in features_df.columns:
            avg_cpu = features_df['cpu_percent'].mean()
            if avg_cpu > 75:
                recommendations.append("High CPU utilization trend - consider system performance optimization")
        
        if not recommendations:
            recommendations.append("Network performance appears stable - continue regular monitoring")
        
        return recommendations
    
    def _calculate_overall_health_trend(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate overall network health trend
        """
        try:
            # Calculate health scores for each time period
            health_scores = []
            
            for i in range(len(features_df)):
                row = features_df.iloc[i:i+1]
                score = self.calculate_health_score(row)
                health_scores.append(score)
            
            if len(health_scores) < 2:
                return {'trend': 'insufficient_data', 'current_score': 0}
            
            health_series = pd.Series(health_scores)
            trend_info = self._calculate_trend(health_series)
            
            return {
                'trend': trend_info['trend'],
                'change_percent': trend_info['change_percent'],
                'current_score': round(health_series.tail(5).mean(), 1),
                'baseline_score': round(health_series.head(5).mean(), 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating health trend: {str(e)}")
            return {'trend': 'calculation_error', 'current_score': 0}

# Global diagnostic engine instance
diagnostic_engine = NetworkDiagnosticEngine()