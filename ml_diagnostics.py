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
        Rule-based diagnostic analysis
        """
        issues = []
        recommendations = []
        
        # Network performance rules
        avg_latency = features_df['ping_latency'].mean()
        avg_packet_loss = features_df['ping_packet_loss'].mean()
        avg_jitter = features_df['jitter'].mean()
        
        if avg_latency > 100:
            issues.append({
                'type': 'high_latency',
                'severity': 'high' if avg_latency > 200 else 'medium',
                'description': f'High average latency: {avg_latency:.1f}ms',
                'recommendation': 'Check network routing and bandwidth utilization'
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
        
        # System performance rules
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
            'recommendations': recommendations
        }
    
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