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
            'anomaly_detector': 'anomaly_model.joblib',
            'health_classifier': 'health_classifier.joblib',
            'performance_predictor': 'performance_predictor.joblib',
            'scaler': 'feature_scaler.joblib',
            'health_encoder': 'health_encoder.joblib'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                try:
                    if model_name == 'scaler':
                        self.scalers['main'] = joblib.load(filepath)
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
                filepath = os.path.join(self.models_dir, "feature_scaler.joblib")
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
            'network': 0.4,
            'system': 0.3,
            'reliability': 0.2,
            'efficiency': 0.1
        }
        
        # Network health (lower latency, packet loss, jitter is better)
        network_score = 100
        if features['ping_latency'].mean() > 0:
            network_score -= min(features['ping_latency'].mean() / 2, 50)  # Penalize high latency
        if features['ping_packet_loss'].mean() > 0:
            network_score -= min(features['ping_packet_loss'].mean() * 10, 40)  # Penalize packet loss
        if features['jitter'].mean() > 0:
            network_score -= min(features['jitter'].mean() / 2, 10)  # Penalize high jitter
        
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
            efficiency_score * weights['efficiency']
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
            
            # Train scaler
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
                    X_perf_scaled = scaler.fit_transform(X_perf)
                    performance_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    performance_predictor.fit(X_perf_scaled, y_perf)
                    self.models['performance_predictor'] = performance_predictor
                    # Store the feature columns used for performance prediction
                    self.performance_feature_columns = feature_cols
            
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
                    
                    if anomaly_count > 0:
                        diagnosis['issues_detected'].append({
                            'type': 'anomaly',
                            'severity': 'high' if anomaly_count > len(results) * 0.2 else 'medium',
                            'description': f'Detected {anomaly_count} anomalous measurements out of {len(results)} total',
                            'recommendation': 'Review network configuration and system health during anomalous periods'
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

# Global diagnostic engine instance
diagnostic_engine = NetworkDiagnosticEngine()