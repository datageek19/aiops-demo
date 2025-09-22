"""
Advanced AIOps Features - Extended Functionality

This module provides advanced features for the AIOps alert analysis system:
- Real-time alert stream processing
- Advanced anomaly detection
- Multi-variate time series forecasting  
- Alert correlation analysis
- Performance optimization techniques
- Integration with external monitoring systems
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
from queue import Queue
import time
from typing import Dict, List, Any, Optional, Callable

# Advanced ML libraries
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import OPTICS
from scipy import stats
from scipy.signal import find_peaks

# Time series libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

class RealTimeAlertProcessor:
    """Real-time alert stream processing and analysis"""
    
    def __init__(self, buffer_size=1000, processing_interval=60):
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        self.alert_buffer = Queue(maxsize=buffer_size)
        self.is_running = False
        self.processing_thread = None
        self.callbacks = []
        
        # Real-time analytics
        self.current_metrics = {
            'total_alerts': 0,
            'alerts_per_minute': 0,
            'critical_alerts': 0,
            'system_health_score': 100,
            'anomaly_score': 0
        }
        
        # Historical data for trend analysis
        self.historical_data = []
        
    def add_callback(self, callback: Callable):
        """Add callback function for real-time processing results"""
        self.callbacks.append(callback)
    
    def start_processing(self):
        """Start real-time alert processing"""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("✓ Real-time alert processing started")
    
    def stop_processing(self):
        """Stop real-time alert processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        print("✓ Real-time alert processing stopped")
    
    def add_alert(self, alert: Dict[str, Any]):
        """Add new alert to processing buffer"""
        if not self.alert_buffer.full():
            alert['received_timestamp'] = datetime.now()
            self.alert_buffer.put(alert)
            return True
        return False
    
    def _processing_loop(self):
        """Main processing loop for real-time analysis"""
        while self.is_running:
            try:
                # Process alerts in buffer
                alerts_batch = []
                
                # Collect alerts from buffer
                while not self.alert_buffer.empty() and len(alerts_batch) < 100:
                    alerts_batch.append(self.alert_buffer.get())
                
                if alerts_batch:
                    # Process the batch
                    analysis_results = self._process_alert_batch(alerts_batch)
                    
                    # Update metrics
                    self._update_metrics(alerts_batch, analysis_results)
                    
                    # Trigger callbacks
                    for callback in self.callbacks:
                        try:
                            callback(alerts_batch, analysis_results, self.current_metrics)
                        except Exception as e:
                            print(f"Callback error: {e}")
                
                # Wait for next processing interval
                time.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(5)
    
    def _process_alert_batch(self, alerts_batch: List[Dict]) -> Dict:
        """Process a batch of alerts and return analysis results"""
        
        if not alerts_batch:
            return {}
        
        df = pd.DataFrame(alerts_batch)
        
        # Basic statistics
        stats = {
            'total_count': len(df),
            'critical_count': len(df[df.get('criticality') == 'critical']),
            'unique_systems': df.get('system_category', pd.Series()).nunique(),
            'avg_processing_time': 0  # Can be calculated based on timestamps
        }
        
        # Detect anomalies in the batch
        anomalies = self._detect_batch_anomalies(df)
        
        # System health assessment
        health_score = self._calculate_system_health(df)
        
        return {
            'statistics': stats,
            'anomalies': anomalies,
            'health_score': health_score,
            'timestamp': datetime.now()
        }
    
    def _detect_batch_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in current alert batch"""
        anomalies = []
        
        # Sudden spike in alert volume
        if len(df) > 50:  # Threshold for spike detection
            anomalies.append({
                'type': 'volume_spike',
                'severity': 'high',
                'description': f'Alert volume spike: {len(df)} alerts in batch'
            })
        
        # High proportion of critical alerts
        if 'criticality' in df.columns:
            critical_ratio = len(df[df['criticality'] == 'critical']) / len(df)
            if critical_ratio > 0.3:
                anomalies.append({
                    'type': 'critical_ratio_high',
                    'severity': 'medium',
                    'description': f'High critical alert ratio: {critical_ratio:.2%}'
                })
        
        # Single system generating many alerts
        if 'system_category' in df.columns:
            system_counts = df['system_category'].value_counts()
            if len(system_counts) > 0 and system_counts.iloc[0] > len(df) * 0.7:
                anomalies.append({
                    'type': 'system_concentration',
                    'severity': 'medium',
                    'description': f'Single system generating {system_counts.iloc[0]} alerts'
                })
        
        return anomalies
    
    def _calculate_system_health(self, df: pd.DataFrame) -> float:
        """Calculate overall system health score (0-100)"""
        
        base_score = 100
        
        # Deduct points for critical alerts
        if 'criticality' in df.columns:
            critical_count = len(df[df['criticality'] == 'critical'])
            base_score -= critical_count * 5
        
        # Deduct points for high alert volume
        volume_penalty = min(len(df) * 0.1, 20)
        base_score -= volume_penalty
        
        # Deduct points for system concentration
        if 'system_category' in df.columns and len(df) > 0:
            system_diversity = df['system_category'].nunique() / len(df)
            if system_diversity < 0.3:
                base_score -= 10
        
        return max(base_score, 0)
    
    def _update_metrics(self, alerts_batch: List[Dict], analysis_results: Dict):
        """Update real-time metrics"""
        
        self.current_metrics['total_alerts'] += len(alerts_batch)
        
        if 'statistics' in analysis_results:
            stats = analysis_results['statistics']
            self.current_metrics['critical_alerts'] += stats.get('critical_count', 0)
        
        if 'health_score' in analysis_results:
            # Exponential moving average for health score
            alpha = 0.3
            current_health = analysis_results['health_score']
            self.current_metrics['system_health_score'] = (
                alpha * current_health + 
                (1 - alpha) * self.current_metrics['system_health_score']
            )
        
        # Calculate alerts per minute (simple approximation)
        self.current_metrics['alerts_per_minute'] = len(alerts_batch) / (self.processing_interval / 60)
        
        # Anomaly score based on detected anomalies
        if 'anomalies' in analysis_results:
            anomaly_count = len(analysis_results['anomalies'])
            self.current_metrics['anomaly_score'] = min(anomaly_count * 10, 100)
        
        # Store historical data
        self.historical_data.append({
            'timestamp': datetime.now(),
            'metrics': self.current_metrics.copy(),
            'batch_size': len(alerts_batch)
        })
        
        # Keep only recent history (last 1000 entries)
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]

class AdvancedAnomalyDetector:
    """Advanced anomaly detection with multiple algorithms"""
    
    def __init__(self):
        self.detectors = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'statistical': None,  # Will be initialized with data
            'temporal': None
        }
        self.is_fitted = False
        self.feature_scaler = MinMaxScaler()
        
    def fit(self, alerts_df: pd.DataFrame, metrics_df: pd.DataFrame):
        """Fit anomaly detection models"""
        
        # Prepare features
        features = self._prepare_anomaly_features(alerts_df, metrics_df)
        
        if len(features) < 10:  # Need minimum samples
            print("Warning: Insufficient data for anomaly detection")
            return
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Fit Isolation Forest
        self.detectors['isolation_forest'].fit(features_scaled)
        
        # Fit statistical detector (Z-score based)
        self.detectors['statistical'] = {
            'mean': np.mean(features_scaled, axis=0),
            'std': np.std(features_scaled, axis=0),
            'threshold': 2.5  # Z-score threshold
        }
        
        # Fit temporal detector
        if 'timestamp' in alerts_df.columns:
            self._fit_temporal_detector(alerts_df)
        
        self.is_fitted = True
        print("✓ Anomaly detection models fitted")
    
    def detect_anomalies(self, alerts_df: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict:
        """Detect anomalies using multiple methods"""
        
        if not self.is_fitted:
            return {'error': 'Models not fitted'}
        
        # Prepare features
        features = self._prepare_anomaly_features(alerts_df, metrics_df)
        
        if len(features) == 0:
            return {'anomalies': []}
        
        features_scaled = self.feature_scaler.transform(features)
        
        anomalies = {
            'isolation_forest': [],
            'statistical': [],
            'temporal': [],
            'consensus': []
        }
        
        # Isolation Forest detection
        if_scores = self.detectors['isolation_forest'].decision_function(features_scaled)
        if_anomalies = self.detectors['isolation_forest'].predict(features_scaled)
        
        for i, (score, is_anomaly) in enumerate(zip(if_scores, if_anomalies)):
            if is_anomaly == -1:
                anomalies['isolation_forest'].append({
                    'index': i,
                    'score': float(score),
                    'method': 'isolation_forest'
                })
        
        # Statistical detection (Z-score)
        if self.detectors['statistical']:
            stat_detector = self.detectors['statistical']
            z_scores = np.abs((features_scaled - stat_detector['mean']) / (stat_detector['std'] + 1e-8))
            
            for i, z_score_row in enumerate(z_scores):
                max_z_score = np.max(z_score_row)
                if max_z_score > stat_detector['threshold']:
                    anomalies['statistical'].append({
                        'index': i,
                        'score': float(max_z_score),
                        'method': 'statistical'
                    })
        
        # Temporal detection
        if 'timestamp' in alerts_df.columns:
            temporal_anomalies = self._detect_temporal_anomalies(alerts_df)
            anomalies['temporal'] = temporal_anomalies
        
        # Consensus anomalies (detected by multiple methods)
        if_indices = {a['index'] for a in anomalies['isolation_forest']}
        stat_indices = {a['index'] for a in anomalies['statistical']}
        
        consensus_indices = if_indices.intersection(stat_indices)
        for idx in consensus_indices:
            anomalies['consensus'].append({
                'index': idx,
                'methods': ['isolation_forest', 'statistical'],
                'confidence': 'high'
            })
        
        return anomalies
    
    def _prepare_anomaly_features(self, alerts_df: pd.DataFrame, metrics_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        
        features = []
        
        # Alert-based features
        if not alerts_df.empty:
            # Numerical features
            numerical_cols = ['resolution_time', 'affected_hosts']
            for col in numerical_cols:
                if col in alerts_df.columns:
                    features.append(alerts_df[col].fillna(alerts_df[col].median()).values)
        
        # Metrics-based features
        if not metrics_df.empty and 'alert_id' in metrics_df.columns:
            # Aggregate metrics per alert
            metrics_agg = metrics_df.groupby('alert_id')['value'].agg(['mean', 'max', 'std']).fillna(0)
            
            if len(metrics_agg) > 0:
                features.extend([
                    metrics_agg['mean'].values,
                    metrics_agg['max'].values,
                    metrics_agg['std'].values
                ])
        
        if not features:
            return np.array([])
        
        # Ensure all features have the same length
        min_length = min(len(f) for f in features)
        features = [f[:min_length] for f in features]
        
        return np.column_stack(features) if features else np.array([])
    
    def _fit_temporal_detector(self, alerts_df: pd.DataFrame):
        """Fit temporal anomaly detector"""
        
        # Create time series of alert counts
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        hourly_counts = alerts_df.groupby(alerts_df['timestamp'].dt.floor('H')).size()
        
        if len(hourly_counts) > 24:  # Need sufficient data
            # Use moving average and standard deviation
            window = 24  # 24-hour window
            moving_avg = hourly_counts.rolling(window=window).mean()
            moving_std = hourly_counts.rolling(window=window).std()
            
            self.detectors['temporal'] = {
                'moving_avg': moving_avg,
                'moving_std': moving_std,
                'threshold': 2.0  # Standard deviations
            }
    
    def _detect_temporal_anomalies(self, alerts_df: pd.DataFrame) -> List[Dict]:
        """Detect temporal anomalies"""
        
        if not self.detectors['temporal']:
            return []
        
        anomalies = []
        
        # Create current time series
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        current_counts = alerts_df.groupby(alerts_df['timestamp'].dt.floor('H')).size()
        
        temporal_detector = self.detectors['temporal']
        
        for timestamp, count in current_counts.items():
            # Find corresponding historical average and std
            hour_of_day = timestamp.hour
            day_of_week = timestamp.dayofweek
            
            # Simple temporal anomaly: count significantly different from historical
            # This is a simplified version - in practice, you'd use more sophisticated methods
            if count > 10:  # Threshold for considering as anomaly
                anomalies.append({
                    'timestamp': timestamp,
                    'count': count,
                    'type': 'temporal_spike',
                    'method': 'temporal'
                })
        
        return anomalies

class MultiVariateTimeSeriesForecaster:
    """Advanced time series forecasting for alert prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
    def fit(self, alerts_df: pd.DataFrame, forecast_horizon=24):
        """Fit multivariate time series models"""
        
        # Prepare time series data
        ts_data = self._prepare_time_series_data(alerts_df)
        
        if ts_data.empty:
            print("Warning: Insufficient data for time series forecasting")
            return
        
        self.forecast_horizon = forecast_horizon
        
        # Fit models for different aggregation levels
        self._fit_total_alerts_model(ts_data)
        self._fit_system_specific_models(ts_data, alerts_df)
        self._fit_criticality_models(ts_data, alerts_df)
        
        self.is_fitted = True
        print("✓ Time series forecasting models fitted")
    
    def forecast(self, steps_ahead=24) -> Dict:
        """Generate forecasts for alert volumes"""
        
        if not self.is_fitted:
            return {'error': 'Models not fitted'}
        
        forecasts = {}
        
        # Total alerts forecast
        if 'total_alerts' in self.models:
            total_forecast = self._generate_forecast('total_alerts', steps_ahead)
            forecasts['total_alerts'] = total_forecast
        
        # System-specific forecasts
        system_forecasts = {}
        for system, model in self.models.items():
            if system.startswith('system_'):
                system_name = system.replace('system_', '')
                forecast = self._generate_forecast(system, steps_ahead)
                system_forecasts[system_name] = forecast
        
        if system_forecasts:
            forecasts['by_system'] = system_forecasts
        
        # Criticality forecasts
        criticality_forecasts = {}
        for criticality, model in self.models.items():
            if criticality.startswith('criticality_'):
                crit_name = criticality.replace('criticality_', '')
                forecast = self._generate_forecast(criticality, steps_ahead)
                criticality_forecasts[crit_name] = forecast
        
        if criticality_forecasts:
            forecasts['by_criticality'] = criticality_forecasts
        
        return forecasts
    
    def _prepare_time_series_data(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for modeling"""
        
        if 'timestamp' not in alerts_df.columns:
            return pd.DataFrame()
        
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Create hourly aggregations
        hourly_data = alerts_df.groupby(alerts_df['timestamp'].dt.floor('H')).size()
        hourly_data.index.name = 'timestamp'
        
        # Create complete time range
        full_range = pd.date_range(
            start=hourly_data.index.min(),
            end=hourly_data.index.max(),
            freq='H'
        )
        
        # Reindex to fill missing hours with 0
        hourly_data = hourly_data.reindex(full_range, fill_value=0)
        
        return pd.DataFrame({'total_alerts': hourly_data})
    
    def _fit_total_alerts_model(self, ts_data: pd.DataFrame):
        """Fit model for total alert volume"""
        
        if 'total_alerts' not in ts_data.columns or len(ts_data) < 48:
            return
        
        try:
            # Use Exponential Smoothing for total alerts
            model = ExponentialSmoothing(
                ts_data['total_alerts'],
                trend='add',
                seasonal='add',
                seasonal_periods=24  # Daily seasonality
            )
            
            fitted_model = model.fit()
            self.models['total_alerts'] = fitted_model
            
        except Exception as e:
            print(f"Warning: Could not fit total alerts model: {e}")
    
    def _fit_system_specific_models(self, ts_data: pd.DataFrame, alerts_df: pd.DataFrame):
        """Fit models for each system"""
        
        if 'system_category' not in alerts_df.columns:
            return
        
        for system in alerts_df['system_category'].unique():
            try:
                system_data = alerts_df[alerts_df['system_category'] == system]
                system_ts = system_data.groupby(system_data['timestamp'].dt.floor('H')).size()
                
                if len(system_ts) >= 24:  # Need at least 24 hours of data
                    # Simple exponential smoothing for system-specific data
                    model = ExponentialSmoothing(system_ts, trend='add')
                    fitted_model = model.fit()
                    self.models[f'system_{system}'] = fitted_model
                    
            except Exception as e:
                print(f"Warning: Could not fit model for system {system}: {e}")
    
    def _fit_criticality_models(self, ts_data: pd.DataFrame, alerts_df: pd.DataFrame):
        """Fit models for each criticality level"""
        
        if 'criticality' not in alerts_df.columns:
            return
        
        for criticality in alerts_df['criticality'].unique():
            try:
                crit_data = alerts_df[alerts_df['criticality'] == criticality]
                crit_ts = crit_data.groupby(crit_data['timestamp'].dt.floor('H')).size()
                
                if len(crit_ts) >= 12:  # Need at least 12 hours of data
                    model = ExponentialSmoothing(crit_ts)
                    fitted_model = model.fit()
                    self.models[f'criticality_{criticality}'] = fitted_model
                    
            except Exception as e:
                print(f"Warning: Could not fit model for criticality {criticality}: {e}")
    
    def _generate_forecast(self, model_key: str, steps_ahead: int) -> Dict:
        """Generate forecast for a specific model"""
        
        if model_key not in self.models:
            return {'error': f'Model {model_key} not found'}
        
        try:
            model = self.models[model_key]
            forecast = model.forecast(steps_ahead)
            
            # Generate confidence intervals (simplified)
            forecast_mean = forecast.values if hasattr(forecast, 'values') else forecast
            forecast_std = np.std(forecast_mean) if len(forecast_mean) > 1 else 1.0
            
            confidence_intervals = {
                'lower_80': forecast_mean - 1.28 * forecast_std,
                'upper_80': forecast_mean + 1.28 * forecast_std,
                'lower_95': forecast_mean - 1.96 * forecast_std,
                'upper_95': forecast_mean + 1.96 * forecast_std
            }
            
            return {
                'forecast': forecast_mean.tolist() if hasattr(forecast_mean, 'tolist') else [forecast_mean],
                'confidence_intervals': confidence_intervals,
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            return {'error': f'Forecast generation failed: {e}'}

class AlertCorrelationAnalyzer:
    """Analyze correlations between different types of alerts"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.correlation_rules = []
        self.temporal_correlations = {}
        
    def analyze_correlations(self, alerts_df: pd.DataFrame) -> Dict:
        """Analyze correlations between alert types and systems"""
        
        correlations = {}
        
        # System-Alert Type correlations
        if 'system_category' in alerts_df.columns and 'alert_type' in alerts_df.columns:
            system_alert_corr = self._analyze_system_alert_correlations(alerts_df)
            correlations['system_alert'] = system_alert_corr
        
        # Temporal correlations
        if 'timestamp' in alerts_df.columns:
            temporal_corr = self._analyze_temporal_correlations(alerts_df)
            correlations['temporal'] = temporal_corr
        
        # Criticality correlations
        if 'criticality' in alerts_df.columns:
            criticality_corr = self._analyze_criticality_correlations(alerts_df)
            correlations['criticality'] = criticality_corr
        
        # Generate correlation rules
        self.correlation_rules = self._generate_correlation_rules(correlations)
        
        return {
            'correlations': correlations,
            'rules': self.correlation_rules,
            'insights': self._generate_correlation_insights(correlations)
        }
    
    def _analyze_system_alert_correlations(self, alerts_df: pd.DataFrame) -> Dict:
        """Analyze correlations between systems and alert types"""
        
        # Create contingency table
        contingency = pd.crosstab(alerts_df['system_category'], alerts_df['alert_type'])
        
        # Calculate correlation coefficient
        correlation_matrix = contingency.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        'alert_type_1': correlation_matrix.columns[i],
                        'alert_type_2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
        
        return {
            'matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def _analyze_temporal_correlations(self, alerts_df: pd.DataFrame) -> Dict:
        """Analyze temporal correlations between alerts"""
        
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Analyze alert sequences (alerts that occur within time windows)
        time_window = timedelta(minutes=30)  # 30-minute window
        sequences = []
        
        for i, alert in alerts_df.iterrows():
            # Find alerts within time window
            window_start = alert['timestamp'] - time_window
            window_end = alert['timestamp'] + time_window
            
            related_alerts = alerts_df[
                (alerts_df['timestamp'] >= window_start) &
                (alerts_df['timestamp'] <= window_end) &
                (alerts_df.index != i)
            ]
            
            if len(related_alerts) > 0:
                sequences.append({
                    'primary_alert': {
                        'type': alert.get('alert_type', 'unknown'),
                        'system': alert.get('system_category', 'unknown'),
                        'timestamp': alert['timestamp']
                    },
                    'related_alerts': related_alerts[['alert_type', 'system_category', 'timestamp']].to_dict('records')
                })
        
        # Analyze most common sequences
        sequence_patterns = {}
        for seq in sequences:
            primary_type = seq['primary_alert']['type']
            for related in seq['related_alerts']:
                pattern = f"{primary_type} -> {related['alert_type']}"
                sequence_patterns[pattern] = sequence_patterns.get(pattern, 0) + 1
        
        # Get top patterns
        top_patterns = sorted(sequence_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'sequences': sequences[:50],  # Limit for performance
            'common_patterns': top_patterns,
            'total_sequences': len(sequences)
        }
    
    def _analyze_criticality_correlations(self, alerts_df: pd.DataFrame) -> Dict:
        """Analyze correlations with alert criticality"""
        
        criticality_analysis = {}
        
        # System vs Criticality
        if 'system_category' in alerts_df.columns:
            system_crit = pd.crosstab(alerts_df['system_category'], alerts_df['criticality'])
            criticality_analysis['system_criticality'] = system_crit.to_dict()
        
        # Alert Type vs Criticality
        if 'alert_type' in alerts_df.columns:
            type_crit = pd.crosstab(alerts_df['alert_type'], alerts_df['criticality'])
            criticality_analysis['type_criticality'] = type_crit.to_dict()
        
        # Time vs Criticality
        if 'timestamp' in alerts_df.columns:
            alerts_df['hour'] = pd.to_datetime(alerts_df['timestamp']).dt.hour
            time_crit = pd.crosstab(alerts_df['hour'], alerts_df['criticality'])
            criticality_analysis['time_criticality'] = time_crit.to_dict()
        
        return criticality_analysis
    
    def _generate_correlation_rules(self, correlations: Dict) -> List[Dict]:
        """Generate actionable correlation rules"""
        
        rules = []
        
        # Rules from system-alert correlations
        if 'system_alert' in correlations:
            for corr in correlations['system_alert'].get('strong_correlations', []):
                if corr['correlation'] > 0.7:
                    rules.append({
                        'type': 'co_occurrence',
                        'rule': f"When {corr['alert_type_1']} occurs, {corr['alert_type_2']} is likely to occur",
                        'confidence': corr['correlation'],
                        'category': 'system_alert'
                    })
        
        # Rules from temporal correlations
        if 'temporal' in correlations:
            for pattern, count in correlations['temporal'].get('common_patterns', [])[:5]:
                if count > 5:  # Minimum occurrences
                    rules.append({
                        'type': 'sequence',
                        'rule': f"Alert sequence pattern: {pattern}",
                        'frequency': count,
                        'category': 'temporal'
                    })
        
        return rules
    
    def _generate_correlation_insights(self, correlations: Dict) -> List[str]:
        """Generate human-readable insights from correlations"""
        
        insights = []
        
        # System-alert insights
        if 'system_alert' in correlations:
            strong_corrs = correlations['system_alert'].get('strong_correlations', [])
            if strong_corrs:
                insights.append(f"Found {len(strong_corrs)} strong correlations between alert types")
                
                strongest = max(strong_corrs, key=lambda x: abs(x['correlation']))
                insights.append(
                    f"Strongest correlation: {strongest['alert_type_1']} and {strongest['alert_type_2']} "
                    f"(correlation: {strongest['correlation']:.2f})"
                )
        
        # Temporal insights
        if 'temporal' in correlations:
            total_sequences = correlations['temporal'].get('total_sequences', 0)
            if total_sequences > 0:
                insights.append(f"Detected {total_sequences} temporal alert sequences")
                
                common_patterns = correlations['temporal'].get('common_patterns', [])
                if common_patterns:
                    top_pattern = common_patterns[0]
                    insights.append(f"Most common sequence: {top_pattern[0]} (occurred {top_pattern[1]} times)")
        
        return insights

def create_advanced_demo():
    """Demonstrate advanced AIOps features"""
    
    print("🚀 Advanced AIOps Features Demo")
    print("=" * 50)
    
    # Generate sample data
    from aiops_alert_analysis import SyntheticDataGenerator
    
    data_generator = SyntheticDataGenerator(seed=42)
    alerts_df = data_generator.generate_alerts(num_alerts=500, days_back=15)
    metrics_df = data_generator.generate_metrics_data(alerts_df)
    
    print(f"✓ Generated {len(alerts_df)} alerts for advanced analysis")
    
    # 1. Real-time Processing Demo
    print("\n1. Real-time Alert Processing")
    print("-" * 30)
    
    processor = RealTimeAlertProcessor(buffer_size=100, processing_interval=1)
    
    def alert_callback(alerts_batch, analysis_results, metrics):
        print(f"  Processed batch: {len(alerts_batch)} alerts, Health Score: {metrics['system_health_score']:.1f}")
    
    processor.add_callback(alert_callback)
    processor.start_processing()
    
    # Simulate real-time alerts
    for _, alert in alerts_df.head(20).iterrows():
        processor.add_alert(alert.to_dict())
        time.sleep(0.1)  # Simulate real-time arrival
    
    time.sleep(2)  # Let processing complete
    processor.stop_processing()
    
    # 2. Advanced Anomaly Detection
    print("\n2. Advanced Anomaly Detection")
    print("-" * 30)
    
    anomaly_detector = AdvancedAnomalyDetector()
    anomaly_detector.fit(alerts_df, metrics_df)
    
    # Detect anomalies in subset
    test_alerts = alerts_df.sample(n=50, random_state=42)
    test_metrics = metrics_df[metrics_df['alert_id'].isin(test_alerts['alert_id'])]
    
    anomalies = anomaly_detector.detect_anomalies(test_alerts, test_metrics)
    
    print(f"  Isolation Forest anomalies: {len(anomalies['isolation_forest'])}")
    print(f"  Statistical anomalies: {len(anomalies['statistical'])}")
    print(f"  Temporal anomalies: {len(anomalies['temporal'])}")
    print(f"  Consensus anomalies: {len(anomalies['consensus'])}")
    
    # 3. Time Series Forecasting
    print("\n3. Multi-variate Time Series Forecasting")
    print("-" * 30)
    
    forecaster = MultiVariateTimeSeriesForecaster()
    forecaster.fit(alerts_df, forecast_horizon=24)
    
    forecasts = forecaster.forecast(steps_ahead=12)
    
    if 'total_alerts' in forecasts:
        total_forecast = forecasts['total_alerts']['forecast']
        print(f"  Next 12 hours forecast: {[f'{x:.1f}' for x in total_forecast[:6]]}")
    
    if 'by_system' in forecasts:
        print(f"  System-specific forecasts: {len(forecasts['by_system'])} systems")
    
    # 4. Correlation Analysis
    print("\n4. Alert Correlation Analysis")
    print("-" * 30)
    
    correlator = AlertCorrelationAnalyzer()
    correlation_analysis = correlator.analyze_correlations(alerts_df)
    
    print(f"  Correlation rules generated: {len(correlation_analysis['rules'])}")
    print(f"  Key insights: {len(correlation_analysis['insights'])}")
    
    for insight in correlation_analysis['insights'][:3]:
        print(f"    • {insight}")
    
    print("\n✅ Advanced features demonstration complete!")
    
    return {
        'processor_metrics': processor.current_metrics,
        'anomalies': anomalies,
        'forecasts': forecasts,
        'correlations': correlation_analysis
    }

if __name__ == "__main__":
    # Run advanced features demo
    results = create_advanced_demo()
