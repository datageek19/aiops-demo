"""
AIOps Alert Analysis and Prediction System

This comprehensive solution addresses alert clustering, frequency analysis,
and predictive modeling for IT operations management.

Key Features:
- Alert clustering based on criticality and system association
- Temporal pattern analysis and frequency detection
- Predictive models for alert likelihood and system attribution
- Multi-modal data integration (metrics, logs, traces, profiles)
- Real-time alert classification and recommendation system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Utility libraries
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import pickle

class SyntheticDataGenerator:
    """Generate realistic synthetic alert data for AIOps scenarios"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Define system categories and their characteristics
        self.systems = {
            'web_servers': {
                'services': ['nginx', 'apache', 'tomcat', 'iis'],
                'common_alerts': ['high_cpu', 'memory_leak', 'connection_timeout', 'ssl_certificate_expiry'],
                'criticality_dist': [0.1, 0.3, 0.4, 0.2]  # critical, high, medium, low
            },
            'databases': {
                'services': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
                'common_alerts': ['connection_pool_exhausted', 'slow_query', 'disk_space_low', 'replication_lag'],
                'criticality_dist': [0.2, 0.4, 0.3, 0.1]
            },
            'cloud_services': {
                'services': ['aws_ec2', 'aws_rds', 'azure_vm', 'gcp_compute', 'kubernetes'],
                'common_alerts': ['instance_down', 'auto_scaling_triggered', 'network_latency', 'cost_anomaly'],
                'criticality_dist': [0.15, 0.35, 0.35, 0.15]
            },
            'monitoring': {
                'services': ['prometheus', 'grafana', 'datadog', 'newrelic', 'splunk'],
                'common_alerts': ['metric_collection_failed', 'dashboard_timeout', 'alert_fatigue', 'storage_quota_exceeded'],
                'criticality_dist': [0.05, 0.25, 0.5, 0.2]
            },
            'security': {
                'services': ['firewall', 'ids', 'antivirus', 'siem', 'vulnerability_scanner'],
                'common_alerts': ['suspicious_activity', 'failed_login_attempts', 'malware_detected', 'policy_violation'],
                'criticality_dist': [0.3, 0.4, 0.2, 0.1]
            }
        }
        
        self.criticality_levels = ['critical', 'high', 'medium', 'low']
        self.alert_statuses = ['open', 'acknowledged', 'resolved', 'closed']
        
    def generate_alerts(self, num_alerts=1000, days_back=30):
        """Generate synthetic alert data"""
        alerts = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for i in range(num_alerts):
            # Select system category
            system_category = random.choice(list(self.systems.keys()))
            system_info = self.systems[system_category]
            
            # Select specific service
            service = random.choice(system_info['services'])
            
            # Select alert type
            alert_type = random.choice(system_info['common_alerts'])
            
            # Assign criticality based on system-specific distribution
            criticality = np.random.choice(
                self.criticality_levels,
                p=system_info['criticality_dist']
            )
            
            # Generate timestamp with realistic patterns
            # More alerts during business hours and weekdays
            timestamp = self._generate_realistic_timestamp(start_date, end_date)
            
            # Generate alert description
            description = f"{alert_type.replace('_', ' ').title()} detected on {service} in {system_category.replace('_', ' ')}"
            
            # Add some noise and variations
            if random.random() < 0.3:  # 30% chance of multi-system alerts
                additional_system = random.choice([s for s in self.systems.keys() if s != system_category])
                description += f" affecting {additional_system.replace('_', ' ')}"
                system_category = f"{system_category},{additional_system}"
            
            alert = {
                'alert_id': f"ALT-{i+1:05d}",
                'timestamp': timestamp,
                'system_category': system_category,
                'service': service,
                'alert_type': alert_type,
                'criticality': criticality,
                'description': description,
                'status': random.choice(self.alert_statuses),
                'resolution_time': self._generate_resolution_time(criticality),
                'affected_hosts': random.randint(1, 10),
                'source_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            }
            alerts.append(alert)
        
        return pd.DataFrame(alerts)
    
    def _generate_realistic_timestamp(self, start_date, end_date):
        """Generate timestamps with realistic patterns"""
        # More likely during business hours (9 AM - 6 PM)
        # More likely on weekdays
        
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Adjust for business hours bias
        if random.random() < 0.7:  # 70% during business hours
            hour = random.randint(9, 18)
            random_date = random_date.replace(hour=hour)
        
        # Adjust for weekday bias
        if random.random() < 0.8 and random_date.weekday() >= 5:  # 80% on weekdays
            days_to_subtract = random_date.weekday() - 4
            random_date = random_date - timedelta(days=days_to_subtract)
        
        return random_date
    
    def _generate_resolution_time(self, criticality):
        """Generate realistic resolution times based on criticality"""
        base_times = {
            'critical': 30,    # 30 minutes average
            'high': 120,       # 2 hours average
            'medium': 480,     # 8 hours average
            'low': 1440        # 24 hours average
        }
        
        base_time = base_times[criticality]
        # Add some randomness (±50%)
        actual_time = base_time * (0.5 + random.random())
        return int(actual_time)
    
    def generate_metrics_data(self, alerts_df, num_metrics_per_alert=5):
        """Generate synthetic metrics data associated with alerts"""
        metrics_data = []
        
        metric_types = [
            'cpu_usage', 'memory_usage', 'disk_io', 'network_io',
            'response_time', 'error_rate', 'throughput', 'latency'
        ]
        
        for _, alert in alerts_df.iterrows():
            for _ in range(num_metrics_per_alert):
                metric = {
                    'alert_id': alert['alert_id'],
                    'metric_name': random.choice(metric_types),
                    'value': random.uniform(0, 100),
                    'threshold': random.uniform(70, 90),
                    'timestamp': alert['timestamp'],
                    'host': f"host-{random.randint(1, 50)}"
                }
                metrics_data.append(metric)
        
        return pd.DataFrame(metrics_data)
    
    def generate_log_data(self, alerts_df):
        """Generate synthetic log data"""
        log_data = []
        
        log_levels = ['ERROR', 'WARN', 'INFO', 'DEBUG']
        log_templates = [
            "Connection timeout occurred",
            "Memory usage exceeded threshold",
            "Service restart initiated",
            "Authentication failed",
            "Database query slow",
            "Network latency detected"
        ]
        
        for _, alert in alerts_df.iterrows():
            num_logs = random.randint(1, 10)
            for i in range(num_logs):
                log_entry = {
                    'alert_id': alert['alert_id'],
                    'timestamp': alert['timestamp'] + timedelta(minutes=random.randint(-30, 30)),
                    'log_level': random.choice(log_levels),
                    'message': random.choice(log_templates),
                    'source': alert['service'],
                    'host': f"host-{random.randint(1, 50)}"
                }
                log_data.append(log_entry)
        
        return pd.DataFrame(log_data)

class AlertClusteringAnalyzer:
    """Advanced alert clustering and analysis system"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.label_encoders = {}
        self.clustering_models = {}
        
    def prepare_features(self, alerts_df):
        """Prepare features for clustering analysis"""
        features_df = alerts_df.copy()
        
        # Encode categorical variables
        categorical_cols = ['system_category', 'service', 'alert_type', 'criticality', 'status']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col])
            else:
                features_df[f'{col}_encoded'] = self.label_encoders[col].transform(features_df[col])
        
        # Extract temporal features
        features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
        features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_business_hours'] = features_df['hour'].between(9, 17).astype(int)
        
        # TF-IDF features from descriptions
        tfidf_features = self.tfidf_vectorizer.fit_transform(features_df['description'])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine all features
        feature_columns = [
            'system_category_encoded', 'service_encoded', 'alert_type_encoded',
            'criticality_encoded', 'status_encoded', 'resolution_time',
            'affected_hosts', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours'
        ]
        
        numerical_features = features_df[feature_columns]
        combined_features = pd.concat([numerical_features.reset_index(drop=True), tfidf_df], axis=1)
        
        return combined_features
    
    def perform_clustering(self, features, n_clusters_range=(2, 10)):
        """Perform multiple clustering algorithms and select the best"""
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        clustering_results = {}
        
        # K-Means clustering
        best_kmeans_score = -1
        best_kmeans_k = 2
        
        for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            if len(set(cluster_labels)) > 1:  # Ensure we have multiple clusters
                score = silhouette_score(features_scaled, cluster_labels)
                if score > best_kmeans_score:
                    best_kmeans_score = score
                    best_kmeans_k = k
        
        # Fit best K-Means
        kmeans_best = KMeans(n_clusters=best_kmeans_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_best.fit_predict(features_scaled)
        
        clustering_results['kmeans'] = {
            'model': kmeans_best,
            'labels': kmeans_labels,
            'score': best_kmeans_score,
            'n_clusters': best_kmeans_k
        }
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features_scaled)
        
        if len(set(dbscan_labels)) > 1:
            dbscan_score = silhouette_score(features_scaled, dbscan_labels)
        else:
            dbscan_score = -1
        
        clustering_results['dbscan'] = {
            'model': dbscan,
            'labels': dbscan_labels,
            'score': dbscan_score,
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        }
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=best_kmeans_k)
        hierarchical_labels = hierarchical.fit_predict(features_scaled)
        hierarchical_score = silhouette_score(features_scaled, hierarchical_labels)
        
        clustering_results['hierarchical'] = {
            'model': hierarchical,
            'labels': hierarchical_labels,
            'score': hierarchical_score,
            'n_clusters': best_kmeans_k
        }
        
        # Select best clustering method
        best_method = max(clustering_results.keys(), key=lambda x: clustering_results[x]['score'])
        
        self.clustering_models = clustering_results
        return clustering_results, best_method
    
    def analyze_clusters(self, alerts_df, cluster_labels, method_name):
        """Analyze and interpret clustering results"""
        
        analysis_df = alerts_df.copy()
        analysis_df['cluster'] = cluster_labels
        
        cluster_analysis = {}
        
        for cluster_id in sorted(analysis_df['cluster'].unique()):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
            
            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(analysis_df) * 100,
                'criticality_distribution': cluster_data['criticality'].value_counts().to_dict(),
                'system_distribution': cluster_data['system_category'].value_counts().to_dict(),
                'alert_type_distribution': cluster_data['alert_type'].value_counts().to_dict(),
                'avg_resolution_time': cluster_data['resolution_time'].mean(),
                'avg_affected_hosts': cluster_data['affected_hosts'].mean(),
                'temporal_patterns': {
                    'peak_hours': cluster_data['timestamp'].dt.hour.value_counts().head(3).to_dict(),
                    'peak_days': cluster_data['timestamp'].dt.day_name().value_counts().head(3).to_dict()
                }
            }
            
            cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis

class AlertFrequencyAnalyzer:
    """Analyze temporal patterns and frequency of alerts"""
    
    def __init__(self):
        self.frequency_models = {}
    
    def analyze_temporal_patterns(self, alerts_df):
        """Comprehensive temporal pattern analysis"""
        
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        alerts_df = alerts_df.sort_values('timestamp')
        
        # Create time series data
        daily_counts = alerts_df.groupby(alerts_df['timestamp'].dt.date).size()
        hourly_counts = alerts_df.groupby(alerts_df['timestamp'].dt.hour).size()
        weekly_counts = alerts_df.groupby(alerts_df['timestamp'].dt.day_name()).size()
        
        # System-specific patterns
        system_temporal = {}
        for system in alerts_df['system_category'].unique():
            system_data = alerts_df[alerts_df['system_category'] == system]
            system_temporal[system] = {
                'hourly_pattern': system_data.groupby(system_data['timestamp'].dt.hour).size().to_dict(),
                'daily_pattern': system_data.groupby(system_data['timestamp'].dt.day_name()).size().to_dict(),
                'total_alerts': len(system_data)
            }
        
        # Criticality-based patterns
        criticality_temporal = {}
        for criticality in alerts_df['criticality'].unique():
            crit_data = alerts_df[alerts_df['criticality'] == criticality]
            criticality_temporal[criticality] = {
                'hourly_pattern': crit_data.groupby(crit_data['timestamp'].dt.hour).size().to_dict(),
                'daily_pattern': crit_data.groupby(crit_data['timestamp'].dt.day_name()).size().to_dict(),
                'total_alerts': len(crit_data)
            }
        
        # Detect anomalies in frequency
        anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        daily_counts_array = daily_counts.values.reshape(-1, 1)
        anomalies = anomaly_detector.fit_predict(daily_counts_array)
        anomaly_dates = daily_counts[anomalies == -1].index.tolist()
        
        return {
            'daily_counts': daily_counts.to_dict(),
            'hourly_patterns': hourly_counts.to_dict(),
            'weekly_patterns': weekly_counts.to_dict(),
            'system_temporal_patterns': system_temporal,
            'criticality_temporal_patterns': criticality_temporal,
            'anomaly_dates': [str(date) for date in anomaly_dates]
        }
    
    def build_frequency_prediction_model(self, alerts_df):
        """Build models to predict alert frequency"""
        
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Prepare time series data
        daily_alerts = alerts_df.groupby(alerts_df['timestamp'].dt.date).size()
        daily_alerts.index = pd.to_datetime(daily_alerts.index)
        
        # Feature engineering for prediction
        features = []
        targets = []
        
        for i in range(7, len(daily_alerts)):  # Use 7-day window
            feature_window = daily_alerts.iloc[i-7:i].values
            target = daily_alerts.iloc[i]
            
            # Add temporal features
            current_date = daily_alerts.index[i]
            temporal_features = [
                current_date.dayofweek,
                current_date.day,
                current_date.month,
                int(current_date.weekday() >= 5)  # is_weekend
            ]
            
            combined_features = list(feature_window) + temporal_features
            features.append(combined_features)
            targets.append(target)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Train prediction model
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Random Forest for frequency prediction
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)
        
        self.frequency_models['random_forest'] = {
            'model': rf_model,
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': rf_model.feature_importances_
        }
        
        return self.frequency_models

class AlertPredictionSystem:
    """Advanced alert prediction and system attribution system"""
    
    def __init__(self):
        self.prediction_models = {}
        self.system_attribution_models = {}
        self.tokenizer = None
        
    def prepare_prediction_features(self, alerts_df, metrics_df, logs_df):
        """Prepare comprehensive features for prediction"""
        
        # Merge alert data with metrics and logs
        alerts_with_metrics = alerts_df.merge(
            metrics_df.groupby('alert_id').agg({
                'value': ['mean', 'max', 'std'],
                'metric_name': lambda x: ','.join(x.unique())
            }).round(2),
            on='alert_id',
            how='left'
        )
        
        # Flatten column names
        alerts_with_metrics.columns = [
            '_'.join(col).strip() if isinstance(col, tuple) else col 
            for col in alerts_with_metrics.columns
        ]
        
        # Add log-based features
        log_features = logs_df.groupby('alert_id').agg({
            'log_level': lambda x: ','.join(x.unique()),
            'message': lambda x: ' '.join(x),
            'source': lambda x: ','.join(x.unique())
        })
        
        prediction_data = alerts_with_metrics.merge(
            log_features, on='alert_id', how='left'
        )
        
        return prediction_data
    
    def build_alert_likelihood_model(self, prediction_data):
        """Build models to predict alert likelihood"""
        
        # Prepare features
        feature_columns = [
            'resolution_time', 'affected_hosts', 'value_mean', 'value_max', 'value_std'
        ]
        
        # Handle missing values
        for col in feature_columns:
            if col in prediction_data.columns:
                prediction_data[col] = prediction_data[col].fillna(prediction_data[col].median())
        
        # Create binary target: high-impact alerts (critical or high criticality)
        prediction_data['high_impact'] = prediction_data['criticality'].isin(['critical', 'high']).astype(int)
        
        # Encode categorical features
        categorical_features = ['system_category', 'service', 'alert_type']
        encoded_features = []
        
        for col in categorical_features:
            if col in prediction_data.columns:
                encoded = pd.get_dummies(prediction_data[col], prefix=col)
                encoded_features.append(encoded)
        
        # Combine all features
        numerical_features = prediction_data[feature_columns]
        if encoded_features:
            all_features = pd.concat([numerical_features] + encoded_features, axis=1)
        else:
            all_features = numerical_features
        
        # Handle any remaining NaN values
        all_features = all_features.fillna(0)
        
        # Split data
        X = all_features
        y = prediction_data['high_impact']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)
        
        # Predictions for detailed analysis
        y_pred = rf_model.predict(X_test)
        
        self.prediction_models['alert_likelihood'] = {
            'model': rf_model,
            'train_score': train_score,
            'test_score': test_score,
            'feature_names': X.columns.tolist(),
            'feature_importance': rf_model.feature_importances_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return self.prediction_models['alert_likelihood']
    
    def build_system_attribution_model(self, prediction_data):
        """Build model to predict which system will generate alerts"""
        
        # Prepare text features from descriptions and log messages
        text_features = []
        for _, row in prediction_data.iterrows():
            combined_text = f"{row.get('description', '')} {row.get('message', '')}"
            text_features.append(combined_text)
        
        # Tokenize text features
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_features)
        
        sequences = self.tokenizer.texts_to_sequences(text_features)
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
        
        # Prepare target labels
        system_encoder = LabelEncoder()
        y_systems = system_encoder.fit_transform(prediction_data['system_category'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, y_systems, test_size=0.2, stratify=y_systems, random_state=42
        )
        
        # Build LSTM model for system attribution
        model = Sequential([
            Embedding(input_dim=1000, output_dim=64, input_length=100),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(len(system_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        self.system_attribution_models['lstm'] = {
            'model': model,
            'tokenizer': self.tokenizer,
            'label_encoder': system_encoder,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'training_history': history.history
        }
        
        return self.system_attribution_models['lstm']

class AIOpsVisualizationDashboard:
    """Comprehensive visualization and reporting dashboard"""
    
    def __init__(self):
        self.figures = {}
    
    def create_clustering_visualization(self, alerts_df, cluster_labels, features):
        """Create clustering visualization"""
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(StandardScaler().fit_transform(features))
        
        # Create scatter plot
        fig = px.scatter(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            color=cluster_labels.astype(str),
            hover_data={
                'Alert Type': alerts_df['alert_type'],
                'System': alerts_df['system_category'],
                'Criticality': alerts_df['criticality']
            },
            title='Alert Clustering Visualization (PCA)',
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
        )
        
        self.figures['clustering'] = fig
        return fig
    
    def create_temporal_analysis_dashboard(self, temporal_analysis):
        """Create comprehensive temporal analysis dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Daily Alert Counts',
                'Hourly Alert Patterns',
                'Weekly Alert Distribution',
                'System-wise Temporal Patterns'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily counts
        daily_data = temporal_analysis['daily_counts']
        fig.add_trace(
            go.Scatter(
                x=list(daily_data.keys()),
                y=list(daily_data.values()),
                mode='lines+markers',
                name='Daily Alerts'
            ),
            row=1, col=1
        )
        
        # Hourly patterns
        hourly_data = temporal_analysis['hourly_patterns']
        fig.add_trace(
            go.Bar(
                x=list(hourly_data.keys()),
                y=list(hourly_data.values()),
                name='Hourly Distribution'
            ),
            row=1, col=2
        )
        
        # Weekly patterns
        weekly_data = temporal_analysis['weekly_patterns']
        fig.add_trace(
            go.Bar(
                x=list(weekly_data.keys()),
                y=list(weekly_data.values()),
                name='Weekly Distribution'
            ),
            row=2, col=1
        )
        
        # System patterns (show top 3 systems)
        system_data = temporal_analysis['system_temporal_patterns']
        top_systems = sorted(system_data.items(), key=lambda x: x[1]['total_alerts'], reverse=True)[:3]
        
        for i, (system, data) in enumerate(top_systems):
            hourly_pattern = data['hourly_pattern']
            fig.add_trace(
                go.Scatter(
                    x=list(hourly_pattern.keys()),
                    y=list(hourly_pattern.values()),
                    mode='lines',
                    name=f'{system} Hourly'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Temporal Analysis Dashboard",
            showlegend=True
        )
        
        self.figures['temporal'] = fig
        return fig
    
    def create_prediction_model_performance(self, model_results):
        """Create model performance visualization"""
        
        # Feature importance plot
        if 'feature_importance' in model_results:
            importance_data = model_results['feature_importance']
            feature_names = model_results.get('feature_names', [f'Feature_{i}' for i in range(len(importance_data))])
            
            # Get top 10 most important features
            top_indices = np.argsort(importance_data)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = [importance_data[i] for i in top_indices]
            
            fig = px.bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                title='Top 10 Feature Importance for Alert Prediction',
                labels={'x': 'Importance Score', 'y': 'Features'}
            )
            
            self.figures['feature_importance'] = fig
            return fig
    
    def generate_comprehensive_report(self, alerts_df, cluster_analysis, temporal_analysis, prediction_results):
        """Generate comprehensive analysis report"""
        
        report = {
            'summary': {
                'total_alerts': len(alerts_df),
                'unique_systems': alerts_df['system_category'].nunique(),
                'unique_services': alerts_df['service'].nunique(),
                'date_range': {
                    'start': str(alerts_df['timestamp'].min()),
                    'end': str(alerts_df['timestamp'].max())
                },
                'criticality_distribution': alerts_df['criticality'].value_counts().to_dict(),
                'avg_resolution_time': alerts_df['resolution_time'].mean()
            },
            'clustering_insights': cluster_analysis,
            'temporal_insights': temporal_analysis,
            'prediction_performance': prediction_results,
            'recommendations': self._generate_recommendations(alerts_df, cluster_analysis, temporal_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, alerts_df, cluster_analysis, temporal_analysis):
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Critical system recommendations
        critical_alerts = alerts_df[alerts_df['criticality'] == 'critical']
        if len(critical_alerts) > 0:
            top_critical_system = critical_alerts['system_category'].value_counts().index[0]
            recommendations.append(
                f"Focus on {top_critical_system}: This system generates {len(critical_alerts[critical_alerts['system_category'] == top_critical_system])} critical alerts. Consider implementing proactive monitoring."
            )
        
        # Temporal recommendations
        if 'hourly_patterns' in temporal_analysis:
            peak_hour = max(temporal_analysis['hourly_patterns'], key=temporal_analysis['hourly_patterns'].get)
            recommendations.append(
                f"Peak Alert Hour: Most alerts occur at hour {peak_hour}. Consider staffing adjustments or proactive maintenance during this time."
            )
        
        # Anomaly recommendations
        if temporal_analysis.get('anomaly_dates'):
            recommendations.append(
                f"Anomaly Detection: {len(temporal_analysis['anomaly_dates'])} days with unusual alert patterns detected. Investigate these dates for potential systemic issues."
            )
        
        return recommendations

def main():
    """Main execution function demonstrating the complete AIOps alert analysis system"""
    
    print("🚀 Starting AIOps Alert Analysis System")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating Synthetic Alert Data...")
    data_generator = SyntheticDataGenerator(seed=42)
    
    # Generate alerts
    alerts_df = data_generator.generate_alerts(num_alerts=1000, days_back=30)
    print(f"   ✓ Generated {len(alerts_df)} alerts")
    
    # Generate associated metrics and logs
    metrics_df = data_generator.generate_metrics_data(alerts_df)
    logs_df = data_generator.generate_log_data(alerts_df)
    print(f"   ✓ Generated {len(metrics_df)} metrics records")
    print(f"   ✓ Generated {len(logs_df)} log records")
    
    # Step 2: Alert Clustering Analysis
    print("\n2. Performing Alert Clustering Analysis...")
    clustering_analyzer = AlertClusteringAnalyzer()
    
    # Prepare features and perform clustering
    features = clustering_analyzer.prepare_features(alerts_df)
    clustering_results, best_method = clustering_analyzer.perform_clustering(features)
    
    print(f"   ✓ Best clustering method: {best_method}")
    print(f"   ✓ Silhouette score: {clustering_results[best_method]['score']:.3f}")
    print(f"   ✓ Number of clusters: {clustering_results[best_method]['n_clusters']}")
    
    # Analyze clusters
    cluster_labels = clustering_results[best_method]['labels']
    cluster_analysis = clustering_analyzer.analyze_clusters(alerts_df, cluster_labels, best_method)
    
    print(f"   ✓ Cluster analysis completed for {len(cluster_analysis)} clusters")
    
    # Step 3: Temporal Pattern Analysis
    print("\n3. Analyzing Temporal Patterns and Frequencies...")
    frequency_analyzer = AlertFrequencyAnalyzer()
    
    temporal_analysis = frequency_analyzer.analyze_temporal_patterns(alerts_df)
    frequency_models = frequency_analyzer.build_frequency_prediction_model(alerts_df)
    
    print(f"   ✓ Identified {len(temporal_analysis['anomaly_dates'])} anomalous days")
    print(f"   ✓ Frequency prediction model accuracy: {frequency_models['random_forest']['test_score']:.3f}")
    
    # Step 4: Alert Prediction System
    print("\n4. Building Alert Prediction Models...")
    prediction_system = AlertPredictionSystem()
    
    # Prepare prediction data
    prediction_data = prediction_system.prepare_prediction_features(alerts_df, metrics_df, logs_df)
    
    # Build alert likelihood model
    likelihood_model = prediction_system.build_alert_likelihood_model(prediction_data)
    print(f"   ✓ Alert likelihood model accuracy: {likelihood_model['test_score']:.3f}")
    
    # Build system attribution model
    attribution_model = prediction_system.build_system_attribution_model(prediction_data)
    print(f"   ✓ System attribution model accuracy: {attribution_model['test_accuracy']:.3f}")
    
    # Step 5: Visualization and Reporting
    print("\n5. Creating Visualizations and Reports...")
    dashboard = AIOpsVisualizationDashboard()
    
    # Create visualizations
    clustering_fig = dashboard.create_clustering_visualization(alerts_df, cluster_labels, features)
    temporal_fig = dashboard.create_temporal_analysis_dashboard(temporal_analysis)
    importance_fig = dashboard.create_prediction_model_performance(likelihood_model)
    
    # Generate comprehensive report
    comprehensive_report = dashboard.generate_comprehensive_report(
        alerts_df, cluster_analysis, temporal_analysis, 
        {'likelihood_model': likelihood_model, 'attribution_model': attribution_model}
    )
    
    print("   ✓ Visualizations created")
    print("   ✓ Comprehensive report generated")
    
    # Step 6: Display Key Results
    print("\n" + "=" * 60)
    print("📊 KEY ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 ALERT SUMMARY:")
    print(f"   • Total Alerts: {comprehensive_report['summary']['total_alerts']}")
    print(f"   • Unique Systems: {comprehensive_report['summary']['unique_systems']}")
    print(f"   • Average Resolution Time: {comprehensive_report['summary']['avg_resolution_time']:.1f} minutes")
    
    print(f"\n🔍 CLUSTERING INSIGHTS:")
    for cluster_id, analysis in list(cluster_analysis.items())[:3]:  # Show top 3 clusters
        print(f"   • {cluster_id}: {analysis['size']} alerts ({analysis['percentage']:.1f}%)")
        print(f"     - Top criticality: {max(analysis['criticality_distribution'], key=analysis['criticality_distribution'].get)}")
        print(f"     - Top system: {max(analysis['system_distribution'], key=analysis['system_distribution'].get)}")
    
    print(f"\n⏰ TEMPORAL PATTERNS:")
    peak_hour = max(temporal_analysis['hourly_patterns'], key=temporal_analysis['hourly_patterns'].get)
    peak_day = max(temporal_analysis['weekly_patterns'], key=temporal_analysis['weekly_patterns'].get)
    print(f"   • Peak Hour: {peak_hour}:00 ({temporal_analysis['hourly_patterns'][peak_hour]} alerts)")
    print(f"   • Peak Day: {peak_day} ({temporal_analysis['weekly_patterns'][peak_day]} alerts)")
    print(f"   • Anomalous Days: {len(temporal_analysis['anomaly_dates'])}")
    
    print(f"\n🤖 PREDICTION MODEL PERFORMANCE:")
    print(f"   • Alert Likelihood Accuracy: {likelihood_model['test_score']:.3f}")
    print(f"   • System Attribution Accuracy: {attribution_model['test_accuracy']:.3f}")
    print(f"   • Top Feature: {likelihood_model['feature_names'][np.argmax(likelihood_model['feature_importance'])]}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, recommendation in enumerate(comprehensive_report['recommendations'], 1):
        print(f"   {i}. {recommendation}")
    
    print(f"\n✅ Analysis Complete!")
    print("   All models, visualizations, and reports are ready for deployment.")
    
    # Save results
    results = {
        'alerts_data': alerts_df,
        'metrics_data': metrics_df,
        'logs_data': logs_df,
        'clustering_results': clustering_results,
        'cluster_analysis': cluster_analysis,
        'temporal_analysis': temporal_analysis,
        'prediction_models': {
            'likelihood': likelihood_model,
            'attribution': attribution_model
        },
        'comprehensive_report': comprehensive_report
    }
    
    return results

if __name__ == "__main__":
    # Execute the complete AIOps alert analysis system
    results = main()
    
    # Optional: Save results to file
    print(f"\n💾 Saving results to 'aiops_analysis_results.pkl'...")
    with open('/home/jshayi2/data/redppo/aiops_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("   ✓ Results saved successfully!")
