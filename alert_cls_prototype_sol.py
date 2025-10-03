import pandas as pd
import json
import ast
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx
import difflib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('default')
sns.set_palette("husl")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

class ComprehensiveAlertProcessor:
    """
    Complete AIOps Alert Processing Pipeline:
    1. Graph-enriched alert processing
    2. Temporal deduplication
    3. Alert consolidation using graph relationships
    4. RCA classification
    """
    
    def __init__(self, alerts_csv_path, graph_json_path):
        self.alerts_csv_path = alerts_csv_path
        self.graph_json_path = graph_json_path
        
        # Data containers
        self.firing_alerts = []
        self.graph_relationships = []
        self.service_graph = nx.DiGraph()
        self.service_to_graph = {}
        
        # Processing results
        self.enriched_alerts = []
        self.deduplicated_alerts = []
        self.consolidated_alerts = []
        self.classified_alerts = []
        
        # ML models and feature engineering
        self.rca_classifier = None
        self.rca_scaler = None
        self.rca_label_encoder = None
        self.feature_selector = None
        self.best_features = []
        
        # Enhanced ML models
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.feature_importance = None
        
        # Engineering features
        self.scalers = {}
        self.feature_names = []
        
    def load_firing_alerts(self):
        """Load and filter firing alerts with temporal parsing"""
        print("Loading and processing alerts...")
        
        df_raw = pd.read_csv(self.alerts_csv_path, dtype=str, on_bad_lines='skip')
        
        print(f"Loaded {len(df_raw)} rows")
        
        df_raw = df_raw.rename(columns={
            df_raw.columns[0]: "attribute", 
            df_raw.columns[-1]: "value"
        })
        
        print(f"Raw data columns: {list(df_raw.columns)}")
        print(f"Sample attribute values: {df_raw['attribute'].unique()}")
        
        if 'status' in df_raw.columns:
            df_raw = df_raw.rename(columns={'status': 'status_original'})
        
        id_cols = [c for c in df_raw.columns if c not in ("attribute", "value")]
        print(f"ID columns: {id_cols}")
        
        df_pivoted = df_raw.pivot_table(
            index=id_cols, 
            columns='attribute', 
            values='value', 
            aggfunc='first'
        ).reset_index()
        
        df_pivoted = df_pivoted.rename(columns={
            'labels': 'payload_metadata',
            'annotations': 'alert_description'
        })
        
        print(f"Reshaped to {len(df_pivoted)} alerts")
        print(f"Pivoted data columns: {list(df_pivoted.columns)}")
        
        if 'status' in df_pivoted.columns:
            firing_df = df_pivoted[df_pivoted['status'].str.strip().str.lower() == 'firing']
            print(f"Found {len(firing_df)} firing alerts out of {len(df_pivoted)} total")
        else:
            firing_df = df_pivoted
        
        self.firing_alerts = firing_df.to_dict('records')
        
        print(f"Found {len(self.firing_alerts)} firing alerts")
        
        for alert in self.firing_alerts:
            self._parse_alert_payload_enhanced(alert)
            self._parse_temporal_info_enhanced(alert)
        
        return self.firing_alerts
    
    def _parse_alert_payload_enhanced(self, alert):
        """Parse payload metadata and alert description from pivoted alert data"""
        labels_str = alert.get('payload_metadata', '')
        if labels_str:
            try:
                labels = ast.literal_eval(labels_str)
                
                if 'cluster' not in alert:
                    alert['cluster'] = labels.get('cluster', '')
                if 'namespace' not in alert:
                    alert['namespace'] = labels.get('namespace', '')
                alert['pod'] = labels.get('pod', '')
                alert['node'] = labels.get('node', '')
                if 'anomaly_resource_type' not in alert:
                    alert['anomaly_resource_type'] = labels.get('anomaly_resource_type', '')
                alert['workload_type'] = labels.get('workload_type', '')
                alert['platform'] = labels.get('platform', '')
                if 'anomaly_entity_type' not in alert:
                    alert['anomaly_entity_type'] = labels.get('anomaly_entity_type', '')
                if 'alert_category' not in alert:
                    alert['alert_category'] = labels.get('alert_category', '')
                if 'alert_subcategory' not in alert:
                    alert['alert_subcategory'] = labels.get('alert_subcategory', '')
                if 'alert_name' not in alert:
                    alert['alertname'] = labels.get('alertname', '')
                
            except Exception as e:
                print(f"Warning: Could not parse labels for alert {alert.get('_id', 'unknown') }")
        
        annotations_str = alert.get('alert_description', '')
        if annotations_str:
            try:
                annotations = ast.literal_eval(annotations_str)
                alert['description'] = annotations.get('description', '')
            except Exception as e:
                print(f"Warning: Could not parse annotations for alert {alert.get('_id', 'unknown')}")
        
        payload_consolidated = {
            'labels': ast.literal_eval(labels_str) if labels_str else {},
            'annotations': ast.literal_eval(annotations_str) if annotations_str else {}
        }
        alert['payload_consolidated'] = str(payload_consolidated)
    
    def _parse_temporal_info_enhanced(self, alert):
        """Parse temporal information for deduplication from pivoted data"""
        try:
            # Try multiple possible column names for start time
            starts_at = alert.get('startsAt') or alert.get('starts_at', '')
            if starts_at:
                alert['start_datetime'] = pd.to_datetime(starts_at)
                alert['start_timestamp'] = alert['start_datetime'].timestamp()
                alert['start_hour'] = alert['start_datetime'].hour
                alert['start_minute'] = alert['start_datetime'].minute
            else:
                # Fallback to original columns if available
                starts_at = alert.get('starts_at', '')
                if starts_at:
                    alert['start_datetime'] = pd.to_datetime(starts_at)
                    alert['start_timestamp'] = alert['start_datetime'].timestamp()
                    alert['start_hour'] = alert['start_datetime'].hour
                    alert['start_minute'] = alert['start_datetime'].minute
            
            # Check if ongoing (firing alerts typically have placeholder end times)
            ends_at = alert.get('endsAt') or alert.get('ends_at', '')
            alert['is_ongoing'] = ends_at in ['1-01-01 00:00:00.000', '0001-01-01T00:00:00Z', '', '2025-01-01T00:00:00Z']
            
        except Exception as e:
            print(f"Warning: Could not parse timestamps for alert {alert.get('_id', 'unknown')}")
            alert['start_datetime'] = None
            alert['start_timestamp'] = 0
            alert['is_ongoing'] = True
    
    def load_graph_data(self):
        """Load graph data and build service mappings"""
        print("Loading graph data...")
        
        with open(self.graph_json_path, 'r') as f:
            self.graph_relationships = json.load(f)
        
        print(f"Loaded {len(self.graph_relationships)} graph relationships")
        
        # Build service mappings
        self.service_graph = nx.DiGraph()
        
        for rel in self.graph_relationships:
            # Map using source_properties.name -> alert.service_name
            source_props = rel.get('source_properties') or {}
            target_props = rel.get('target_properties') or {}
            
            source_service_name = source_props.get('name', '')
            target_service_name = target_props.get('name', '')
            
            if source_service_name:
                self.service_to_graph[source_service_name] = {
                    'graph_name': rel.get('source_name', ''),
                    'properties': source_props,
                    'type': rel.get('source_label', ''),
                    'environment': source_props.get('environment', ''),
                    'namespace': source_props.get('namespace', ''),
                    'cluster': source_props.get('cluster', '')
                }
                
                self.service_graph.add_node(source_service_name, **source_props)
            
            if target_service_name:
                self.service_to_graph[target_service_name] = {
                    'graph_name': rel.get('target_name', ''),
                    'properties': target_props,
                    'type': rel.get('target_label', ''),
                    'environment': target_props.get('environment', ''),
                    'namespace': target_props.get('namespace', ''),
                    'cluster': target_props.get('cluster', '')
                }
                
                self.service_graph.add_node(target_service_name, **target_props)
            
            # Add relationship edge
            rel_type = rel.get('relationship_type', '')
            if source_service_name and target_service_name and rel_type:
                self.service_graph.add_edge(
                    source_service_name,
                    target_service_name,
                    relationship_type=rel_type
                )
        
        print(f"Built service graph with {len(self.service_to_graph)} services")
        return self.service_to_graph
    
    def _find_fuzzy_service_match(self, alert_service_name):
        """Find best fuzzy match for service name"""
        if not alert_service_name:
            return None
        
        best_match = None
        best_score = 0.0
        
        # Try different matching strategies
        for graph_service_name in self.service_to_graph.keys():
            score = 0.0
            
            # Strategy 1: Direct substring match
            if alert_service_name.lower() in graph_service_name.lower():
                score = 0.8
            elif graph_service_name.lower() in alert_service_name.lower():
                score = 0.7
            
            # Strategy 2: Similarity scoring
            similarity = difflib.SequenceMatcher(None, alert_service_name.lower(), graph_service_name.lower()).ratio()
            score = max(score, similarity * 0.6)
            
            # Strategy 3: Check for common patterns in Kubernetes naming
            if 'aks-' in alert_service_name and 'aks-' in graph_service_name:
                # Extract cluster parts
                alert_parts = alert_service_name.split('-')[1:3] if len(alert_service_name.split('-')) >= 3 else []
                graph_parts = graph_service_name.split('-')[1:3] if len(graph_service_name.split('-')) >= 3 else []
                if alert_parts and graph_parts:
                    alert_key = '-'.join(alert_parts)
                    graph_key = '-'.join(graph_parts)
                    if alert_key == graph_key:
                        score = 0.9
            
            if score > best_score and score > 0.5:  # Minimum threshold
                best_score = score
                best_match = graph_service_name
        
        return best_match
    
    def enrich_alerts_with_graph_context(self):
        """Enrich alerts with graph relationship context"""
        print("Enriching alerts with graph context...")
        
        enriched_count = 0
        fuzzy_matches = 0
        
        for alert in self.firing_alerts:
            service_name = str(alert.get('service_name') or '').strip()
            alert['has_graph_context'] = False
            alert['calls_services'] = []
            alert['called_by_services'] = []
            alert['belongs_to_services'] = []
            alert['owns_services'] = []
            alert['service_centrality'] = 0.0
            alert['matched_graph_service'] = service_name
            
            # Direct match first
            if service_name in self.service_to_graph:
                matched_service = service_name
                alert['match_type'] = 'exact'
            else:
                # Fuzzy matching fallback
                matched_service = self._find_fuzzy_service_match(service_name)
                if matched_service:
                    fuzzy_matches += 1
                    alert['matched_graph_service'] = matched_service
                    alert['match_type'] = 'fuzzy'
                else:
                    enriched_count += 0  # Skip counting as enriched
                    continue
            
            if matched_service in self.service_to_graph:
                graph_info = self.service_to_graph[matched_service]
                alert['has_graph_context'] = True
                alert['graph_type'] = graph_info['type']
                alert['graph_environment'] = graph_info['environment']
                alert['graph_namespace'] = graph_info['namespace']
                alert['graph_cluster'] = graph_info['cluster']
                
                try:
                    centrality = nx.degree_centrality(self.service_graph)[matched_service]
                    alert['service_centrality'] = centrality
                except:
                    alert['service_centrality'] = 0.0
                
                for successor in self.service_graph.successors(matched_service):
                    edge_data = self.service_graph.get_edge_data(matched_service, successor)
                    if edge_data:
                        rel_type = edge_data.get('relationship_type')
                        if rel_type == 'CALLS':
                            alert['calls_services'].append(successor)
                        elif rel_type == 'BELONGS_TO':
                            alert['belongs_to_services'].append(successor)
                
                for predecessor in self.service_graph.predecessors(matched_service):
                    edge_data = self.service_graph.get_edge_data(predecessor, matched_service)
                    if edge_data:
                        rel_type = edge_data.get('relationship_type')
                        if rel_type == 'CALLS':
                            alert['called_by_services'].append(predecessor)
                        elif rel_type == 'BELONGS_TO':
                            alert['owns_services'].append(predecessor)
                
                enriched_count += 1
        
        self.enriched_alerts = self.firing_alerts
        print(f"Enriched {enriched_count}/{len(self.firing_alerts)} alerts with graph context")
        print(f"Fuzzy matches used: {fuzzy_matches}")
        return self.enriched_alerts
    
    def temporal_deduplication(self, time_window_minutes=5):
        """
        Remove temporal duplicates based on:
        1. Same service + resource type + severity
        2. Within time window
        3. Similar descriptions
        """
        print(f"Performing temporal deduplication (window: {time_window_minutes} minutes)...")
        
        # Sort alerts by timestamp
        sorted_alerts = sorted(
            [a for a in self.enriched_alerts if a.get('start_datetime')],
            key=lambda x: x['start_datetime']
        )
        
        deduplicated = []
        duplicate_groups = []
        processed_ids = set()
        
        for i, alert in enumerate(sorted_alerts):
            if alert.get('_id') in processed_ids:
                continue
            
            # Find duplicates within time window
            duplicates = [alert]
            
            for j in range(i + 1, len(sorted_alerts)):
                other_alert = sorted_alerts[j]
                
                if other_alert.get('_id') in processed_ids:
                    continue
                
                # Check time window
                time_diff = (other_alert['start_datetime'] - alert['start_datetime']).total_seconds() / 60
                if time_diff > time_window_minutes:
                    break  # Beyond time window
                
                # Check if it's a duplicate
                if self._is_temporal_duplicate(alert, other_alert):
                    duplicates.append(other_alert)
                    processed_ids.add(other_alert.get('_id'))
            
            if len(duplicates) > 1:
                # Create consolidated alert from duplicates
                consolidated = self._consolidate_duplicate_alerts(duplicates)
                deduplicated.append(consolidated)
                duplicate_groups.append({
                    'representative': consolidated,
                    'duplicates': duplicates[1:],
                    'count': len(duplicates)
                })
            else:
                # Single alert
                alert['is_duplicate'] = False
                alert['duplicate_count'] = 1
                deduplicated.append(alert)
            
            processed_ids.add(alert.get('_id'))
        
        self.deduplicated_alerts = deduplicated
        
        original_count = len(self.enriched_alerts)
        final_count = len(deduplicated)
        reduction = original_count - final_count
        reduction_pct = (reduction / original_count * 100) if original_count > 0 else 0
        
        # Validation checks
        if reduction_pct > 50:
            print(f"WARNING: High deduplication rate ({reduction_pct:.1f}%) - may be over-aggressive")
        elif reduction_pct < 20:
            print(f"INFO: Conservative deduplication ({reduction_pct:.1f}%) - consider tightening criteria")
        
        print(f"Temporal deduplication: {original_count} → {final_count} alerts ({reduction} duplicates removed, {reduction_pct:.1f}%)")
        return deduplicated, duplicate_groups
    
    def _is_temporal_duplicate(self, alert1, alert2):
        """Check if two alerts are temporal duplicates"""
        # Same service
        if alert1.get('service_name') != alert2.get('service_name'):
            return False
        
        # Same resource type
        if alert1.get('anomaly_resource_type') != alert2.get('anomaly_resource_type'):
            return False
        
        # Same severity
        if alert1.get('severity') != alert2.get('severity'):
            return False
        
        # Similar descriptions (if available)
        desc1 = alert1.get('description', '').lower()
        desc2 = alert2.get('description', '').lower()
        
        if desc1 and desc2:
            # Remove specific identifiers for comparison
            desc1_clean = self._clean_description(desc1)
            desc2_clean = self._clean_description(desc2)
            
            similarity = difflib.SequenceMatcher(None, desc1_clean, desc2_clean).ratio()
            if similarity < 0.85:  # Increased threshold for more strict deduplication
                return False
        
        return True
    
    def _clean_description(self, description):
        """Clean description for comparison by removing specific identifiers"""
        import re
        # Remove pod names, node names, IDs
        cleaned = re.sub(r'\b\w+-[a-f0-9]{8,}-\w+\b', '[POD]', description)
        cleaned = re.sub(r'\baks-\w+-\w+-\w+\b', '[NODE]', cleaned)
        cleaned = re.sub(r'\b[a-f0-9]{8,}\b', '[ID]', cleaned)
        return cleaned
    
    def _consolidate_duplicate_alerts(self, duplicates):
        """Consolidate duplicate alerts into a single representative"""
        representative = duplicates[0].copy()
        
        # Mark as consolidated
        representative['is_duplicate'] = True
        representative['duplicate_count'] = len(duplicates)
        
        # Collect affected resources
        pods = set()
        nodes = set()
        namespaces = set()
        
        for alert in duplicates:
            if alert.get('pod'):
                pods.add(alert['pod'])
            if alert.get('node'):
                nodes.add(alert['node'])
            if alert.get('namespace'):
                namespaces.add(alert['namespace'])
        
        representative['affected_pods'] = ', '.join(list(pods)[:10])
        representative['affected_nodes'] = ', '.join(list(nodes)[:10])
        representative['affected_namespaces'] = ', '.join(list(namespaces))
        
        # Time range
        start_times = [d['start_datetime'] for d in duplicates if d.get('start_datetime')]
        if start_times:
            representative['first_occurrence'] = min(start_times)
            representative['last_occurrence'] = max(start_times)
            representative['duration_minutes'] = (max(start_times) - min(start_times)).total_seconds() / 60
        
        return representative
    
    def consolidate_alerts_by_relationships(self):
        """Consolidate alerts using graph relationships for RCA grouping"""
        print("Consolidating alerts by graph relationships...")
        
        consolidated_groups = []
        processed_alerts = set()
        
        for alert in self.deduplicated_alerts:
            if alert.get('_id') in processed_alerts:
                continue
            
            # Find related alerts through graph relationships
            related_alerts = self._find_relationship_group(alert)
            
            if len(related_alerts) > 1:
                # Create consolidated group
                consolidated_group = {
                    'group_id': len(consolidated_groups) + 1,
                    'primary_alert': alert,
                    'related_alerts': related_alerts[1:],
                    'total_alerts': len(related_alerts),
                    'relationship_type': self._determine_group_relationship_type(related_alerts),
                    'rca_category': self._predict_rca_category(related_alerts)
                }
                
                consolidated_groups.append(consolidated_group)
                
                # Mark as processed
                for rel_alert in related_alerts:
                    processed_alerts.add(rel_alert.get('_id'))
            else:
                # Single alert - still add for classification
                single_group = {
                    'group_id': len(consolidated_groups) + 1,
                    'primary_alert': alert,
                    'related_alerts': [],
                    'total_alerts': 1,
                    'relationship_type': 'isolated',
                    'rca_category': self._predict_rca_category([alert])
                }
                
                consolidated_groups.append(single_group)
                processed_alerts.add(alert.get('_id'))
        
        self.consolidated_alerts = consolidated_groups
        
        multi_alert_groups = [g for g in consolidated_groups if g['total_alerts'] > 1]
        print(f"Created {len(consolidated_groups)} consolidated groups ({len(multi_alert_groups)} multi-alert groups)")
        
        return consolidated_groups
    
    def _find_relationship_group(self, primary_alert):
        """Find alerts related through graph relationships"""
        service_name = str(primary_alert.get('service_name') or '').strip()
        related_alerts = [primary_alert]
        
        if not primary_alert.get('has_graph_context'):
            return related_alerts
        
        # Get related services
        related_services = set()
        related_services.update(primary_alert.get('calls_services', []))
        related_services.update(primary_alert.get('called_by_services', []))
        related_services.update(primary_alert.get('belongs_to_services', []))
        related_services.update(primary_alert.get('owns_services', []))
        
        # Find alerts for related services
        for alert in self.deduplicated_alerts:
            if (alert.get('_id') != primary_alert.get('_id') and 
                str(alert.get('service_name', '')).strip() in related_services):
                
                # Check temporal proximity (within 30 minutes for relationship grouping)
                if (primary_alert.get('start_datetime') and alert.get('start_datetime')):
                    time_diff = abs((alert['start_datetime'] - primary_alert['start_datetime']).total_seconds() / 60)
                    if time_diff <= 30:  # 30-minute window for relationship grouping
                        related_alerts.append(alert)
        
        return related_alerts
    
    def _determine_group_relationship_type(self, alerts):
        """Determine the primary relationship type for a group"""
        if len(alerts) <= 1:
            return 'isolated'
        
        primary_alert = alerts[0]
        
        # Check if it's a service dependency chain
        calls_services = primary_alert.get('calls_services', [])
        called_by_services = primary_alert.get('called_by_services', [])
        
        related_service_names = [a.get('service_name') for a in alerts[1:]]
        
        if any(service in calls_services for service in related_service_names):
            return 'service_dependency_downstream'
        elif any(service in called_by_services for service in related_service_names):
            return 'service_dependency_upstream'
        elif any(service in primary_alert.get('belongs_to_services', []) for service in related_service_names):
            return 'ownership_relationship'
        else:
            return 'infrastructure_related'
    
    def _predict_rca_category(self, alerts):
        """Predict RCA category based on alert characteristics and relationships"""
        if not alerts:
            return 'unknown'
        
        primary_alert = alerts[0]
        
        # Analyze resource types
        resource_types = [a.get('anomaly_resource_type', '') for a in alerts]
        resource_counter = Counter(resource_types)
        dominant_resource = resource_counter.most_common(1)[0][0] if resource_counter else ''
        
        # Analyze service centrality
        avg_centrality = np.mean([a.get('service_centrality', 0) for a in alerts])
        
        # Analyze relationship patterns
        has_dependencies = any(a.get('calls_services') or a.get('called_by_services') for a in alerts)
        
        # Enhanced rule-based RCA prediction
        if len(alerts) > 5 and ('network' in dominant_resource or 'disk' in dominant_resource):
            return 'infrastructure_wide_failure'
        elif len(alerts) > 2 and has_dependencies:
            return 'cascading_service_failure'
        elif ('cpu' in dominant_resource or 'memory' in dominant_resource) and avg_centrality > 0.1:
            return 'resource_exhaustion_critical_service'
        elif 'memory' in dominant_resource and len(alerts) > 1:
            return 'memory_leak_or_pressure'
        elif len(alerts) > 1:
            # Multi-alert events suggest systemic issues
            if avg_centrality > 0.05:
                return 'distributed_service_issue'
            else:
                return 'correlated_incidents'
        elif 'disk' in dominant_resource or 'storage' in dominant_resource:
            return 'storage_issue'
        elif 'network' in dominant_resource or 'traffic' in dominant_resource:
            return 'network_performance_issue'
        elif avg_centrality > 0.2:
            return 'critical_service_degradation'
        else:
            return 'isolated_service_issue'
    
    def prepare_advanced_features(self):
        """Prepare enhanced features for ML-based RCA classification"""
        print("Preparing advanced features for RCA classification...")
        
        features = []
        labels = []
        
        for group in self.consolidated_alerts:
            primary_alert = group['primary_alert']
            
            # Basic alert characteristics
            alert_count = group['total_alerts']
            has_graph_context = 1 if primary_alert.get('has_graph_context') else 0
            service_centrality = primary_alert.get('service_centrality', 0)
            is_duplicate = 1 if primary_alert.get('is_duplicate', False) else 0
            duplicate_count = primary_alert.get('duplicate_count', 1)
            
            # Enhanced resource analysis
            resource_type = str(primary_alert.get('anomaly_resource_type', '')).lower()
            resource_features = {
                'resource_memory': 1 if 'memory' in resource_type else 0,
                'resource_cpu': 1 if 'cpu' in resource_type else 0,
                'resource_network': 1 if 'network' in resource_type else 0,
                'resource_disk': 1 if 'disk' in resource_type else 0,
                'resource_traffic': 1 if 'traffic' in resource_type else 0,
                'resource_storage': 1 if 'storage' in resource_type else 0,
            }
            
            # Relationship complexity features
            calls_count = len(primary_alert.get('calls_services', []))
            called_by_count = len(primary_alert.get('called_by_services', []))
            belongs_to_count = len(primary_alert.get('belongs_to_services', []))
            owns_count = len(primary_alert.get('owns_services', []))
            
            # Advanced relationship features
            total_dependencies = calls_count + called_by_count
            dependency_ratio = calls_count / (called_by_count + 1)  # Avoid division by zero
            relationship_complexity = len(set(
                primary_alert.get('calls_services', []) + 
                primary_alert.get('called_by_services', [])
            ))
            
            # Temporal features with engineering
            hour_of_day = primary_alert.get('start_hour', 0)
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
            duration_minutes = primary_alert.get('duration_minutes', 0)
            weekday = pd.to_datetime(primary_alert.get('starts_at', '')).weekday() if primary_alert.get('starts_at') else 0
            
            # Service environment features
            cluster_name = str(primary_alert.get('cluster', '')).lower()
            namespace = str(primary_alert.get('namespace', '')).lower()
            environment_features = {
                'prod_environment': 1 if 'prod' in cluster_name else 0,
                'staging_environment': 1 if 'staging' in cluster_name else 0,
                'dev_environment': 1 if 'dev' in cluster_name else 0,
                'network_namespace': 1 if 'network' in namespace else 0,
                'default_namespace': 1 if namespace == 'default' else 0,
            }
            
            # Entity type analysis
            entity_type = str(primary_alert.get('anomaly_entity_type', '')).lower()
            entity_features = {
                'entity_type_pod': 1 if 'pod' in entity_type else 0,
                'entity_type_service': 1 if 'service' in entity_type else 0,
                'entity_type_node': 1 if 'node' in entity_type else 0,
                'entity_type_deployment': 1 if 'deployment' in entity_type else 0,
            }
            
            # Severity and impact features
            severity = str(primary_alert.get('severity', '')).lower()
            severity_features = {
                'severity_warning': 1 if 'warning' in severity else 0,
                'severity_critical': 1 if 'critical' in severity else 0,
                'severity_error': 1 if 'error' in severity else 0,
                'severity_info': 1 if 'info' in severity else 0,
            }
            
            # Alert description analysis (TF-IDF like features)
            description = str(str(primary_alert.get('description', '')).lower())
            description_features = {
                'desc_len': len(description),
                'desc_has_high': 1 if 'high' in description else 0,
                'desc_has_failed': 1 if 'fail' in description else 0,
                'desc_has_error': 1 if 'error' in description else 0,
                'desc_has_threshold': 1 if 'threshold' in description else 0,
                'desc_has_response': 1 if 'response' in description else 0,
                'desc_word_count': len(description.split()),
            }
            
            # Graph topology features
            graph_features = {
                'graph_clustering_coeff': 0.0,  # Placeholder for cluster coefficient
                'graph_betweenness': service_centrality,  # Use centrality as proxy
                'isolated_service': 1 if not has_graph_context else 0,
                'highly_connected': 1 if service_centrality > 0.1 else 0,
            }
            
            # Alert intensity features
            intensity_features = {
                'alert_frequency_per_hour': alert_count / max(duration_minutes / 60, 1),
                'avg_alerts_per_service': alert_count / max(total_dependencies + 1, 1),
                'duplicate_intensity': duplicate_count / alert_count if alert_count > 0 else 0,
            }
            
            # Combine all features
            feature_vector = {
                # Core features
                'alert_count': alert_count,
                'has_graph_context': has_graph_context,
                'service_centrality': service_centrality,
                'is_duplicate': is_duplicate,
                'duplicate_count': duplicate_count,
                
                # Resource features
                **resource_features,
                
                # Relationship features
                'calls_count': calls_count,
                'called_by_count': called_by_count,
                'belongs_to_count': belongs_to_count,
                'owns_count': owns_count,
                'total_dependencies': total_dependencies,
                'dependency_ratio': dependency_ratio,
                'relationship_complexity': relationship_complexity,
                
                # Temporal features
                'hour_of_day': hour_of_day,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'duration_minutes': duration_minutes,
                'weekday': weekday,
                
                # Environment features
                **environment_features,
                
                # Entity features  
                **entity_features,
                
                # Severity features
                **severity_features,
                
                # Description features
                **description_features,
                
                # Graph features
                **graph_features,
                
                # Intensity features
                **intensity_features,
            }
            
            features.append(feature_vector)
            labels.append(group['rca_category'])
        
        self.classification_features = pd.DataFrame(features)
        self.classification_labels = labels
        self.feature_names = self.classification_features.columns.tolist()
        
        print(f"Prepared {len(features)} advanced feature vectors with {len(self.feature_names)} features")
        print(f"Feature categories: Core, Resource, Relationship, Temporal, Environment, Entity, Severity, Description, Graph, Intensity")
        return self.classification_features, self.classification_labels
    
    def feature_selection_and_preprocessing(self, X, y):
        """Advanced feature selection and preprocessing"""
        print("Performing feature selection and preprocessing...")
        
        # Fill missing values
        X_filled = X.fillna(0)
        
        # Handle infinite values
        X_filled = X_filled.replace([np.inf, -np.inf], 0)
        
        # Encode labels
        self.rca_label_encoder = LabelEncoder()
        y_encoded = self.rca_label_encoder.fit_transform(y)
        
        # Split data for feature selection
        X_train, X_test, y_train, y_test = train_test_split(
            X_filled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Multiple preprocessing approaches
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        # Feature selection strategies - adjust k based on available features
        total_features = X_filled.shape[1]
        selection_methods = {
            'univariate': SelectKBest(score_func=mutual_info_classif, k=min(20, max(5, total_features//3))),
            'model_based': SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42)),
            'recursive': RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=min(15, max(3, total_features//4)))
        }
        
        best_score = 0
        best_features = X_filled.columns.tolist()
        best_scaler = 'standard'
        
        # Default fallback scaler
        fallback_scaler = StandardScaler()
        self.rca_scaler = fallback_scaler
        
        for scaler_name, scaler in self.scalers.items():
            for method_name, selector in selection_methods.items():
                try:
                    # Scale features
                    X_scaled = scaler.fit_transform(X_train)
                    
                    # Select features
                    if hasattr(selector, 'fit_transform'):
                        X_selected = selector.fit_transform(X_scaled, y_train)
                    else:
                        selector.fit(X_scaled, y_train)
                        X_selected = selector.transform(X_scaled)
                    
                    selected_features = X_filled.columns[selector.get_support()].tolist()
                    
                    # Quick performance check with RandomForest
                    temp_rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    score = temp_rf.fit(X_selected, y_train).score(X_selected, y_train)
                    
                    if score > best_score:
                        best_score = score
                        best_scaler = scaler_name
                        best_features = selected_features
                        self.feature_selector = selector
                        # Fit scaler only on selected features for consistency
                        self.rca_scaler = type(scaler)()  # Create new instance
                        X_train_selected = X_train[selected_features]
                        self.rca_scaler.fit(X_train_selected)
                        
                except Exception as e:
                    print(f"Feature selection method {method_name} with {(scaler_name)}, attempting fallback...")
                    continue
        
        print(f"Best feature selection: {len(best_features)} features selected")
        print(f"Best scaling: {best_scaler}")
        
        # Fallback if no methods worked
        if self.rca_scaler is None:
            print("Using fallback scaler (StandardScaler)")
            self.rca_scaler = StandardScaler()
            # If feature selection failed completely, use top features by variance
            if len(best_features) == total_features:
                print("Feature selection failed, using top variance features...")
                variances = X_filled.var().sort_values(ascending=False)
                best_features = variances.head(min(20, total_features)).index.tolist()
                print(f"Selected top {len(best_features)} features by variance")
            
            # Fit scaler on selected features
            X_filled_selected = X_filled[best_features]
            self.rca_scaler.fit(X_filled_selected)
        
        self.best_features = best_features
        return self.rca_scaler.transform(X_filled[best_features]), y_encoded
    
    def train_multiple_models(self, X_scaled, y_encoded):
        """Train multiple ML models and compare performance"""
        print("Training multiple ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Define models with hyperparameters
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=3,
                min_samples_leaf=1, random_state=42, class_weight='balanced'
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300, max_depth=15, min_samples_split=3,
                min_samples_leaf=1, random_state=42, class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                subsample=0.8, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42, class_weight='balanced'
            ),
            'NeuralNetwork': MLPClassifier(
                hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                alpha=0.001, random_state=42, max_iter=1000
            )
        }
        
        # Add XGBoost and LightGBM if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, random_state=42, eval_metric='logloss'
            )
        
        if LIGHTGBM_AVAILABLE:
            models_config['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, random_state=42, verbose=-1
            )
        
        # Train and evaluate all models
        best_score = 0
        
        for model_name, model in models_config.items():
            try:
                print(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
                
                # Store model and performance
                self.models[model_name] = model
                self.model_performance[model_name] = {
                    'train_score': train_score,
                    'cv_score': test_score,
                    'test_score': model.score(X_test, y_test)
                }
                
                print(f"{model_name}: Train={train_score:.3f}, CV={test_score:.3f}, Test={model.score(X_test, y_test):.3f}")
                
                # Update best model
                if test_score > best_score:
                    best_score = test_score
                    self.best_model = model_name
                    self.rca_classifier = model
                    
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        print(f"\nBest performing model: {self.best_model} (CV Score: {best_score:.3f})")
        return self.rca_classifier
    
    def extract_feature_importance(self, X_feature_names=None):
        """Extract and analyze feature importance across models"""
        print("Analyzing feature importance...")
        
        importance_data = []
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance_data.append({
                        'model': model_name,
                        'importance': importances,
                        'length': len(importances)
                    })
                elif hasattr(model, 'coef_'):
                    # For logistic regression
                    coef_importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
                    importance_data.append({
                        'model': model_name,
                        'importance': coef_importances,
                        'length': len(coef_importances)
                    })
            except Exception as e:
                print(f"Could not extract importance from {model_name}: {e}")
                continue
        
        if importance_data:
            # Use the most common length (should be the length of best_features)
            lengths = [data['length'] for data in importance_data]
            most_common_length = max(set(lengths), key=lengths.count)
            
            print(f"Most common importance array length: {most_common_length}")
            
            # Filter data to only include arrays of the correct length
            filtered_data = [data for data in importance_data if data['length'] == most_common_length]
            
            if filtered_data:
                # Average importance across models with same length
                avg_importance = np.mean([data['importance'] for data in filtered_data], axis=0)
                
                # Use best_features as feature names since that's what models were trained on
                if X_feature_names and len(X_feature_names) == most_common_length:
                    feature_names = X_feature_names
                elif hasattr(self, 'best_features') and len(self.best_features) == most_common_length:
                    feature_names = self.best_features
                else:
                    # Fallback to generic names
                    feature_names = [f'feature_{i}' for i in range(most_common_length)]
                
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'avg_importance': avg_importance,
                    'model_count': len(filtered_data)
                }).sort_values('avg_importance', ascending=False)
                
                print("Top 15 Most Important Features (Averaged Across Models):")
                print(self.feature_importance.head(15))
            else:
                print("No compatible importance data found across models")
                self.feature_importance = None
        else:
            print("No importance data extracted from any models")
            self.feature_importance = None
            
        return self.feature_importance
    
    def train_advanced_classifier(self):
        """Advanced multi-model RCA classification training"""
        print("Starting advanced RCA classifier training...")
        
        if not hasattr(self, 'classification_features'):
            self.prepare_advanced_features()
        
        X = self.classification_features
        y = self.classification_labels
        
        # Feature selection and preprocessing
        X_scaled, y_encoded = self.feature_selection_and_preprocessing(X, y)
        
        # Train multiple models
        self.rca_classifier = self.train_multiple_models(X_scaled, y_encoded)
        
        # Extract feature importance
        self.extract_feature_importance(self.best_features)
        
        return self.rca_classifier
    
    def classify_alerts_for_rca(self):
        """Classify consolidated alerts for RCA using advanced models"""
        print("Classifying alerts for RCA with advanced models...")
        
        if not self.rca_classifier:
            print("No model available. Training advanced classifier...")
            self.train_advanced_classifier()
        
        # Prepare features for prediction
        try:
            # Try to use selected features first
            X_subset = self.classification_features[self.best_features].fillna(0)
        except (KeyError, AttributeError):
            # Fallback to all features if selection failed
            print("Using all features as fallback...")
            X_subset = self.classification_features.fillna(0)
        
        X_subset = X_subset.replace([np.inf, -np.inf], 0)
        X_scaled = self.rca_scaler.transform(X_subset)
        
        # Predict RCA categories using best model
        predictions = self.rca_classifier.predict(X_scaled)
        
        # Get prediction probabilities if available
        try:
            probabilities = self.rca_classifier.predict_proba(X_scaled)
        except:
            probabilities = None
        
        # Update consolidated alerts with predictions
        for i, group in enumerate(self.consolidated_alerts):
            predicted_category = self.rca_label_encoder.inverse_transform([predictions[i]])[0]
            
            # Calculate confidence
            if probabilities is not None:
                confidence = np.max(probabilities[i])
            else:
                # Use rule-based confidence as fallback
                confidence = 0.85 if group['total_alerts'] > 2 else 0.70
            
            group['ml_rca_category'] = predicted_category
            group['ml_confidence'] = confidence
            group['best_model'] = self.best_model
            group['final_rca_category'] = predicted_category
        
        self.classified_alerts = self.consolidated_alerts
        
        # Generate model comparison report
        self.generate_model_comparison_report()
        
        print("Advanced RCA classification completed")
        return self.classified_alerts
    
    def generate_model_comparison_report(self):
        """Generate model performance comparison report"""
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("=" * 60)
        
        if not self.model_performance:
            print("No model performance data available")
            return
        
        # Sort models by CV score
        sorted_models = sorted(
            self.model_performance.items(),
            key=lambda x: x[1]['cv_score'],
            reverse=True
        )
        
        print("Model Performance Rankings:")
        print("-" * 60)
        print(f"{'Rank':<4} {'Model':<15} {'Train':<8} {'CV':<8} {'Test':<8} {'Accuracy Score'}")
        print("-" * 60)
        
        for rank, (model_name, perf) in enumerate(sorted_models, 1):
            print(f"{rank:<4} {model_name:<15} {perf['train_score']:<8.3f} {perf['cv_score']:<8.3f} {perf['test_score']:<8.3f} {perf['cv_score']:.1%}")
        
        print("-" * 60)
        print(f"Best Model: {self.best_model}")
        print(f"Feature Count: {len(self.best_features)}")
        print(f"Total Models Trained: {len(self.models)}")
        
        # Feature importance insights
        if self.feature_importance is not None:
            print(f"\nTop 5 Feature Categories:")
            print("-" * 30)
            for i, row in self.feature_importance.head(5).iterrows():
                print(f"{i+1}. {row['feature']}: {row['avg_importance']:.3f}")
        
        print("=" * 60)
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE AIOPS ALERT PROCESSING & RCA CLASSIFICATION")
        print("="*70)
        
        # Processing statistics
        original_count = len(self.firing_alerts)
        enriched_count = len([a for a in self.enriched_alerts if a.get('has_graph_context')])
        exact_matches = len([a for a in self.enriched_alerts if a.get('match_type') == 'exact'])
        fuzzy_matches = len([a for a in self.enriched_alerts if a.get('match_type') == 'fuzzy'])
        deduplicated_count = len(self.deduplicated_alerts)
        consolidated_count = len(self.consolidated_alerts)
        
        print(f"\nPROCESSING PIPELINE RESULTS:")
        print(f"  Original firing alerts: {original_count}")
        print(f"  Graph-enriched alerts: {enriched_count} ({enriched_count/original_count*100:.1f}%)")
        print(f"    - Exact matches: {exact_matches}")
        print(f"    - Fuzzy matches: {fuzzy_matches}")
        print(f"  After deduplication: {deduplicated_count}")
        print(f"  Consolidated groups: {consolidated_count}")
        
        # Deduplication effectiveness
        dedup_reduction = ((original_count - deduplicated_count) / original_count * 100) if original_count > 0 else 0
        print(f"  Deduplication reduction: {dedup_reduction:.1f}%")
        
        # RCA classification results
        rca_categories = Counter(group['final_rca_category'] for group in self.classified_alerts)
        
        print(f"\nRCA CLASSIFICATION RESULTS:")
        for category, count in rca_categories.most_common():
            print(f"  {category}: {count} groups")
        
        # High-priority issues
        high_priority = [
            g for g in self.classified_alerts 
            if g['total_alerts'] > 2 or 'critical' in g['final_rca_category'] or 'cascading' in g['final_rca_category']
        ]
        
        print(f"\nHIGH-PRIORITY RCA GROUPS: {len(high_priority)}")
        for i, group in enumerate(sorted(high_priority, key=lambda x: x['total_alerts'], reverse=True)[:5], 1):
            primary = group['primary_alert']
            print(f"  {i}. {group['final_rca_category']} ({group['total_alerts']} alerts)")
            print(f"     Service: {primary.get('service_name', 'Unknown')}")
            print(f"     Resource: {primary.get('anomaly_resource_type', primary.get('resource_type', 'Unknown'))}")
            print(f"     Confidence: {group.get('ml_confidence', 0):.2f}")
        
        return {
            'original_alerts': original_count,
            'deduplicated_alerts': deduplicated_count,
            'consolidated_groups': consolidated_count,
            'rca_categories': dict(rca_categories),
            'high_priority_groups': len(high_priority)
        }
    
    def create_pipeline_performance_chart(self):
        """Create pipeline performance visualization"""
        print("Generating pipeline performance charts...")
        
        # Prepare data for visualization
        original_count = len(self.firing_alerts)
        enriched_count = len([a for a in self.enriched_alerts if a.get('has_graph_context')])
        deduplicated_count = len(self.deduplicated_alerts)
        consolidated_count = len(self.consolidated_alerts)
        
        stages = ['Original\nAlerts', 'Graph\nEnriched', 'After\nDeduplication', 'Consolidated\nGroups']
        counts = [original_count, enriched_count, deduplicated_count, consolidated_count]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔄 AIOps Alert Processing Pipeline Performance', fontsize=16, fontweight='bold')
        
        # 1. Pipeline Flow Chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(stages, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('📊 Alert Volume Through Pipeline Stages', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Alerts/Groups')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentage reduction annotations
        reduction_1 = ((original_count - enriched_count) / original_count) * 100
        reduction_2 = ((original_count - deduplicated_count) / original_count) * 100
        reduction_3 = ((original_count - consolidated_count) / original_count) * 100
        
        ax1.annotate(f'{reduction_2:.1f}%\nReduction', xy=(2, deduplicated_count),
                    xytext=(1.5, deduplicated_count + 100),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, ha='center', color='red', fontweight='bold')
        
        # 2. Processing Efficiency Chart
        enrichment_rate = (enriched_count / original_count) * 100
        deduplication_rate = ((original_count - deduplicated_count) / original_count) * 100
        
        categories = ['Graph\nEnrichment', 'Deduplication\nReduction']
        rates = [enrichment_rate, deduplication_rate]
        
        bars2 = ax2.bar(categories, rates, color=['#FFB347', '#87CEEB'], alpha=0.8)
        ax2.set_title('⚡ Processing Efficiency Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, max(rates) * 1.2)
        
        for bar, rate in zip(bars2, rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. RCA Category Distribution (Pie Chart)
        rca_categories = Counter(group['final_rca_category'] for group in self.classified_alerts)
        labels = list(rca_categories.keys())
        sizes = list(rca_categories.values())
        
        # Truncate long labels
        labels_short = [label.replace('_', ' ').title()[:15] for label in labels]
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels_short, autopct='%1.1f%%',
                                          startangle=90, colors=plt.cm.Set3.colors)
        ax3.set_title('🎯 RCA Classification Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 4. Model Performance Comparison
        if hasattr(self, 'model_performance') and self.model_performance:
            models = list(self.model_performance.keys())
            cv_scores = [self.model_performance[model]['cv_score'] * 100 for model in models]
            
            bars3 = ax4.barh(models, cv_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            ax4.set_title('🏆 Model Performance (CV Score)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Cross-Validation Accuracy (%)')
            
            # Highlight best model
            if hasattr(self, 'best_model'):
                best_idx = models.index(self.best_model)
                bars3[best_idx].set_color('#FFD700')  # Gold color for best
                bars3[best_idx].set_edgecolor('red')
                bars3[best_idx].set_linewidth(3)
            
            # Add value labels
            for bar, score in zip(bars3, cv_scores):
                width = bar.get_width()
                ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                        f'{score:.1f}%', ha='left', va='center', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Model Performance\nData Not Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('🏆 Model Performance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('pipeline_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def create_feature_importance_chart(self):
        """Create feature importance visualization"""
        if self.feature_importance is None:
            print("No feature importance data available for visualization")
            return None
            
        print("Generating feature importance charts...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('🧠 Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top Features Bar Chart
        top_features = self.feature_importance.head(15)
        
        bars = ax1.barh(range(len(top_features)), top_features['avg_importance'], 
                       color=plt.cm.plasma(np.linspace(0, 1, len(top_features))))
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels([f.replace('_', ' ').title() for f in top_features['feature']], fontsize=10)
        ax1.set_xlabel('Average Importance Score')
        ax1.set_title('📊 Top 15 Most Important Features', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, top_features['avg_importance'].max() * 1.1)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['avg_importance'])):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'{importance:.3f}', ha='left', va='center', fontweight='bold')
        
        # 2. Feature Categories Breakdown
        feature_categories = {
            'Core': ['alert_count', 'has_graph_context', 'service_centrality'],
            'Resource': ['resource_memory', 'resource_cpu', 'resource_network', 'resource_disk'],
            'Relationship': ['total_dependencies', 'relationship_complexity', 'belongs_to_count'],
            'Temporal': ['hour_of_day', 'duration_minutes', 'hour_sin', 'hour_cos'],
            'Environment': ['prod_environment', 'staging_environment', 'network_namespace'],
            'Description': ['desc_len', 'desc_word_count', 'desc_has_error'],
            'Graph': ['graph_betweenness', 'isolated_service', 'highly_connected'],
            'Intensity': ['alert_frequency_per_hour', 'duplicate_intensity', 'avg_alerts_per_service']
        }
        
        category_importance = {}
        for category, features in feature_categories.items():
            category_features = self.feature_importance[
                self.feature_importance['feature'].isin(features)]
            category_importance[category] = category_features['avg_importance'].sum()
        
        categories = list(category_importance.keys())
        importances = list(category_importance.values())
        
        wedges, texts, autotexts = ax2.pie(importances, labels=categories, autopct='%1.1f%%',
                                          startangle=90, colors=plt.cm.Set2.colors)
        ax2.set_title('📈 Feature Categories Breakdown', fontsize=14, fontweight='bold')
        
        # Make percentage text bold and white
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def create_temporal_analysis_chart(self):
        """Create temporal analysis visualization"""
        print("Generating temporal analysis charts...")
        
        # Prepare temporal data
        alerts_with_time = [a for a in self.enriched_alerts if a.get('start_datetime')]
        
        if not alerts_with_time:
            print("No temporal data available for visualization")
            return None
        
        # Extract temporal features
        hours = [a['start_hour'] for a in alerts_with_time]
        alerts_per_hour = Counter(hours)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('⏰ Temporal Analysis of Alert Patterns', fontsize=16, fontweight='bold')
        
        # 1. Alert Distribution by Hour
        hours_data = sorted(alerts_per_hour.items())
        hours_x, hours_y = zip(*hours_data)
        
        ax1.plot(hours_x, hours_y, marker='o', linewidth=3, markersize=8, color='#FF6B6B')
        ax1.fill_between(hours_x, hours_y, alpha=0.3, color='#FF6B6B')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Alerts')
        ax1.set_title('📊 Alert Volume by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)
        
        # Highlight peak hours
        max_hour = max(hours_data, key=lambda x: x[1])
        ax1.annotate(f'Peak: {max_hour[1]} alerts\nat {max_hour[0]:02d}:00', 
                    xy=(max_hour[0], max_hour[1]), xytext=(max_hour[0]+3, max_hour[1]+10),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, fontweight='bold')
        
        # 2. Alert Severity Distribution Over Time
        severity_hours = {}
        for alert in alerts_with_time:
            hour = alert['start_hour']
            severity = alert.get('severity', 'unknown').lower()
            if hour not in severity_hours:
                severity_hours[hour] = Counter()
            severity_hours[hour][severity] += 1
        
        severities = ['critical', 'warning', 'error', 'info']
        colors = ['#DC143C', '#FFA500', '#FF0000', '#32CD32']
        
        for i, severity in enumerate(severities):
            severity_counts = [severity_hours.get(hour, Counter()).get(severity, 0) for hour in range(24)]
            ax2.plot(range(24), severity_counts, label=severity.title(), 
                    color=colors[i], linewidth=2, marker='o', markersize=4)
        
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Alerts')
        ax2.set_title('🚨 Alert Severity Patterns', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def create_service_network_visualization(self):
        """Create service network visualization"""
        print("Generating service network visualization...")
        
        if not hasattr(self, 'service_graph') or len(self.service_graph) == 0:
            print("No service graph data available for visualization")
            return None
        
        # Create simplified network for visualization (top services only)
        centrality = nx.degree_centrality(self.service_graph)
        top_services = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if not top_services:
            print("No services with centrality data available")
            return None
        
        # Create subgraph with top services
        top_service_names = [service for service, _ in top_services]
        connected_services = set(top_service_names)
        
        # Add connected neighbors
        for service in top_service_names[:10]:  # Limit to prevent too many nodes
            neighbors = list(self.service_graph.neighbors(service))[:5]  # Top 5 neighbors
            connected_services.update(neighbors)
        
        subgraph = self.service_graph.subgraph(list(connected_services))
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Calculate layout
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Draw network
        node_sizes = [centrality[node] * 2000 for node in subgraph.nodes()]
        node_colors = [centrality[node] for node in subgraph.nodes()]
        
        nx.draw(subgraph, pos, ax=ax, 
                node_size=node_sizes, 
                node_color=node_colors,
                cmap=plt.cm.Reds,
                alpha=0.8,
                edge_color='gray',
                width=0.5,
                with_labels=False)
        
        # Add labels for top services only
        top_labels = {service: service.split('-')[-1][:15] for service, _ in top_services[:10]}
        nx.draw_networkx_labels(subgraph, pos, top_labels, font_size=8, ax=ax)
        
        ax.set_title('🌐 Service Dependency Network\n(Top Services by Centrality)', 
                    fontsize=14, fontweight='bold')
        
        # Create legend
        import matplotlib.patches as mpatches
        legend_elements = [mpatches.Patch(color=plt.cm.Reds(0.8), label='High Centrality'),
                          mpatches.Patch(color=plt.cm.Reds(0.4), label='Medium Centrality'),
                          mpatches.Patch(color=plt.cm.Reds(0.2), label='Low Centrality')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('service_network_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def create_comprehensive_dashboard(self, save_path='aiops_dashboard.png'):
        """Create comprehensive visualization dashboard"""
        print("🎨 Creating comprehensive AIOps dashboard...")
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        
        # Define grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('🚀 AIOps Alert Processing Pipeline - Comprehensive Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Pipeline Flow Overview (top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        original_count = len(self.firing_alerts)
        enriched_count = len([a for a in self.enriched_alerts if a.get('has_graph_context')])
        deduplicated_count = len(self.deduplicated_alerts)
        consolidated_count = len(self.consolidated_alerts)
        
        stages = ['Original\nAlerts', 'Enriched\nAlerts', 'Deduplicated\nAlerts', 'Consolidated\nGroups']
        counts = [original_count, enriched_count, deduplicated_count, consolidated_count]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(stages, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('📊 Pipeline Processing Overview', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add flow arrows
        for i in range(len(stages)-1):
            ax1.annotate('→', xy=(i+0.5, max(counts)/2), fontsize=20, ha='center', va='center')
        
        # 2. Deduplication Effectiveness (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        dedup_rate = ((original_count - deduplicated_count) / original_count) * 100
        enrichment_rate = (enriched_count / original_count) * 100
        
        metrics = ['Deduplication\nRate', 'Enrichment\nRate']
        values = [dedup_rate, enrichment_rate]
        colors_metrics = ['#FF9999', '#99CCFF']
        
        bars2 = ax2.bar(metrics, values, color=colors_metrics, alpha=0.8)
        ax2.set_title('⚡ Processing Efficiency', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Percentage (%)')
        
        for bar, value in zip(bars2, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. RCA Distribution (top-right bottom)
        ax3 = fig.add_subplot(gs[0, 3])
        rca_categories = Counter(group['final_rca_category'] for group in self.classified_alerts)
        
        # Take top 5 categories
        top_categories = dict(rca_categories.most_common(5))
        labels = [cat.replace('_', ' ').title()[:12] for cat in top_categories.keys()]
        sizes = list(top_categories.values())
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.0f%%',
                                          startangle=90, colors=plt.cm.Set3.colors)
        ax3.set_title('🎯 RCA Categories', fontsize=12, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # 4. Model Performance (second row, spans 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        if hasattr(self, 'model_performance') and self.model_performance:
            models = list(self.model_performance.keys())
            cv_scores = [self.model_performance[model]['cv_score'] * 100 for model in models]
            
            bars4 = ax4.barh(models, cv_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            ax4.set_title('🏆 Model Performance Comparison', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Cross-Validation Accuracy (%)')
            
            # Highlight best model
            if hasattr(self, 'best_model') and self.best_model in models:
                best_idx = models.index(self.best_model)
                bars4[best_idx].set_color('#FFD700')
                bars4[best_idx].set_edgecolor('red')
                bars4[best_idx].set_linewidth(3)
            
            for bar, score in zip(bars4, cv_scores):
                width = bar.get_width()
                ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                        f'{score:.1f}%', ha='left', va='center', fontweight='bold')
        
        # 5. Feature Importance (second row, spans 2 columns)
        ax5 = fig.add_subplot(gs[1, 2:])
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            y_pos = range(len(top_features))
            
            bars5 = ax5.barh(y_pos, top_features['avg_importance'], 
                            color=plt.cm.plasma(np.linspace(0, 1, len(top_features))))
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels([f.replace('_', ' ').title()[:15] for f in top_features['feature']], 
                              fontsize=10)
            ax5.set_xlabel('Importance Score')
            ax5.set_title('🧠 Top Feature Importance', fontsize=14, fontweight='bold')
            
            for bar, importance in zip(bars5, top_features['avg_importance']):
                width = bar.get_width()
                ax5.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                        f'{importance:.3f}', ha='left', va='center', fontweight='bold')
        
        # 6. Alert Timeline (third row, spans 4 columns)
        ax6 = fig.add_subplot(gs[2, :])
        
        # Get temporal data
        alerts_with_time = [a for a in self.enriched_alerts if a.get('start_datetime')]
        if alerts_with_time:
            hours = [a['start_hour'] for a in alerts_with_time]
            alerts_per_hour = Counter(hours)
            
            hours_data = sorted(alerts_per_hour.items())
            hours_x, hours_y = zip(*hours_data) if hours_data else ([], [])
            
            ax6.plot(hours_x, hours_y, marker='o', linewidth=3, markersize=6, color='#FF6B6B')
            ax6.fill_between(hours_x, hours_y, alpha=0.3, color='#FF6B6B')
            ax6.set_xlabel('Hour of Day')
            ax6.set_ylabel('Alert Count')
            ax6.set_title('⏰ Alert Volume by Hour', fontsize=14, fontweight='bold')
            ax6.set_xticks(range(0, 24, 2))
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No Temporal Data Available', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=14)
            ax6.set_title('⏰ Alert Timeline', fontsize=14, fontweight='bold')
        
        # 7. High Priority Incidents (bottom row, spans 2 columns)
        ax7 = fig.add_subplot(gs[3, :2])
        
        high_priority = [g for g in self.classified_alerts 
                        if g['total_alerts'] > 2 or 'critical' in g['final_rca_category']]
        
        if high_priority:
            top_incidents = sorted(high_priority, key=lambda x: x['total_alerts'], reverse=True)[:5]
            
            incident_names = [incident['primary_alert'].get('service_name', 'Unknown')[:20] 
                            for incident in top_incidents]
            alert_counts = [incident['total_alerts'] for incident in top_incidents]
            rca_cats = [incident['final_rca_category'].replace('_', ' ').title()[:15] 
                       for incident in top_incidents]
            
            x_pos = range(len(incident_names))
            bars7 = ax7.bar(x_pos, alert_counts, color=plt.cm.Reds(np.linspace(0.3, 1, len(incident_names))))
            
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(incident_names, rotation=45, ha='right', fontsize=10)
            ax7.set_ylabel('Alert Count')
            ax7.set_title('🚨 Top High-Priority Incidents', fontsize=14, fontweight='bold')
            
            # Add RCA category labels
            for i, (bar, count, rca) in enumerate(zip(bars7, alert_counts, rca_cats)):
                ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                        f'{count} alerts\n{rca}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        else:
            ax7.text(0.5, 0.5, 'No High-Priority Incidents Found', ha='center', va='center',
                    transform=ax7.transAxes, fontsize=12)
            ax7.set_title('🚨 High-Priority Incidents', fontsize=14, fontweight='bold')
        
        # 8. Summary Statistics (bottom row, spans 2 columns)
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        
        # Calculate summary stats
        fuzzy_matches = len([a for a in self.enriched_alerts if a.get('match_type') == 'fuzzy'])
        exact_matches = len([a for a in self.enriched_alerts if a.get('match_type') == 'exact'])
        
        stats_text = f"""
📊 PIPELINE SUMMARY STATISTICS
═══════════════════════════════════

🔢 Processing Metrics:
• Original Alerts: {original_count:,}
• Graph Enriched: {enriched_count:,} ({enriched_count/original_count*100:.1f}%)
• After Deduplication: {deduplicated_count:,}
• Consolidated Groups: {consolidated_count:,}

🎯 Enhancement Results:
• Exact Matches: {exact_matches}
• Fuzzy Matches: {fuzzy_matches}
• Deduplication Rate: {((original_count - deduplicated_count) / original_count * 100):.1f}%
• Enrichment Rate: {(enriched_count/original_count*100):.1f}%

🧠 ML Performance:
• Best Model: {getattr(self, 'best_model', 'N/A')}
• Features Selected: {len(getattr(self, 'best_features', [] ))}
• Models Trained: {len(getattr(self, 'models', {} ))}

🚨 Critical Issues:
• High-Priority Groups: {len([g for g in self.classified_alerts if g['total_alerts'] > 2])}
• Cascading Failures: {len([g for g in self.classified_alerts if 'cascading' in g['final_rca_category']])}
• Avg Confidence: {np.mean([g.get('ml_confidence', 0) for g in self.classified_alerts]):.1f}%
        """
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive dashboard saved as: {save_path}")
        return fig
    
    def save_results(self, output_dir='/data/redppo/'):
        """Save all processing results"""
        print(f"\nSAVING COMPREHENSIVE RESULTS to {output_dir}")
        
        # Save deduplicated alerts
        if self.deduplicated_alerts:
            pd.DataFrame(self.deduplicated_alerts).to_csv('deduplicated_alerts.csv', index=False)
            print(f"  Deduplicated alerts: deduplicated_alerts.csv")
        
        # Save consolidated groups with RCA classification
        if self.classified_alerts:
            consolidated_data = []
            for group in self.classified_alerts:
                primary = group['primary_alert']
                consolidated_data.append({
                    'group_id': group['group_id'],
                    'rca_category': group['final_rca_category'],
                    'ml_confidence': group.get('ml_confidence', 0),
                    'alert_count': group['total_alerts'],
                    'relationship_type': group['relationship_type'],
                    'service_name': primary.get('service_name', ''),
                    'resource_type': primary.get('anomaly_resource_type', ''),
                    'severity': primary.get('severity', ''),
                    'description': primary.get('description', '')[:200],
                    'has_graph_context': primary.get('has_graph_context', False),
                    'service_centrality': primary.get('service_centrality', 0),
                    'calls_services': ', '.join(primary.get('calls_services', [])),
                    'called_by_services': ', '.join(primary.get('called_by_services', []))
                })
            
            pd.DataFrame(consolidated_data).to_csv('rca_classified_alerts.csv', index=False)
            print(f"  RCA classified alerts: rca_classified_alerts.csv")
        
        print(f"\nComplete processing pipeline finished!")

def main():
    """Main execution function"""
    print("="*70)
    print("COMPREHENSIVE AIOPS ALERT PROCESSING & RCA CLASSIFICATION")
    print("="*70)
    
    processor = ComprehensiveAlertProcessor(
        alerts_csv_path='C:/Users/jurat.shayidin/aiops/alert_data.csv',
        graph_json_path='C:/Users/jurat.shayidin/aiops/graph_data.json'
    )
    
    # Step 1: Load and enrich alerts
    processor.load_firing_alerts()
    processor.load_graph_data()
    processor.enrich_alerts_with_graph_context()
    
    # Step 2: Temporal deduplication
    processor.temporal_deduplication()
    
    # Step 3: Consolidate by relationships
    processor.consolidate_alerts_by_relationships()
    
    # Step 4: Advanced RCA classification
    processor.train_advanced_classifier()
    processor.classify_alerts_for_rca()
    
    # Step 5: Generate visualizations
    print("\n🎨 Generating comprehensive visualizations...")
    
    try:
        # Create individual visualization charts
        processor.create_pipeline_performance_chart()
        
        if hasattr(processor, 'feature_importance') and processor.feature_importance is not None:
            processor.create_feature_importance_chart()
        
        processor.create_temporal_analysis_chart()
        processor.create_service_network_visualization()
        
        # Create comprehensive dashboard
        processor.create_comprehensive_dashboard('aiops_comprehensive_dashboard.png')
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing with results generation...")
    
    # Step 6: Generate report and save results
    results = processor.generate_comprehensive_report()
    processor.save_results()
    
    return processor, results

if __name__ == "__main__":
    processor, results = main()