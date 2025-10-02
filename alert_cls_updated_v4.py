import pandas as pd
import json
import ast
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx
import difflib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
        
        # RCA classifier
        self.rca_classifier = None
        self.rca_scaler = None
        self.rca_label_encoder = None
        
    def load_firing_alerts(self):
        """Load and filter firing alerts with temporal parsing"""
        print("Loading firing alerts...")
        
        df = pd.read_csv(self.alerts_csv_path)
        firing_df = df[df['status'].str.strip().str.lower() == 'firing']
        self.firing_alerts = firing_df.to_dict('records')
        
        print(f"Loaded {len(self.firing_alerts)} firing alerts")
        
        # Parse payload and temporal info
        for alert in self.firing_alerts:
            self._parse_alert_payload(alert)
            self._parse_temporal_info(alert)
        
        return self.firing_alerts
    
    def _parse_alert_payload(self, alert):
        """Parse payload_consolidated to extract metadata"""
        payload_str = alert.get('payload_consolidated', '')
        if not payload_str:
            return
        
        try:
            payload = ast.literal_eval(payload_str)
            
            # Extract labels
            labels = payload.get('labels', {})
            alert['cluster'] = labels.get('cluster', '')
            alert['namespace'] = labels.get('namespace', '')
            alert['pod'] = labels.get('pod', '')
            alert['node'] = labels.get('node', '')
            alert['anomaly_resource_type'] = labels.get('anomaly_resource_type', '')
            alert['workload_type'] = labels.get('workload_type', '')
            alert['platform'] = labels.get('platform', '')
            alert['anomaly_entity_type'] = labels.get('anomaly_entity_type', '')
            
            # Extract description
            annotations = payload.get('annotations', {})
            alert['description'] = annotations.get('description', '')
            
        except Exception as e:
            print(f"Warning: Could not parse payload for alert {alert.get('_id', 'unknown')}")
    
    def _parse_temporal_info(self, alert):
        """Parse temporal information for deduplication"""
        try:
            starts_at = alert.get('starts_at', '')
            if starts_at:
                alert['start_datetime'] = pd.to_datetime(starts_at)
                alert['start_timestamp'] = alert['start_datetime'].timestamp()
                alert['start_hour'] = alert['start_datetime'].hour
                alert['start_minute'] = alert['start_datetime'].minute
            
            # Check if ongoing (firing alerts typically have placeholder end times)
            ends_at = alert.get('ends_at', '')
            alert['is_ongoing'] = ends_at in ['1-01-01 00:00:00.000', '0001-01-01T00:00:00Z', '']
            
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
            source_props = rel.get('source_properties', {})
            target_props = rel.get('target_properties', {})
            
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
    
    def enrich_alerts_with_graph_context(self):
        """Enrich alerts with graph relationship context"""
        print("Enriching alerts with graph context...")
        
        enriched_count = 0
        
        for alert in self.firing_alerts:
            service_name = alert.get('service_name', '').strip()
            
            # Initialize graph context
            alert['has_graph_context'] = False
            alert['calls_services'] = []
            alert['called_by_services'] = []
            alert['belongs_to_services'] = []
            alert['owns_services'] = []
            alert['service_centrality'] = 0.0
            
            if service_name in self.service_to_graph:
                graph_info = self.service_to_graph[service_name]
                alert['has_graph_context'] = True
                alert['graph_type'] = graph_info['type']
                alert['graph_environment'] = graph_info['environment']
                alert['graph_namespace'] = graph_info['namespace']
                alert['graph_cluster'] = graph_info['cluster']
                
                # Calculate service centrality
                try:
                    centrality = nx.degree_centrality(self.service_graph)[service_name]
                    alert['service_centrality'] = centrality
                except:
                    alert['service_centrality'] = 0.0
                
                # Find relationships
                for successor in self.service_graph.successors(service_name):
                    edge_data = self.service_graph.get_edge_data(service_name, successor)
                    if edge_data:
                        rel_type = edge_data.get('relationship_type')
                        if rel_type == 'CALLS':
                            alert['calls_services'].append(successor)
                        elif rel_type == 'BELONGS_TO':
                            alert['belongs_to_services'].append(successor)
                
                for predecessor in self.service_graph.predecessors(service_name):
                    edge_data = self.service_graph.get_edge_data(predecessor, service_name)
                    if edge_data:
                        rel_type = edge_data.get('relationship_type')
                        if rel_type == 'CALLS':
                            alert['called_by_services'].append(predecessor)
                        elif rel_type == 'BELONGS_TO':
                            alert['owns_services'].append(predecessor)
                
                enriched_count += 1
        
        self.enriched_alerts = self.firing_alerts
        print(f"Enriched {enriched_count}/{len(self.firing_alerts)} alerts with graph context")
        return self.enriched_alerts
    
    def temporal_deduplication(self, time_window_minutes=10):
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
        
        print(f"Temporal deduplication: {original_count} → {final_count} alerts ({reduction} duplicates removed)")
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
            if similarity < 0.8:  # Not similar enough
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
        service_name = primary_alert.get('service_name', '').strip()
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
                alert.get('service_name', '').strip() in related_services):
                
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
        
        # Rule-based RCA prediction
        if len(alerts) > 3 and 'network' in dominant_resource:
            return 'network_infrastructure_issue'
        elif len(alerts) > 1 and has_dependencies:
            return 'cascading_service_failure'
        elif 'cpu' in dominant_resource and avg_centrality > 0.1:
            return 'resource_exhaustion_critical_service'
        elif 'memory' in dominant_resource:
            return 'memory_leak_or_pressure'
        elif len(alerts) == 1 and avg_centrality < 0.05:
            return 'isolated_service_issue'
        elif 'disk' in dominant_resource:
            return 'storage_issue'
        else:
            return 'application_performance_issue'
    
    def prepare_classification_features(self):
        """Prepare features for ML-based RCA classification"""
        print("Preparing features for RCA classification...")
        
        features = []
        labels = []
        
        for group in self.consolidated_alerts:
            primary_alert = group['primary_alert']
            
            # Feature vector
            feature_vector = {
                # Alert characteristics
                'alert_count': group['total_alerts'],
                'has_graph_context': 1 if primary_alert.get('has_graph_context') else 0,
                'service_centrality': primary_alert.get('service_centrality', 0),
                'is_duplicate': 1 if primary_alert.get('is_duplicate', False) else 0,
                'duplicate_count': primary_alert.get('duplicate_count', 1),
                
                # Resource type features
                'resource_cpu': 1 if 'cpu' in primary_alert.get('anomaly_resource_type', '').lower() else 0,
                'resource_memory': 1 if 'memory' in primary_alert.get('anomaly_resource_type', '').lower() else 0,
                'resource_network': 1 if 'network' in primary_alert.get('anomaly_resource_type', '').lower() else 0,
                'resource_disk': 1 if 'disk' in primary_alert.get('anomaly_resource_type', '').lower() else 0,
                
                # Relationship features
                'calls_count': len(primary_alert.get('calls_services', [])),
                'called_by_count': len(primary_alert.get('called_by_services', [])),
                'belongs_to_count': len(primary_alert.get('belongs_to_services', [])),
                'owns_count': len(primary_alert.get('owns_services', [])),
                
                # Temporal features
                'hour_of_day': primary_alert.get('start_hour', 0),
                'duration_minutes': primary_alert.get('duration_minutes', 0),
                
                # Infrastructure features
                'entity_type_pod': 1 if primary_alert.get('anomaly_entity_type', '').lower() == 'pod' else 0,
                'entity_type_service': 1 if primary_alert.get('anomaly_entity_type', '').lower() == 'service' else 0,
                'entity_type_node': 1 if primary_alert.get('anomaly_entity_type', '').lower() == 'node' else 0,
                
                # Severity
                'severity_warning': 1 if primary_alert.get('severity', '').lower() == 'warning' else 0,
                'severity_critical': 1 if primary_alert.get('severity', '').lower() == 'critical' else 0,
            }
            
            features.append(feature_vector)
            labels.append(group['rca_category'])
        
        self.classification_features = pd.DataFrame(features)
        self.classification_labels = labels
        
        print(f"Prepared {len(features)} feature vectors for classification")
        return self.classification_features, self.classification_labels
    
    def train_rca_classifier(self):
        """Train RCA classifier"""
        print("Training RCA classifier...")
        
        if not hasattr(self, 'classification_features'):
            self.prepare_classification_features()
        
        X = self.classification_features.fillna(0)
        y = self.classification_labels
        
        # Encode labels
        self.rca_label_encoder = LabelEncoder()
        y_encoded = self.rca_label_encoder.fit_transform(y)
        
        # Scale features
        self.rca_scaler = StandardScaler()
        X_scaled = self.rca_scaler.fit_transform(X)
        
        # Train classifier
        self.rca_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.rca_classifier.fit(X_scaled, y_encoded)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rca_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features for RCA:")
        print(feature_importance.head(10))
        
        return self.rca_classifier
    
    def classify_alerts_for_rca(self):
        """Classify consolidated alerts for RCA"""
        print("Classifying alerts for RCA...")
        
        if not self.rca_classifier:
            self.train_rca_classifier()
        
        # Predict RCA categories
        X = self.classification_features.fillna(0)
        X_scaled = self.rca_scaler.transform(X)
        
        predictions = self.rca_classifier.predict(X_scaled)
        probabilities = self.rca_classifier.predict_proba(X_scaled)
        
        # Update consolidated alerts with predictions
        for i, group in enumerate(self.consolidated_alerts):
            predicted_category = self.rca_label_encoder.inverse_transform([predictions[i]])[0]
            confidence = np.max(probabilities[i])
            
            group['ml_rca_category'] = predicted_category
            group['ml_confidence'] = confidence
            group['final_rca_category'] = predicted_category  # Could combine with rule-based
        
        self.classified_alerts = self.consolidated_alerts
        
        print("RCA classification completed")
        return self.classified_alerts
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE AIOPS ALERT PROCESSING & RCA CLASSIFICATION")
        print("="*70)
        
        # Processing statistics
        original_count = len(self.firing_alerts)
        enriched_count = len([a for a in self.enriched_alerts if a.get('has_graph_context')])
        deduplicated_count = len(self.deduplicated_alerts)
        consolidated_count = len(self.consolidated_alerts)
        
        print(f"\n📊 PROCESSING PIPELINE RESULTS:")
        print(f"  Original firing alerts: {original_count}")
        print(f"  Graph-enriched alerts: {enriched_count}")
        print(f"  After deduplication: {deduplicated_count}")
        print(f"  Consolidated groups: {consolidated_count}")
        
        # Deduplication effectiveness
        dedup_reduction = ((original_count - deduplicated_count) / original_count * 100) if original_count > 0 else 0
        print(f"  Deduplication reduction: {dedup_reduction:.1f}%")
        
        # RCA classification results
        rca_categories = Counter(group['final_rca_category'] for group in self.classified_alerts)
        
        print(f"\n🔍 RCA CLASSIFICATION RESULTS:")
        for category, count in rca_categories.most_common():
            print(f"  {category}: {count} groups")
        
        # High-priority issues
        high_priority = [
            g for g in self.classified_alerts 
            if g['total_alerts'] > 2 or 'critical' in g['final_rca_category'] or 'cascading' in g['final_rca_category']
        ]
        
        print(f"\n🚨 HIGH-PRIORITY RCA GROUPS: {len(high_priority)}")
        for i, group in enumerate(sorted(high_priority, key=lambda x: x['total_alerts'], reverse=True)[:5], 1):
            primary = group['primary_alert']
            print(f"  {i}. {group['final_rca_category']} ({group['total_alerts']} alerts)")
            print(f"     Service: {primary.get('service_name', 'Unknown')}")
            print(f"     Resource: {primary.get('anomaly_resource_type', 'Unknown')}")
            print(f"     Confidence: {group.get('ml_confidence', 0):.2f}")
        
        return {
            'original_alerts': original_count,
            'deduplicated_alerts': deduplicated_count,
            'consolidated_groups': consolidated_count,
            'rca_categories': dict(rca_categories),
            'high_priority_groups': len(high_priority)
        }
    
    def save_results(self, output_dir='/data/redppo/'):
        """Save all processing results"""
        print(f"\n💾 SAVING COMPREHENSIVE RESULTS to {output_dir}")
        
        # Save deduplicated alerts
        if self.deduplicated_alerts:
            pd.DataFrame(self.deduplicated_alerts).to_csv(f'{output_dir}deduplicated_alerts.csv', index=False)
            print(f"  ✅ Deduplicated alerts: deduplicated_alerts.csv")
        
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
            
            pd.DataFrame(consolidated_data).to_csv(f'{output_dir}rca_classified_alerts.csv', index=False)
            print(f"  ✅ RCA classified alerts: rca_classified_alerts.csv")
        
        print(f"\n🎉 Complete processing pipeline finished!")

def main():
    """Main execution function"""
    print("="*70)
    print("COMPREHENSIVE AIOPS ALERT PROCESSING & RCA CLASSIFICATION")
    print("="*70)
    
    processor = ComprehensiveAlertProcessor(
        alerts_csv_path='/data/redppo/alerts_pivoted_deduplicated.csv',
        graph_json_path='/data/redppo/graph_data.json'
    )
    
    # Step 1: Load and enrich alerts
    processor.load_firing_alerts()
    processor.load_graph_data()
    processor.enrich_alerts_with_graph_context()
    
    # Step 2: Temporal deduplication
    processor.temporal_deduplication()
    
    # Step 3: Consolidate by relationships
    processor.consolidate_alerts_by_relationships()
    
    # Step 4: RCA classification
    processor.prepare_classification_features()
    processor.train_rca_classifier()
    processor.classify_alerts_for_rca()
    
    # Step 5: Generate report and save results
    results = processor.generate_comprehensive_report()
    processor.save_results()
    
    return processor, results

if __name__ == "__main__":
    processor, results = main()