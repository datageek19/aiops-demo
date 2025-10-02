# eenhanced alert data classification prototype solution
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import ast
import re
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import networkx as nx
import difflib

class GraphEnrichedAlertsProcessor:
    """
    Advanced AIOps Alert Processing with Graph Context and ML Preparation
    
    This class provides comprehensive alert processing including:
    1. Alert data loading and pivoting/deduplication
    2. Graph data integration and service mapping
    3. Relationship-based alert expansion and context enrichment
    4. Graph-aware alert grouping for RCA
    5. Feature engineering for ML classification
    6. Root cause analysis preparation
    """
    
    def __init__(self, alerts_csv_path, graph_json_path):
        self.alerts_csv_path = alerts_csv_path
        self.graph_json_path = graph_json_path
        
        # Data containers
        self.raw_alerts = []
        self.pivoted_alerts = []
        self.firing_alerts = []
        self.graph_data = []
        self.service_graph = nx.DiGraph()
        
        # Processing results
        self.enriched_alerts = []
        self.alert_groups = []
        self.rca_groups = []
        self.ml_features = None
        
        # Mappings
        self.service_to_node = {}
        self.node_to_service = {}
        
    def load_and_pivot_alerts(self):
        """
        Load raw alert data and pivot/deduplicate by alert ID
        
        The raw alert data has multiple rows per alert (status, labels, annotations, etc.)
        This method consolidates them into single alert records.
        """
        print("Loading and pivoting alert data...")
        
        # Load raw alerts
        df = pd.read_csv(self.alerts_csv_path)
        print(f"Loaded {len(df)} raw alert rows")
        
        # Group by alert ID and consolidate
        pivoted_alerts = []
        
        for alert_id, group in df.groupby('_id'):
            # Start with base alert info from first row
            base_alert = group.iloc[0].to_dict()
            
            # Consolidate payload data from different row types
            consolidated_payload = {}
            
            for _, row in group.iterrows():
                row_type = row.get('payload', '').strip()
                
                if row_type == 'firing' or row_type == 'resolved':
                    consolidated_payload['status_payload'] = row_type
                elif row.get('payload', '').startswith('{'):
                    # Parse structured payload
                    try:
                        payload_data = ast.literal_eval(row['payload'])
                        if isinstance(payload_data, dict):
                            consolidated_payload.update(payload_data)
                    except:
                        pass
            
            # Create consolidated alert
            alert = {
                '_id': alert_id,
                'batch_id': base_alert.get('batch_id', ''),
                'starts_at': base_alert.get('starts_at', ''),
                'ends_at': base_alert.get('ends_at', ''),
                'status': base_alert.get('status', ''),
                'service_name': base_alert.get('service_name', ''),
                'alert_category': base_alert.get('alert_category', ''),
                'alert_subcategory': base_alert.get('alert_subcategory', ''),
                'severity': base_alert.get('severity', ''),
                'alert_name': base_alert.get('alert_name', ''),
                'recorded_at': base_alert.get('recorded_at', ''),
                'original_row_count': len(group)
            }
            
            # Add consolidated payload fields
            alert.update(consolidated_payload)
            pivoted_alerts.append(alert)
        
        self.pivoted_alerts = pivoted_alerts
        print(f"Pivoted to {len(pivoted_alerts)} unique alerts")
        
        # Filter firing alerts
        self.firing_alerts = [
            alert for alert in pivoted_alerts 
            if alert.get('status', '').strip().lower() == 'firing'
        ]
        print(f"Found {len(self.firing_alerts)} firing alerts")
        
        return self.firing_alerts
    
    # timestamp parsing helper
    def _parse_timestamp_info(self, alert):
        """Parse and normalize timestamp information"""
        try:
            starts_at = alert.get('starts_at', '')
            ends_at = alert.get('ends_at', '')
            
            # Parse start time
            if starts_at:
                alert['start_datetime'] = pd.to_datetime(starts_at)
                alert['start_hour'] = alert['start_datetime'].hour
                alert['start_minute'] = alert['start_datetime'].minute
            
            # Check if alert is ongoing (firing alerts have placeholder end time)
            alert['is_ongoing'] = ends_at == '1-01-01 00:00:00.000'
            
        except Exception as e:
            print(f"Warning: Could not parse timestamps for alert {alert.get('_id', 'unknown')}: {e}")
            alert['start_datetime'] = None
            alert['is_ongoing'] = True
    
    # 
    def load_graph_data(self):
        """Load service graph and create service mappings"""
        print("Loading service graph data...")
        
        with open(self.graph_json_path, 'r') as f:
            graph_data = json.load(f)
        
        print(f"Loaded {len(graph_data)} graph relationships")
        
        # Build service mappings and graph
        self.service_graph = nx.DiGraph()
        
        for relationship in graph_data:
            source_name = relationship.get('source_name', '')
            target_name = relationship.get('target_name', '')
            rel_type = relationship.get('relationship_type', '')
            
            if source_name:
                # Store service info for mapping
                self.graph_services[source_name] = {
                    'name': source_name,
                    'properties': relationship.get('source_properties', {}),
                    'type': relationship.get('source_label', ''),
                    'environment': relationship.get('source_properties', {}).get('environment', ''),
                    'namespace': relationship.get('source_properties', {}).get('namespace', '')
                }
                
                # Add to graph
                self.service_graph.add_node(source_name, **relationship.get('source_properties', {}))
            
            if target_name:
                self.graph_services[target_name] = {
                    'name': target_name,
                    'properties': relationship.get('target_properties', {}),
                    'type': relationship.get('target_label', ''),
                    'environment': relationship.get('target_properties', {}).get('environment', ''),
                    'namespace': relationship.get('target_properties', {}).get('namespace', '')
                }
                
                self.service_graph.add_node(target_name, **relationship.get('target_properties', {}))
            
            # Add relationship
            if source_name and target_name and rel_type:
                self.service_graph.add_edge(source_name, target_name, relationship_type=rel_type)
        
        print(f"Built service graph with {len(self.graph_services)} services")
        return self.graph_services
    
    def enrich_alerts_with_graph_context(self):
        """Enrich alerts with graph context and relationships"""
        print("Enriching alerts with graph context...")
        
        enriched_alerts = []
        
        for alert in self.firing_alerts:
            enriched_alert = alert.copy()
            service_name = alert.get('service_name', '').strip()
            
            if service_name in self.service_graph:
                # Add direct graph properties
                node_props = self.service_graph.nodes[service_name]
                enriched_alert['graph_node_type'] = node_props.get('type', '')
                enriched_alert['graph_environment'] = node_props.get('environment', '')
                enriched_alert['graph_cluster'] = node_props.get('cluster', '')
                enriched_alert['graph_namespace'] = node_props.get('namespace', '')
                
                # Find upstream services (services that call this service)
                upstream_services = list(self.service_graph.predecessors(service_name))
                enriched_alert['upstream_services'] = ','.join(upstream_services[:5])  # Limit to 5
                enriched_alert['upstream_count'] = len(upstream_services)
                
                # Find downstream services (services this service calls)
                downstream_services = list(self.service_graph.successors(service_name))
                enriched_alert['downstream_services'] = ','.join(downstream_services[:5])
                enriched_alert['downstream_count'] = len(downstream_services)
                
                # Calculate service centrality (importance in graph)
                try:
                    centrality = nx.degree_centrality(self.service_graph)[service_name]
                    enriched_alert['service_centrality'] = centrality
                except:
                    enriched_alert['service_centrality'] = 0.0
                
                # Find related services within 2 hops
                try:
                    related_services = set()
                    for path_length in [1, 2]:
                        for target in nx.single_source_shortest_path_length(
                            self.service_graph, service_name, cutoff=path_length
                        ).keys():
                            if target != service_name:
                                related_services.add(target)
                    
                    enriched_alert['related_services'] = ','.join(list(related_services)[:10])
                    enriched_alert['related_services_count'] = len(related_services)
                except:
                    enriched_alert['related_services'] = ''
                    enriched_alert['related_services_count'] = 0
            else:
                # Service not in graph
                enriched_alert['graph_node_type'] = 'unknown'
                enriched_alert['graph_environment'] = ''
                enriched_alert['graph_cluster'] = ''
                enriched_alert['graph_namespace'] = ''
                enriched_alert['upstream_services'] = ''
                enriched_alert['upstream_count'] = 0
                enriched_alert['downstream_services'] = ''
                enriched_alert['downstream_count'] = 0
                enriched_alert['service_centrality'] = 0.0
                enriched_alert['related_services'] = ''
                enriched_alert['related_services_count'] = 0
            
            enriched_alerts.append(enriched_alert)
        
        self.enriched_alerts = enriched_alerts
        print(f"Enriched {len(enriched_alerts)} alerts with graph context")
        
        return enriched_alerts
    
    def find_related_alerts_by_graph(self, alert, max_hops=2):
        """Find alerts related to this alert through graph relationships"""
        service_name = alert.get('service_name', '').strip()
        related_alerts = []
        
        if service_name not in self.service_graph:
            return related_alerts
        
        # Get all services within max_hops
        related_services = set()
        try:
            for target in nx.single_source_shortest_path_length(
                self.service_graph, service_name, cutoff=max_hops
            ).keys():
                if target != service_name:
                    related_services.add(target)
        except:
            pass
        
        # Find alerts for related services
        for other_alert in self.enriched_alerts:
            other_service = other_alert.get('service_name', '').strip()
            if other_service in related_services:
                # Calculate relationship distance
                try:
                    distance = nx.shortest_path_length(
                        self.service_graph, service_name, other_service
                    )
                    other_alert['graph_distance'] = distance
                    related_alerts.append(other_alert)
                except:
                    pass
        
        return related_alerts
    
    def calculate_enhanced_similarity(self, alert1, alert2):
        """
        Enhanced similarity calculation including graph context
        
        Combines semantic similarity with graph relationship context
        """
        # Base semantic similarity (from original solution)
        semantic_score, breakdown = self.calculate_semantic_similarity(alert1, alert2)
        
        # Graph-based similarity
        service1 = alert1.get('service_name', '').strip()
        service2 = alert2.get('service_name', '').strip()
        
        graph_similarity = 0.0
        
        if service1 in self.service_graph and service2 in self.service_graph:
            # Direct connection bonus
            if self.service_graph.has_edge(service1, service2) or self.service_graph.has_edge(service2, service1):
                graph_similarity += 0.8
            
            # Shared neighbors bonus
            neighbors1 = set(self.service_graph.neighbors(service1))
            neighbors2 = set(self.service_graph.neighbors(service2))
            shared_neighbors = len(neighbors1.intersection(neighbors2))
            total_neighbors = len(neighbors1.union(neighbors2))
            
            if total_neighbors > 0:
                neighbor_similarity = shared_neighbors / total_neighbors
                graph_similarity += neighbor_similarity * 0.4
            
            # Environment/cluster similarity
            env1 = alert1.get('graph_environment', '')
            env2 = alert2.get('graph_environment', '')
            if env1 and env2 and env1 == env2:
                graph_similarity += 0.3
            
            cluster1 = alert1.get('graph_cluster', '')
            cluster2 = alert2.get('graph_cluster', '')
            if cluster1 and cluster2 and cluster1 == cluster2:
                graph_similarity += 0.2
        
        # Combine semantic and graph similarities
        # Weight: 70% semantic, 30% graph
        combined_score = (semantic_score * 0.7) + (graph_similarity * 0.3)
        
        breakdown['graph_similarity'] = graph_similarity
        breakdown['combined_score'] = combined_score
        
        return combined_score, breakdown
    
    def calculate_semantic_similarity(self, alert1, alert2):
        """Original semantic similarity calculation (from existing solution)"""
        similarities = []
        
        # Service name similarity (highest weight)
        service1 = alert1.get('service_name', '').strip().lower()
        service2 = alert2.get('service_name', '').strip().lower()
        if service1 and service2:
            service_sim = difflib.SequenceMatcher(None, service1, service2).ratio()
            similarities.append(('service_name', service_sim, 4.0))
        
        # Alert name similarity
        alert_name1 = alert1.get('alert_name', '').strip().lower()
        alert_name2 = alert2.get('alert_name', '').strip().lower()
        if alert_name1 and alert_name2:
            alert_sim = difflib.SequenceMatcher(None, alert_name1, alert_name2).ratio()
            similarities.append(('alert_name', alert_sim, 3.0))
        
        # Category similarity
        category1 = alert1.get('alert_category', '').strip().lower()
        category2 = alert2.get('alert_category', '').strip().lower()
        if category1 and category2:
            category_sim = 1.0 if category1 == category2 else 0.0
            similarities.append(('alert_category', category_sim, 2.0))
        
        # Severity similarity
        severity1 = alert1.get('severity', '').strip().lower()
        severity2 = alert2.get('severity', '').strip().lower()
        if severity1 and severity2:
            severity_sim = 1.0 if severity1 == severity2 else 0.0
            similarities.append(('severity', severity_sim, 2.0))
        
        # Resource type similarity
        resource1 = alert1.get('anomaly_resource_type', '').strip().lower()
        resource2 = alert2.get('anomaly_resource_type', '').strip().lower()
        if resource1 and resource2:
            resource_sim = 1.0 if resource1 == resource2 else 0.0
            similarities.append(('resource_type', resource_sim, 2.0))
        
        if not similarities:
            return 0.0, {}
        
        total_weighted_score = sum(score * weight for _, score, weight in similarities)
        total_weight = sum(weight for _, _, weight in similarities)
        overall_similarity = total_weighted_score / total_weight
        
        breakdown = {factor: score for factor, score, _ in similarities}
        return overall_similarity, breakdown
    
    # 
    def detect_temporal_duplicates(self, time_window_minutes=5):
        """
        Detect duplicate alerts based on temporal proximity and similarity
        
        Alerts are considered duplicates if they:
        1. Have high similarity score (>0.8)
        2. Occur within the time window
        3. Have same service or very similar services
        """
        print(f"Detecting temporal duplicates (window: {time_window_minutes} minutes)...")
        
        # Sort alerts by timestamp
        sorted_alerts = sorted(
            [a for a in self.firing_alerts if a.get('start_datetime')], 
            key=lambda x: x['start_datetime']
        )
        
        duplicates = []
        processed = set()
        
        for i, alert1 in enumerate(sorted_alerts):
            if i in processed:
                continue
                
            duplicate_group = [alert1]
            
            # Look for duplicates in the time window
            for j, alert2 in enumerate(sorted_alerts[i+1:], i+1):
                if j in processed:
                    continue
                
                # Check time window
                time_diff = (alert2['start_datetime'] - alert1['start_datetime']).total_seconds() / 60
                if time_diff > time_window_minutes:
                    break  # No more alerts in time window
                
                # Check similarity
                similarity, _ = self.calculate_alert_similarity(alert1, alert2)
                
                if similarity > 0.8:  # High similarity threshold for duplicates
                    duplicate_group.append(alert2)
                    processed.add(j)
            
            if len(duplicate_group) > 1:
                duplicates.append({
                    'representative': alert1,
                    'duplicates': duplicate_group[1:],
                    'total_count': len(duplicate_group),
                    'time_span_minutes': (duplicate_group[-1]['start_datetime'] - alert1['start_datetime']).total_seconds() / 60
                })
            
            processed.add(i)
        
        print(f"Found {len(duplicates)} duplicate groups affecting {sum(d['total_count'] for d in duplicates)} alerts")
        return duplicates
    
    def group_alerts_with_graph_context(self, similarity_threshold=0.65):
        """
        Enhanced alert grouping using both semantic similarity and graph relationships
        
        Lower threshold because graph context provides additional grouping signal
        """
        print(f"Grouping alerts with graph context (threshold: {similarity_threshold})...")
        
        if not self.enriched_alerts:
            return []
        
        alert_groups = []
        processed_indices = set()
        
        for i, alert1 in enumerate(self.enriched_alerts):
            if i in processed_indices:
                continue
            
            # Start new group
            current_group = {
                'group_id': len(alert_groups) + 1,
                'alerts': [alert1],
                'indices': [i],
                'representative': alert1,
                'similarities': [],
                'graph_relationships': []
            }
            
            # Find similar alerts (semantic + graph)
            for j, alert2 in enumerate(self.enriched_alerts[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                similarity_score, breakdown = self.calculate_enhanced_similarity(alert1, alert2)
                
                if similarity_score >= similarity_threshold:
                    current_group['alerts'].append(alert2)
                    current_group['indices'].append(j)
                    current_group['similarities'].append({
                        'alert_index': j,
                        'similarity_score': similarity_score,
                        'breakdown': breakdown
                    })
                    processed_indices.add(j)
            
            # Find graph-related alerts (even if not semantically similar)
            related_alerts = self.find_related_alerts_by_graph(alert1, max_hops=2)
            for related_alert in related_alerts:
                # Find index of related alert
                for k, enriched_alert in enumerate(self.enriched_alerts):
                    if (enriched_alert.get('_id') == related_alert.get('_id') and 
                        k not in processed_indices and k not in current_group['indices']):
                        
                        current_group['alerts'].append(related_alert)
                        current_group['indices'].append(k)
                        current_group['graph_relationships'].append({
                            'alert_index': k,
                            'graph_distance': related_alert.get('graph_distance', 999),
                            'relationship_type': 'graph_related'
                        })
                        processed_indices.add(k)
                        break
            
            alert_groups.append(current_group)
            processed_indices.add(i)
        
        self.alert_groups = alert_groups
        
        # Show grouping results
        multi_alert_groups = [g for g in alert_groups if len(g['alerts']) > 1]
        single_alert_groups = [g for g in alert_groups if len(g['alerts']) == 1]
        
        print(f"Created {len(alert_groups)} alert groups:")
        print(f"  Groups with multiple alerts: {len(multi_alert_groups)}")
        print(f"  Single alert groups: {len(single_alert_groups)}")
        
        return alert_groups
    
    def create_rca_groups(self):
        """
        Create Root Cause Analysis groups based on graph relationships and patterns
        
        Groups alerts that likely share the same root cause based on:
        1. Service dependency chains
        2. Infrastructure components (clusters, nodes)
        3. Resource type patterns
        4. Temporal correlation
        """
        print("Creating RCA groups...")
        
        rca_groups = []
        
        # Group 1: Service Dependency Chains
        dependency_groups = self._group_by_service_dependencies()
        rca_groups.extend(dependency_groups)
        
        # Group 2: Infrastructure Components
        infra_groups = self._group_by_infrastructure()
        rca_groups.extend(infra_groups)
        
        # Group 3: Resource Type Patterns
        resource_groups = self._group_by_resource_patterns()
        rca_groups.extend(resource_groups)
        
        self.rca_groups = rca_groups
        print(f"Created {len(rca_groups)} RCA groups")
        
        return rca_groups
    
    def _group_by_service_dependencies(self):
        """Group alerts by service dependency chains"""
        dependency_groups = []
        
        # Find service chains with multiple alerts
        service_chains = defaultdict(list)
        
        for group in self.alert_groups:
            if len(group['alerts']) > 1:
                # Get all services in this group
                services = [alert.get('service_name', '') for alert in group['alerts']]
                services = [s for s in services if s.strip()]
                
                if len(services) > 1:
                    # Check if services form a dependency chain
                    chain_key = self._find_service_chain(services)
                    if chain_key:
                        service_chains[chain_key].append(group)
        
        for chain_key, groups in service_chains.items():
            if len(groups) > 0:
                dependency_groups.append({
                    'rca_type': 'service_dependency',
                    'root_cause_hypothesis': f'Cascading failure in service chain: {chain_key}',
                    'affected_services': chain_key.split(' -> '),
                    'alert_groups': groups,
                    'priority': 'high',
                    'investigation_focus': 'Check upstream service health and dependency flow'
                })
        
        return dependency_groups
    
    def _group_by_infrastructure(self):
        """Group alerts by infrastructure components"""
        infra_groups = []
        
        # Group by cluster
        cluster_alerts = defaultdict(list)
        for alert in self.enriched_alerts:
            cluster = alert.get('cluster', '') or alert.get('graph_cluster', '')
            if cluster.strip():
                cluster_alerts[cluster].append(alert)
        
        for cluster, alerts in cluster_alerts.items():
            if len(alerts) > 2:  # Multiple alerts in same cluster
                infra_groups.append({
                    'rca_type': 'infrastructure',
                    'root_cause_hypothesis': f'Infrastructure issue in cluster: {cluster}',
                    'affected_component': cluster,
                    'alerts': alerts,
                    'priority': 'high',
                    'investigation_focus': 'Check cluster health, node status, and resource availability'
                })
        
        # Group by node
        node_alerts = defaultdict(list)
        for alert in self.enriched_alerts:
            node = alert.get('node', '')
            if node.strip():
                node_alerts[node].append(alert)
        
        for node, alerts in node_alerts.items():
            if len(alerts) > 1:  # Multiple alerts on same node
                infra_groups.append({
                    'rca_type': 'infrastructure',
                    'root_cause_hypothesis': f'Node-level issue: {node}',
                    'affected_component': node,
                    'alerts': alerts,
                    'priority': 'medium',
                    'investigation_focus': 'Check node resource utilization and health'
                })
        
        return infra_groups
    
    def _group_by_resource_patterns(self):
        """Group alerts by resource type patterns"""
        resource_groups = []
        
        # Group by resource type
        resource_alerts = defaultdict(list)
        for alert in self.enriched_alerts:
            resource_type = alert.get('anomaly_resource_type', '')
            if resource_type.strip():
                resource_alerts[resource_type].append(alert)
        
        for resource_type, alerts in resource_alerts.items():
            if len(alerts) > 3:  # Multiple alerts of same resource type
                # Check if alerts span multiple services
                services = set(alert.get('service_name', '') for alert in alerts)
                if len(services) > 2:
                    resource_groups.append({
                        'rca_type': 'resource_pattern',
                        'root_cause_hypothesis': f'System-wide {resource_type} issue affecting multiple services',
                        'affected_resource': resource_type,
                        'affected_services': list(services),
                        'alerts': alerts,
                        'priority': 'high',
                        'investigation_focus': f'Check system-wide {resource_type} availability and limits'
                    })
        
        return resource_groups
    
    def _find_service_chain(self, services):
        """Find if services form a dependency chain"""
        # Try to find a path through the services
        for start_service in services:
            if start_service in self.service_graph:
                for end_service in services:
                    if (end_service != start_service and 
                        end_service in self.service_graph):
                        try:
                            path = nx.shortest_path(
                                self.service_graph, start_service, end_service
                            )
                            if len(path) > 1:
                                return ' -> '.join(path)
                        except:
                            continue
        return None
    
    def prepare_ml_features(self):
        """
        Prepare features for ML classification of root cause categories
        
        Features include:
        - Alert characteristics (severity, category, resource type)
        - Service graph features (centrality, connectivity)
        - Temporal features (time patterns)
        - Infrastructure features (cluster, namespace)
        - Relationship features (upstream/downstream counts)
        """
        print("Preparing ML features...")
        
        features = []
        labels = []
        
        for alert in self.enriched_alerts:
            feature_vector = {}
            
            # Alert characteristics
            feature_vector['severity_warning'] = 1 if alert.get('severity', '').lower() == 'warning' else 0
            feature_vector['severity_critical'] = 1 if alert.get('severity', '').lower() == 'critical' else 0
            feature_vector['category_anomaly'] = 1 if alert.get('alert_category', '').lower() == 'anomaly' else 0
            feature_vector['subcategory_resource'] = 1 if alert.get('alert_subcategory', '').lower() == 'resource' else 0
            
            # Resource type features
            resource_type = alert.get('anomaly_resource_type', '').lower()
            feature_vector['resource_cpu'] = 1 if 'cpu' in resource_type else 0
            feature_vector['resource_memory'] = 1 if 'memory' in resource_type else 0
            feature_vector['resource_network'] = 1 if 'network' in resource_type else 0
            feature_vector['resource_disk'] = 1 if 'disk' in resource_type else 0
            
            # Graph features
            feature_vector['service_centrality'] = alert.get('service_centrality', 0.0)
            feature_vector['upstream_count'] = alert.get('upstream_count', 0)
            feature_vector['downstream_count'] = alert.get('downstream_count', 0)
            feature_vector['related_services_count'] = alert.get('related_services_count', 0)
            
            # Infrastructure features
            feature_vector['has_cluster_info'] = 1 if alert.get('cluster', '') or alert.get('graph_cluster', '') else 0
            feature_vector['has_namespace_info'] = 1 if alert.get('namespace', '') or alert.get('graph_namespace', '') else 0
            feature_vector['has_node_info'] = 1 if alert.get('node', '') else 0
            
            # Temporal features
            try:
                start_time = pd.to_datetime(alert.get('starts_at', ''))
                feature_vector['hour_of_day'] = start_time.hour
                feature_vector['day_of_week'] = start_time.dayofweek
                feature_vector['is_weekend'] = 1 if start_time.dayofweek >= 5 else 0
            except:
                feature_vector['hour_of_day'] = 0
                feature_vector['day_of_week'] = 0
                feature_vector['is_weekend'] = 0
            
            # Workload type features
            workload_type = alert.get('workload_type', '').lower()
            feature_vector['workload_deployment'] = 1 if workload_type == 'deployment' else 0
            feature_vector['workload_daemonset'] = 1 if workload_type == 'daemonset' else 0
            feature_vector['workload_statefulset'] = 1 if workload_type == 'statefulset' else 0
            
            features.append(feature_vector)
            
            # Create synthetic labels for demonstration (in real scenario, these would be historical)
            label = self._generate_synthetic_root_cause_label(alert)
            labels.append(label)
        
        # Convert to DataFrame
        self.ml_features = pd.DataFrame(features)
        self.ml_labels = labels
        
        print(f"Prepared {len(features)} feature vectors with {len(feature_vector)} features each")
        print(f"Feature columns: {list(self.ml_features.columns)}")
        
        return self.ml_features, self.ml_labels
    
    def _generate_synthetic_root_cause_label(self, alert):
        """Generate synthetic root cause labels for demonstration"""
        # In real scenario, these would come from historical incident data
        
        resource_type = alert.get('anomaly_resource_type', '').lower()
        service_centrality = alert.get('service_centrality', 0.0)
        upstream_count = alert.get('upstream_count', 0)
        
        if 'network' in resource_type and service_centrality > 0.1:
            return 'network_infrastructure'
        elif 'cpu' in resource_type and upstream_count > 3:
            return 'capacity_scaling'
        elif 'memory' in resource_type:
            return 'memory_leak'
        elif service_centrality > 0.2:
            return 'service_dependency'
        else:
            return 'application_error'
    
    def train_root_cause_classifier(self):
        """
        Train ML classifier for root cause prediction
        
        This is a demonstration of how to set up ML pipeline.
        In production, you would use historical incident data with known root causes.
        """
        print("Training root cause classifier...")
        
        if self.ml_features is None:
            self.prepare_ml_features()
        
        # Prepare data
        X = self.ml_features.fillna(0)
        y = self.ml_labels
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test_scaled)
        
        print("Classification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_
        ))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Store trained components
        self.classifier = classifier
        self.scaler = scaler
        self.label_encoder = label_encoder
        
        return classifier, scaler, label_encoder
    
    def predict_root_causes(self):
        """Predict root causes for current alerts"""
        if not hasattr(self, 'classifier'):
            print("Training classifier first...")
            self.train_root_cause_classifier()
        
        # Predict for all alerts
        X = self.ml_features.fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        
        # Add predictions to alerts
        for i, alert in enumerate(self.enriched_alerts):
            predicted_label = self.label_encoder.inverse_transform([predictions[i]])[0]
            max_prob = np.max(probabilities[i])
            
            alert['predicted_root_cause'] = predicted_label
            alert['prediction_confidence'] = max_prob
        
        print("Root cause predictions completed")
        return predictions, probabilities
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE AIOPS ALERT ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        print(f"\n📊 ALERT STATISTICS:")
        print(f"  Total alerts processed: {len(self.pivoted_alerts)}")
        print(f"  Firing alerts: {len(self.firing_alerts)}")
        print(f"  Alerts with graph context: {len([a for a in self.enriched_alerts if a.get('service_centrality', 0) > 0])}")
        
        # Service analysis
        print(f"\n🔗 SERVICE GRAPH ANALYSIS:")
        print(f"  Services in graph: {self.service_graph.number_of_nodes()}")
        print(f"  Service relationships: {self.service_graph.number_of_edges()}")
        
        service_counts = Counter(alert.get('service_name', '') for alert in self.firing_alerts)
        print(f"\n🚨 TOP AFFECTED SERVICES:")
        for service, count in service_counts.most_common(5):
            if service.strip():
                centrality = 0
                if service in self.service_graph:
                    centrality = nx.degree_centrality(self.service_graph)[service]
                print(f"  {service}: {count} alerts (centrality: {centrality:.3f})")
        
        # Alert grouping results
        print(f"\n🎯 ALERT GROUPING RESULTS:")
        multi_groups = [g for g in self.alert_groups if len(g['alerts']) > 1]
        total_alerts_in_groups = sum(len(g['alerts']) for g in multi_groups)
        noise_reduction = (total_alerts_in_groups - len(multi_groups)) / len(self.firing_alerts) * 100 if self.firing_alerts else 0
        
        print(f"  Alert groups created: {len(self.alert_groups)}")
        print(f"  Multi-alert groups: {len(multi_groups)}")
        print(f"  Noise reduction achieved: {noise_reduction:.1f}%")
        
        # RCA groups
        print(f"\n🔍 ROOT CAUSE ANALYSIS GROUPS:")
        print(f"  RCA groups identified: {len(self.rca_groups)}")
        
        for rca_group in self.rca_groups[:3]:  # Show top 3
            print(f"\n  🎯 {rca_group['rca_type'].upper()}: {rca_group['root_cause_hypothesis']}")
            print(f"     Priority: {rca_group['priority']}")
            print(f"     Investigation: {rca_group['investigation_focus']}")
        
        # ML predictions
        if hasattr(self, 'classifier'):
            print(f"\n🤖 ML ROOT CAUSE PREDICTIONS:")
            predictions = Counter(alert.get('predicted_root_cause', '') for alert in self.enriched_alerts)
            for cause, count in predictions.most_common():
                if cause.strip():
                    print(f"  {cause}: {count} alerts")
        
        # Resource analysis
        print(f"\n📈 RESOURCE ANALYSIS:")
        resource_counts = Counter(
            alert.get('anomaly_resource_type', '') 
            for alert in self.firing_alerts 
            if alert.get('anomaly_resource_type', '').strip()
        )
        for resource, count in resource_counts.most_common():
            print(f"  {resource}: {count} alerts")
        
        return {
            'total_alerts': len(self.pivoted_alerts),
            'firing_alerts': len(self.firing_alerts),
            'alert_groups': len(self.alert_groups),
            'rca_groups': len(self.rca_groups),
            'noise_reduction_pct': noise_reduction,
            'top_services': dict(service_counts.most_common(5)),
            'top_resources': dict(resource_counts.most_common(5))
        }
    
    def save_results(self, output_dir='/data/redppo/'):
        """Save all results to files"""
        print(f"\n💾 SAVING RESULTS to {output_dir}")
        
        # Save enriched alerts
        enriched_df = pd.DataFrame(self.enriched_alerts)
        enriched_df.to_csv(f'{output_dir}enriched_firing_alerts.csv', index=False)
        print(f"  ✅ Saved enriched alerts: enriched_firing_alerts.csv")
        
        # Save alert groups
        groups_data = []
        for group in self.alert_groups:
            for alert in group['alerts']:
                alert_copy = alert.copy()
                alert_copy['group_id'] = group['group_id']
                alert_copy['group_size'] = len(group['alerts'])
                groups_data.append(alert_copy)
        
        groups_df = pd.DataFrame(groups_data)
        groups_df.to_csv(f'{output_dir}grouped_alerts.csv', index=False)
        print(f"  ✅ Saved grouped alerts: grouped_alerts.csv")
        
        # Save RCA groups
        with open(f'{output_dir}rca_groups.json', 'w') as f:
            json.dump(self.rca_groups, f, indent=2, default=str)
        print(f"  ✅ Saved RCA groups: rca_groups.json")
        
        # Save ML features
        if self.ml_features is not None:
            self.ml_features.to_csv(f'{output_dir}ml_features.csv', index=False)
            print(f"  ✅ Saved ML features: ml_features.csv")
        
        print(f"\n🎉 All results saved successfully!")

def main():
    """Main execution function"""
    print("="*80)
    print("GRAPH-ENRICHED AIOPS ALERT PROCESSING PIPELINE")
    print("="*80)
    
    # Initialize processor
    processor = GraphEnrichedAlertsProcessor(
        alerts_csv_path='/data/redppo/alert_data.csv',
        graph_json_path='/data/redppo/graph_data.json'
    )
    
    # Step 1: Load and pivot alerts
    processor.load_and_pivot_alerts()
    
    # Step 2: Load graph data
    processor.load_graph_data()
    
    # Step 3: Enrich alerts with graph context
    processor.enrich_alerts_with_graph_context()
    
    # Step 4: Group alerts with graph relationships
    processor.group_alerts_with_graph_context()
    
    # Step 5: Create RCA groups
    processor.create_rca_groups()
    
    # Step 6: Prepare ML features and train classifier
    processor.prepare_ml_features()
    processor.train_root_cause_classifier()
    
    # Step 7: Predict root causes
    processor.predict_root_causes()
    
    # Step 8: Generate comprehensive report
    results = processor.generate_comprehensive_report()
    
    # Step 9: Save all results
    processor.save_results()
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    return processor, results

if __name__ == "__main__":
    processor, results = main()