# alert events consolidation

import pandas as pd
import json
import ast
from collections import defaultdict, Counter
import networkx as nx
import difflib

class GraphEnrichedAlertProcessor:
    """
    Focused AIOps Alert Processing with Graph Relationships
    
    Key Features:
    1. Map alert service_name to graph source_properties.name
    2. Enrich alerts with CALLS and BELONGS_TO relationships
    3. Group alerts based on service dependencies and ownership
    """
    
    def __init__(self, alerts_csv_path, graph_json_path):
        self.alerts_csv_path = alerts_csv_path
        self.graph_json_path = graph_json_path
        
        # Data containers
        self.firing_alerts = []
        self.graph_relationships = []
        self.service_graph = nx.DiGraph()
        
        # Service mappings: service_name -> graph node info
        self.service_to_graph = {}
        
        # Results
        self.enriched_alerts = []
        self.grouped_alerts = []
        
    def load_firing_alerts(self):
        """Load and filter firing alerts"""
        print("Loading firing alerts...")
        
        df = pd.read_csv(self.alerts_csv_path)
        
        # Filter firing alerts only
        firing_df = df[df['status'].str.strip().str.lower() == 'firing']
        self.firing_alerts = firing_df.to_dict('records')
        
        print(f"Loaded {len(self.firing_alerts)} firing alerts from {len(df)} total alerts")
        
        # Parse payload for each alert
        for alert in self.firing_alerts:
            self._parse_alert_payload(alert)
        
        return self.firing_alerts
    
    def _parse_alert_payload(self, alert):
        """Parse payload_consolidated to extract metadata"""
        payload_str = alert.get('payload_consolidated', '')
        if not payload_str:
            return
        
        try:
            payload = ast.literal_eval(payload_str)
            
            # Extract labels (main metadata)
            labels = payload.get('labels', {})
            alert['cluster'] = labels.get('cluster', '')
            alert['namespace'] = labels.get('namespace', '')
            alert['pod'] = labels.get('pod', '')
            alert['node'] = labels.get('node', '')
            alert['anomaly_resource_type'] = labels.get('anomaly_resource_type', '')
            alert['workload_type'] = labels.get('workload_type', '')
            alert['platform'] = labels.get('platform', '')
            
            # Extract description
            annotations = payload.get('annotations', {})
            alert['description'] = annotations.get('description', '')
            
        except Exception as e:
            print(f"Warning: Could not parse payload for alert {alert.get('_id', 'unknown')}")
    
    def load_graph_data(self):
        """Load graph data and build service mappings"""
        print("Loading graph data...")
        
        with open(self.graph_json_path, 'r') as f:
            self.graph_relationships = json.load(f)
        
        print(f"Loaded {len(self.graph_relationships)} graph relationships")
        
        # Build service mappings and graph
        self.service_graph = nx.DiGraph()
        
        for rel in self.graph_relationships:
            # Extract source service info
            source_name = rel.get('source_name', '')
            source_props = rel.get('source_properties', {})
            source_service_name = source_props.get('name', '')  # This is the key mapping!
            
            # Extract target service info
            target_name = rel.get('target_name', '')
            target_props = rel.get('target_properties', {})
            target_service_name = target_props.get('name', '')
            
            # Store service mappings
            if source_service_name:
                self.service_to_graph[source_service_name] = {
                    'graph_name': source_name,
                    'properties': source_props,
                    'type': rel.get('source_label', ''),
                    'environment': source_props.get('environment', ''),
                    'namespace': source_props.get('namespace', ''),
                    'cluster': source_props.get('cluster', '')
                }
                
                # Add to NetworkX graph
                self.service_graph.add_node(
                    source_service_name, 
                    graph_name=source_name,
                    **source_props
                )
            
            if target_service_name:
                self.service_to_graph[target_service_name] = {
                    'graph_name': target_name,
                    'properties': target_props,
                    'type': rel.get('target_label', ''),
                    'environment': target_props.get('environment', ''),
                    'namespace': target_props.get('namespace', ''),
                    'cluster': target_props.get('cluster', '')
                }
                
                self.service_graph.add_node(
                    target_service_name,
                    graph_name=target_name, 
                    **target_props
                )
            
            # Add relationship edge
            rel_type = rel.get('relationship_type', '')
            if source_service_name and target_service_name and rel_type:
                self.service_graph.add_edge(
                    source_service_name,
                    target_service_name,
                    relationship_type=rel_type,
                    **rel.get('relationship_properties', {})
                )
        
        print(f"Built service graph with {len(self.service_to_graph)} services and {self.service_graph.number_of_edges()} relationships")
        return self.service_to_graph
    
    def enrich_alerts_with_graph_context(self):
        """Enrich alerts with graph relationship context"""
        print("Enriching alerts with graph context...")
        
        enriched_count = 0
        
        for alert in self.firing_alerts:
            service_name = alert.get('service_name', '').strip()
            
            # Initialize graph context fields
            alert['has_graph_context'] = False
            alert['calls_services'] = []
            alert['called_by_services'] = []
            alert['belongs_to_services'] = []
            alert['owns_services'] = []
            alert['related_alerts_count'] = 0
            
            if service_name in self.service_to_graph:
                # Service found in graph
                graph_info = self.service_to_graph[service_name]
                alert['has_graph_context'] = True
                alert['graph_type'] = graph_info['type']
                alert['graph_environment'] = graph_info['environment']
                alert['graph_namespace'] = graph_info['namespace']
                alert['graph_cluster'] = graph_info['cluster']
                
                # Find CALLS relationships (service dependencies)
                calls_services = []
                called_by_services = []
                
                for successor in self.service_graph.successors(service_name):
                    edge_data = self.service_graph.get_edge_data(service_name, successor)
                    if edge_data and edge_data.get('relationship_type') == 'CALLS':
                        calls_services.append(successor)
                
                for predecessor in self.service_graph.predecessors(service_name):
                    edge_data = self.service_graph.get_edge_data(predecessor, service_name)
                    if edge_data and edge_data.get('relationship_type') == 'CALLS':
                        called_by_services.append(predecessor)
                
                alert['calls_services'] = calls_services
                alert['called_by_services'] = called_by_services
                
                # Find BELONGS_TO relationships (ownership)
                belongs_to_services = []
                owns_services = []
                
                for successor in self.service_graph.successors(service_name):
                    edge_data = self.service_graph.get_edge_data(service_name, successor)
                    if edge_data and edge_data.get('relationship_type') == 'BELONGS_TO':
                        belongs_to_services.append(successor)
                
                for predecessor in self.service_graph.predecessors(service_name):
                    edge_data = self.service_graph.get_edge_data(predecessor, service_name)
                    if edge_data and edge_data.get('relationship_type') == 'BELONGS_TO':
                        owns_services.append(predecessor)
                
                alert['belongs_to_services'] = belongs_to_services
                alert['owns_services'] = owns_services
                
                enriched_count += 1
            else:
                # Service not found in graph
                alert['graph_type'] = 'unknown'
                alert['graph_environment'] = ''
                alert['graph_namespace'] = ''
                alert['graph_cluster'] = ''
        
        self.enriched_alerts = self.firing_alerts
        print(f"Enriched {enriched_count}/{len(self.firing_alerts)} alerts with graph context")
        
        return self.enriched_alerts
    
    def find_related_alerts(self, alert):
        """Find alerts related through graph relationships"""
        service_name = alert.get('service_name', '').strip()
        related_alerts = []
        
        if not alert.get('has_graph_context'):
            return related_alerts
        
        # Get all related services (calls + belongs_to)
        related_services = set()
        related_services.update(alert.get('calls_services', []))
        related_services.update(alert.get('called_by_services', []))
        related_services.update(alert.get('belongs_to_services', []))
        related_services.update(alert.get('owns_services', []))
        
        # Find alerts for related services
        for other_alert in self.enriched_alerts:
            other_service = other_alert.get('service_name', '').strip()
            if other_service != service_name and other_service in related_services:
                # Determine relationship type
                relationship_type = 'unknown'
                if other_service in alert.get('calls_services', []):
                    relationship_type = 'calls'
                elif other_service in alert.get('called_by_services', []):
                    relationship_type = 'called_by'
                elif other_service in alert.get('belongs_to_services', []):
                    relationship_type = 'belongs_to'
                elif other_service in alert.get('owns_services', []):
                    relationship_type = 'owns'
                
                other_alert['relationship_to_primary'] = relationship_type
                related_alerts.append(other_alert)
        
        return related_alerts
    
    def calculate_enhanced_similarity(self, alert1, alert2):
        """
        Calculate similarity with graph relationship context
        
        Factors:
        1. Service relationship (50% weight) - CALLS/BELONGS_TO connections
        2. Resource type similarity (25% weight)
        3. Infrastructure context (15% weight) - cluster, namespace
        4. Description similarity (10% weight)
        """
        service1 = alert1.get('service_name', '').strip()
        service2 = alert2.get('service_name', '').strip()
        
        similarities = []
        
        # 1. Service relationship similarity (highest weight)
        relationship_score = 0.0
        
        if service1 == service2:
            relationship_score = 1.0  # Same service
        elif (service2 in alert1.get('calls_services', []) or 
              service2 in alert1.get('called_by_services', []) or
              service2 in alert1.get('belongs_to_services', []) or
              service2 in alert1.get('owns_services', [])):
            relationship_score = 0.8  # Direct relationship
        elif alert1.get('has_graph_context') and alert2.get('has_graph_context'):
            # Check for shared dependencies
            services1 = set(alert1.get('calls_services', []) + alert1.get('called_by_services', []))
            services2 = set(alert2.get('calls_services', []) + alert2.get('called_by_services', []))
            
            if services1.intersection(services2):
                relationship_score = 0.4  # Shared dependencies
        
        similarities.append(('service_relationship', relationship_score, 0.5))
        
        # 2. Resource type similarity
        resource1 = alert1.get('anomaly_resource_type', '').lower().strip()
        resource2 = alert2.get('anomaly_resource_type', '').lower().strip()
        resource_sim = 1.0 if resource1 and resource2 and resource1 == resource2 else 0.0
        similarities.append(('resource_type', resource_sim, 0.25))
        
        # 3. Infrastructure context similarity
        infra_score = 0.0
        cluster1 = alert1.get('cluster', '') or alert1.get('graph_cluster', '')
        cluster2 = alert2.get('cluster', '') or alert2.get('graph_cluster', '')
        namespace1 = alert1.get('namespace', '') or alert1.get('graph_namespace', '')
        namespace2 = alert2.get('namespace', '') or alert2.get('graph_namespace', '')
        
        if cluster1 and cluster2 and cluster1 == cluster2:
            infra_score += 0.6
        if namespace1 and namespace2 and namespace1 == namespace2:
            infra_score += 0.4
        
        infra_score = min(infra_score, 1.0)
        similarities.append(('infrastructure', infra_score, 0.15))
        
        # 4. Description similarity
        desc1 = alert1.get('description', '').lower().strip()
        desc2 = alert2.get('description', '').lower().strip()
        desc_sim = 0.0
        
        if desc1 and desc2 and len(desc1) > 20 and len(desc2) > 20:
            desc_sim = difflib.SequenceMatcher(None, desc1, desc2).ratio()
        
        similarities.append(('description', desc_sim, 0.1))
        
        # Calculate weighted similarity
        total_score = sum(score * weight for _, score, weight in similarities)
        total_weight = sum(weight for _, _, weight in similarities)
        overall_similarity = total_score / total_weight if total_weight > 0 else 0.0
        
        breakdown = {factor: score for factor, score, _ in similarities}
        breakdown['overall'] = overall_similarity
        
        return overall_similarity, breakdown
    
    def group_alerts_by_relationships(self, similarity_threshold=0.6):
        """Group alerts based on graph relationships and similarity"""
        print(f"Grouping alerts by relationships (threshold: {similarity_threshold})...")
        
        alert_groups = []
        processed_indices = set()
        
        for i, alert1 in enumerate(self.enriched_alerts):
            if i in processed_indices:
                continue
            
            current_group = {
                'group_id': len(alert_groups) + 1,
                'alerts': [alert1],
                'representative': alert1,
                'group_type': 'single',
                'relationships': [],
                'similarities': []
            }
            
            # Find related alerts through graph relationships
            related_alerts = self.find_related_alerts(alert1)
            
            # Add directly related alerts
            for related_alert in related_alerts:
                # Find index of related alert
                for j, enriched_alert in enumerate(self.enriched_alerts):
                    if (enriched_alert.get('_id') == related_alert.get('_id') and 
                        j not in processed_indices and j != i):
                        
                        current_group['alerts'].append(related_alert)
                        current_group['relationships'].append({
                            'alert_index': j,
                            'relationship_type': related_alert.get('relationship_to_primary', 'unknown'),
                            'service_name': related_alert.get('service_name', '')
                        })
                        current_group['group_type'] = 'relationship_based'
                        processed_indices.add(j)
                        break
            
            # Also check for high similarity alerts
            for j, alert2 in enumerate(self.enriched_alerts[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                similarity, breakdown = self.calculate_enhanced_similarity(alert1, alert2)
                
                if similarity >= similarity_threshold:
                    current_group['alerts'].append(alert2)
                    current_group['similarities'].append({
                        'alert_index': j,
                        'similarity_score': similarity,
                        'breakdown': breakdown
                    })
                    if current_group['group_type'] == 'single':
                        current_group['group_type'] = 'similarity_based'
                    elif current_group['group_type'] == 'relationship_based':
                        current_group['group_type'] = 'hybrid'
                    
                    processed_indices.add(j)
            
            alert_groups.append(current_group)
            processed_indices.add(i)
        
        self.grouped_alerts = alert_groups
        
        # Statistics
        relationship_groups = [g for g in alert_groups if 'relationship' in g['group_type']]
        similarity_groups = [g for g in alert_groups if 'similarity' in g['group_type']]
        multi_alert_groups = [g for g in alert_groups if len(g['alerts']) > 1]
        
        print(f"Created {len(alert_groups)} groups:")
        print(f"  Relationship-based groups: {len(relationship_groups)}")
        print(f"  Similarity-based groups: {len(similarity_groups)}")
        print(f"  Multi-alert groups: {len(multi_alert_groups)}")
        
        return alert_groups
    
    def analyze_grouping_results(self):
        """Analyze and report grouping effectiveness"""
        print("\n" + "="*60)
        print("GRAPH-ENRICHED ALERT ANALYSIS RESULTS")
        print("="*60)
        
        # Basic statistics
        total_alerts = len(self.firing_alerts)
        enriched_alerts = len([a for a in self.enriched_alerts if a.get('has_graph_context')])
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"  Total firing alerts: {total_alerts}")
        print(f"  Alerts with graph context: {enriched_alerts}")
        print(f"  Graph coverage: {enriched_alerts/total_alerts*100:.1f}%")
        
        # Service mapping results
        print(f"\n🔗 SERVICE MAPPING:")
        print(f"  Services in graph: {len(self.service_to_graph)}")
        
        # Alert service analysis
        alert_services = Counter(alert.get('service_name', '') for alert in self.firing_alerts)
        mapped_services = [s for s in alert_services.keys() if s in self.service_to_graph]
        
        print(f"  Alert services mapped: {len(mapped_services)}/{len(alert_services)}")
        
        # Relationship analysis
        calls_relationships = sum(len(alert.get('calls_services', [])) for alert in self.enriched_alerts)
        belongs_relationships = sum(len(alert.get('belongs_to_services', [])) for alert in self.enriched_alerts)
        
        print(f"\n🔄 RELATIONSHIP ANALYSIS:")
        print(f"  Total CALLS relationships: {calls_relationships}")
        print(f"  Total BELONGS_TO relationships: {belongs_relationships}")
        
        # Grouping results
        multi_groups = [g for g in self.grouped_alerts if len(g['alerts']) > 1]
        relationship_groups = [g for g in multi_groups if 'relationship' in g['group_type']]
        
        print(f"\n🎯 GROUPING RESULTS:")
        print(f"  Total groups: {len(self.grouped_alerts)}")
        print(f"  Multi-alert groups: {len(multi_groups)}")
        print(f"  Relationship-based groups: {len(relationship_groups)}")
        
        if multi_groups:
            total_grouped_alerts = sum(len(g['alerts']) for g in multi_groups)
            reduction_pct = (total_grouped_alerts - len(multi_groups)) / total_alerts * 100
            print(f"  Noise reduction: {reduction_pct:.1f}%")
        
        # Show top groups
        if multi_groups:
            print(f"\n🔍 TOP ALERT GROUPS:")
            sorted_groups = sorted(multi_groups, key=lambda x: len(x['alerts']), reverse=True)
            
            for i, group in enumerate(sorted_groups[:5], 1):
                rep = group['representative']
                print(f"\n  {i}. GROUP {group['group_id']} ({group['group_type']}) - {len(group['alerts'])} alerts")
                print(f"     Primary Service: {rep.get('service_name', 'Unknown')}")
                print(f"     Resource Type: {rep.get('anomaly_resource_type', 'Unknown')}")
                
                # Show relationships
                if group.get('relationships'):
                    rel_services = [r['service_name'] for r in group['relationships'][:3]]
                    print(f"     Related Services: {', '.join(rel_services)}")
        
        return {
            'total_alerts': total_alerts,
            'enriched_alerts': enriched_alerts,
            'mapped_services': len(mapped_services),
            'total_groups': len(self.grouped_alerts),
            'multi_groups': len(multi_groups),
            'relationship_groups': len(relationship_groups)
        }
    
    def save_results(self, output_dir='/data/redppo/'):
        """Save enriched and grouped results"""
        print(f"\n💾 SAVING RESULTS to {output_dir}")
        
        # Save enriched alerts
        enriched_df = pd.DataFrame(self.enriched_alerts)
        enriched_df.to_csv(f'{output_dir}graph_enriched_alerts.csv', index=False)
        print(f"  ✅ Saved enriched alerts: graph_enriched_alerts.csv")
        
        # Save grouped alerts
        grouped_data = []
        for group in self.grouped_alerts:
            for alert in group['alerts']:
                alert_copy = alert.copy()
                alert_copy['group_id'] = group['group_id']
                alert_copy['group_type'] = group['group_type']
                alert_copy['group_size'] = len(group['alerts'])
                grouped_data.append(alert_copy)
        
        grouped_df = pd.DataFrame(grouped_data)
        grouped_df.to_csv(f'{output_dir}relationship_grouped_alerts.csv', index=False)
        print(f"  ✅ Saved grouped alerts: relationship_grouped_alerts.csv")
        
        print(f"\n🎉 Processing complete!")

def main():
    """Main execution function"""
    print("="*60)
    print("GRAPH-ENRICHED ALERT PROCESSING")
    print("="*60)
    
    # Initialize processor
    processor = GraphEnrichedAlertProcessor(
        alerts_csv_path='/data/redppo/alerts_pivoted_deduplicated.csv',
        graph_json_path='/data/redppo/graph_data.json'
    )
    
    # Step 1: Load firing alerts
    processor.load_firing_alerts()
    
    # Step 2: Load graph data and build service mappings
    processor.load_graph_data()
    
    # Step 3: Enrich alerts with graph context
    processor.enrich_alerts_with_graph_context()
    
    # Step 4: Group alerts by relationships and similarity
    processor.group_alerts_by_relationships()
    
    # Step 5: Analyze results
    results = processor.analyze_grouping_results()
    
    # Step 6: Save results
    processor.save_results()
    
    return processor, results

if __name__ == "__main__":
    processor, results = main()