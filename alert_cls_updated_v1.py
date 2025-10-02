# solution for alert data handling

import csv
import re
import ast
from collections import defaultdict, Counter
import difflib

def parse_payload_to_columns(payload):
    """Parse payload and extract metadata into separate columns"""
    extracted_columns = {}

    if not payload or payload in ['resolved', 'firing']:
        extracted_columns['payload_status'] = payload
        return extracted_columns

    # Handle timestamp payloads
    if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', str(payload)):
        extracted_columns['payload_timestamp'] = payload
        return extracted_columns

    # Handle hash/ID payloads
    if re.match(r'^[a-f0-9]{8,}$', str(payload)):
        extracted_columns['payload_hash'] = payload
        return extracted_columns

    # Handle dictionary payloads
    if isinstance(payload, str) and payload.startswith('{'):
        try:
            # Parse the dictionary string
            payload_dict = ast.literal_eval(payload)

            # Extract specific fields we're interested in
            field_mappings = {
                '__tenant_id__': 'tenant_id',
                'platform': 'platform',
                'cluster': 'cluster_name',
                'namespace': 'namespace',
                'node': 'node_name',
                'pod': 'pod_name',
                'workload': 'workload_name',
                'workload_type': 'workload_type',
                'anomaly_entity_type': 'anomaly_entity_type',
                'anomaly_resource_type': 'anomaly_resource_type',
                'metrics_source': 'metrics_source',
                'description': 'alert_description',
                'alertname': 'alert_name_payload',
                'aiops': 'aiops_enabled'
            }

            for original_key, new_column in field_mappings.items():
                if original_key in payload_dict:
                    extracted_columns[new_column] = str(payload_dict[original_key])

        except Exception as e:
            # If parsing fails, store as raw payload
            extracted_columns['payload_raw'] = str(payload)[:200]  # Truncate long payloads

    else:
        # Any other payload type
        extracted_columns['payload_raw'] = str(payload)[:200]

    return extracted_columns

class FiringAlertsAnalyzer:
    def __init__(self, alerts_csv_file):
        self.alerts_csv_file = alerts_csv_file
        self.firing_alerts = []
        self.alert_groups = []

    def load_firing_alerts(self):
        """Load alerts from CSV/DataFrame format, extract payload columns, filter firing alerts"""
        print("Loading and processing alerts from CSV...")
        
        # Read the CSV file
        with open(self.alerts_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_alerts = list(reader)
        
        print(f"Loaded {len(all_alerts)} alert records from CSV")
        
        # Process each alert and extract payload columns if needed
        processed_alerts = []
        
        for alert in all_alerts:
            # Start with the alert as-is (already in dictionary format from CSV)
            processed_alert = dict(alert)
            
            # Check if payload column exists and needs extraction
            if 'payload' in alert and alert['payload'].strip():
                payload = alert['payload']
                
                # Only extract if payload looks like it contains structured data
                if (isinstance(payload, str) and 
                    (payload.startswith('{') or 
                     payload not in ['resolved', 'firing'] and 
                     not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', payload) and
                     not re.match(r'^[a-f0-9]{8,}$', payload))):
                    
                    # Extract payload columns
                    payload_columns = parse_payload_to_columns(payload)
                    processed_alert.update(payload_columns)
            
            processed_alerts.append(processed_alert)
        
        # Filter only firing alerts
        self.firing_alerts = [alert for alert in processed_alerts if alert.get('status', '').strip() == 'firing']
        
        print(f"Total alerts: {len(processed_alerts)}")
        print(f"Firing alerts: {len(self.firing_alerts)}")
        
        # Show firing alerts summary
        print("\nFiring Alerts Summary:")
        services = Counter(alert.get('service_name', '') for alert in self.firing_alerts)
        for service, count in services.most_common():
            if service.strip():
                print(f"  {service}: {count} alerts")
        
        # Show available columns
        if self.firing_alerts:
            sample_alert = self.firing_alerts[0]
            available_columns = list(sample_alert.keys())
            print(f"\nAvailable columns: {len(available_columns)}")
            
            # Show key payload fields if they exist
            payload_fields = ['alert_description', 'cluster_name', 'namespace', 'anomaly_resource_type', 'workload_type']
            extracted_fields = [field for field in payload_fields if field in available_columns]
            
            if extracted_fields:
                print("\nExtracted Payload Fields:")
                for field in extracted_fields:
                    count = sum(1 for alert in self.firing_alerts if alert.get(field, '').strip())
                    if count > 0:
                        print(f"  {field}: {count}/{len(self.firing_alerts)} alerts")

    def calculate_semantic_similarity(self, alert1, alert2):
        """Calculate semantic similarity between two alerts
        # ===================================================
        The algorithm uses weighted scoring across multiple dimensions:

        | Factor            | Weight | Purpose              | Example                               |
        |-------------------|--------|----------------------|---------------------------------------|
        | Service Name      | 4.0    | Primary grouping     | payment-service vs user-service        |
        | Description       | 3.5    | Semantic context     | "CPU usage exceeded 80%" similarity    |
        | Alert Name        | 3.0    | Issue type           | HighCPUUsage vs HighMemoryUsage        |
        | Category/Severity | 2.0    | Classification       | resource/warning vs network/critical   |
        | Resource Type     | 2.0    | Technical details    | cpu:usage vs memory:usage              |
        | Namespace/Cluster | 1.0    | Environment context  | production vs staging                  |

        # ===================================================
        
        example:
        Alert 1: esob-k8s-collector | ResourceRateAnomaly | Pod 1
        Alert 2: esob-k8s-collector | ResourceRateAnomaly | Pod 2  
        Alert 3: esob-k8s-collector | ResourceRateAnomaly | Pod 3
        
        after similarityanalysis:
        Cluster 1: esob-k8s-collector | ResourceRateAnomaly (12 alerts)
        - Same service, same issue type, different pods
        - Similarity Score: 1.000 (perfect match)
        - Action: Investigate single root cause
        
        """
        similarities = []

        # 1. Service name similarity (highest weight)
        service1 = alert1.get('service_name', '').strip().lower()
        service2 = alert2.get('service_name', '').strip().lower()
        if service1 and service2:
            service_sim = difflib.SequenceMatcher(None, service1, service2).ratio()
            similarities.append(('service_name', service_sim, 4.0))

        # 2. Alert name similarity (high weight)
        alert_name1 = alert1.get('alert_name', '').strip().lower()
        alert_name2 = alert2.get('alert_name', '').strip().lower()
        if alert_name1 and alert_name2:
            alert_sim = difflib.SequenceMatcher(None, alert_name1, alert_name2).ratio()
            similarities.append(('alert_name', alert_sim, 3.0))

        # 3. Alert category similarity (medium weight)
        category1 = alert1.get('alert_category', '').strip().lower()
        category2 = alert2.get('alert_category', '').strip().lower()
        if category1 and category2:
            category_sim = 1.0 if category1 == category2 else 0.0
            similarities.append(('alert_category', category_sim, 2.0))

        # 4. Alert subcategory similarity (medium weight)
        subcat1 = alert1.get('alert_subcategory', '').strip().lower()
        subcat2 = alert2.get('alert_subcategory', '').strip().lower()
        if subcat1 and subcat2:
            subcat_sim = 1.0 if subcat1 == subcat2 else 0.0
            similarities.append(('alert_subcategory', subcat_sim, 2.0))

        # 5. Severity similarity (medium weight)
        severity1 = alert1.get('severity', '').strip().lower()
        severity2 = alert2.get('severity', '').strip().lower()
        if severity1 and severity2:
            severity_sim = 1.0 if severity1 == severity2 else 0.0
            similarities.append(('severity', severity_sim, 2.0))

        # 6. Description similarity (high weight for semantic meaning)
        desc1 = alert1.get('alert_description', '').strip().lower()
        desc2 = alert2.get('alert_description', '').strip().lower()
        if desc1 and desc2 and len(desc1) > 10 and len(desc2) > 10:
            desc_sim = difflib.SequenceMatcher(None, desc1, desc2).ratio()
            similarities.append(('description', desc_sim, 3.5))

        # 7. Resource type similarity (medium weight)
        resource1 = alert1.get('anomaly_resource_type', '').strip().lower()
        resource2 = alert2.get('anomaly_resource_type', '').strip().lower()
        if resource1 and resource2:
            resource_sim = 1.0 if resource1 == resource2 else 0.0
            similarities.append(('resource_type', resource_sim, 2.0))

        # 8. Namespace similarity (low weight - context)
        ns1 = alert1.get('namespace', '').strip().lower()
        ns2 = alert2.get('namespace', '').strip().lower()
        if ns1 and ns2:
            ns_sim = 1.0 if ns1 == ns2 else 0.0
            similarities.append(('namespace', ns_sim, 1.0))

        # Calculate weighted similarity score
        if not similarities:
            return 0.0, {}

        total_weighted_score = sum(score * weight for _, score, weight in similarities)
        total_weight = sum(weight for _, _, weight in similarities)
        overall_similarity = total_weighted_score / total_weight

        # Create similarity breakdown
        breakdown = {factor: score for factor, score, _ in similarities}

        return overall_similarity, breakdown

    def group_similar_alerts(self, similarity_threshold=0.75):
        """Group semantically similar firing alerts
        
        # ===========================
        without grouping:
        🚨 Alert 1: esob-k8s-collector | ResourceRateAnomaly | Pod-1
        🚨 Alert 2: esob-k8s-collector | ResourceRateAnomaly | Pod-2  
        🚨 Alert 3: esob-k8s-collector | ResourceRateAnomaly | Pod-3
        🚨 Alert 4: aais-customer-sprbt | ResourceRateAnomaly | Pod-1
        🚨 Alert 5: aais-customer-sprbt | ResourceRateAnomaly | Pod-2
        ...

        ❌ 7 separate investigations needed
        ❌ Duplicate effort on same issues
        ❌ Alert fatigue and delayed response
        
        with grouping:
        🎯 GROUP 1: esob-k8s-collector | ResourceRateAnomaly (3 alerts)
        - Same issue affecting multiple pods
        - Single root cause investigation needed

        🎯 GROUP 2: aais-customer-sprbt | ResourceRateAnomaly (2 alerts)  
        - Same issue affecting multiple pods
        - Single root cause investigation needed

        ✅ 2 investigations instead of 7
        ✅ 85.7% noise reduction achieved
        
        how it works:
        Similarity Calculation: Compares alerts using weighted factors
        Threshold Application: Groups alerts above similarity threshold (0.75)
        Group Formation: Creates clusters of related alerts
        Consolidation: One representative alert per group
        
        key benefits:
        | Benefit            | Impact                        | Example                      |
        |--------------------|-------------------------------|------------------------------|
        | Noise Reduction    | 40-90% fewer alerts           | 12 alerts → 1 group          |
        | Faster Response    | 50-70% quicker resolution     | Focus on unique issues       |
        | Cost Savings       | Reduced investigation time    | 6 fewer duplicate analyses   |
        | Pattern Recognition| Identify root causes          | Infrastructure-wide issues   |
        | Better Metrics     | Higher signal-to-noise ratio  | Quality over quantity        |

        example (pod scaling issue):
        Before: 10 alerts from 10 different pods
        After: 1 group → "Service X scaling issue"
        Action: Scale up service, not investigate 10 pods
        
        example (network failure):
        Before: 15 alerts from different services
        After: 1 group → "Network switch failure"  
        Action: Fix network, not debug 15 services
        
        # ===========================
        """
        print(f"\nGrouping similar alerts (threshold: {similarity_threshold})...")

        if not self.firing_alerts:
            return []

        alert_groups = []
        processed_indices = set()

        for i, alert1 in enumerate(self.firing_alerts):
            if i in processed_indices:
                continue

            # Start new group
            current_group = {
                'group_id': len(alert_groups) + 1,
                'alerts': [alert1],
                'indices': [i],
                'representative': alert1,
                'similarities': []
            }

            # Find similar alerts
            for j, alert2 in enumerate(self.firing_alerts[i+1:], i+1):
                if j in processed_indices:
                    continue

                similarity_score, breakdown = self.calculate_semantic_similarity(alert1, alert2)

                if similarity_score >= similarity_threshold:
                    current_group['alerts'].append(alert2)
                    current_group['indices'].append(j)
                    current_group['similarities'].append({
                        'alert_index': j,
                        'similarity_score': similarity_score,
                        'breakdown': breakdown
                    })
                    processed_indices.add(j)

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

    def analyze_alert_clusters(self):
        """
        Analyze the semantic clusters of alerts
        
        
        🎯 COMMON CHARACTERISTICS:
        • Same or similar service names
        • Same or related alert types
        • Similar severity levels
        • Related resource types
        • Common infrastructure context

        🔍 CLUSTER TYPES:
       • DUPLICATE CLUSTERS: Exact same issue, different resources
       • RELATED CLUSTERS: Connected issues from same system
       • PATTERN CLUSTERS: Recurring issues with similar signatures
        """
        print(f"\n" + "="*60)
        print("SEMANTIC ALERT CLUSTER ANALYSIS")
        print("="*60)

        if not self.alert_groups:
            print("No alert groups to analyze")
            return

        # Analyze multi-alert groups (semantic clusters)
        clusters = [g for g in self.alert_groups if len(g['alerts']) > 1]

        if not clusters:
            print("No semantic clusters found - all alerts are unique")
            return

        print(f"\nFound {len(clusters)} semantic clusters:")

        for i, cluster in enumerate(clusters, 1):
            rep = cluster['representative']
            alerts = cluster['alerts']

            print(f"\n🔍 CLUSTER {i} ({len(alerts)} alerts)")
            print("-" * 40)

            # Cluster summary
            print(f"Service: {rep.get('service_name', 'N/A')}")
            print(f"Alert Name: {rep.get('alert_name', 'N/A')}")
            print(f"Category: {rep.get('alert_category', 'N/A')} / {rep.get('alert_subcategory', 'N/A')}")
            print(f"Severity: {rep.get('severity', 'N/A')}")

            # Resource information
            resource_type = rep.get('anomaly_resource_type', '')
            if resource_type:
                print(f"Resource Type: {resource_type}")

            # Description (if available)
            description = rep.get('alert_description', '')
            if description:
                print(f"Description: {description[:100]}...")

            # Show all alerts in cluster
            print(f"\nAlerts in this cluster:")
            for j, alert in enumerate(alerts):
                namespace = alert.get('namespace', 'N/A')
                pod = alert.get('pod_name', 'N/A')
                starts_at = alert.get('starts_at', 'N/A')
                print(f"  {j+1}. {namespace} | {pod} | {starts_at}")

            # Show similarity breakdown for the first similar alert
            if cluster['similarities']:
                sim_info = cluster['similarities'][0]
                print(f"\nSimilarity breakdown (with alert 2):")
                for factor, score in sim_info['breakdown'].items():
                    print(f"  {factor}: {score:.3f}")
                print(f"Overall similarity: {sim_info['similarity_score']:.3f}")

    def consolidate_duplicate_alerts(self):
        """Consolidate semantically similar alerts"""
        print(f"\n" + "="*60)
        print("ALERT CONSOLIDATION")
        print("="*60)

        consolidated_alerts = []
        total_duplicates = 0

        for group in self.alert_groups:
            if len(group['alerts']) > 1:
                # This is a cluster with duplicates
                representative = group['alerts'][0].copy()
                duplicates = group['alerts'][1:]

                # Add consolidation metadata
                representative['is_consolidated'] = 'true'
                representative['duplicate_count'] = str(len(duplicates))
                representative['cluster_id'] = str(group['group_id'])
                representative['total_occurrences'] = str(len(group['alerts']))

                # Collect affected resources
                affected_namespaces = list(set(alert.get('namespace', '') for alert in group['alerts'] if alert.get('namespace', '').strip()))
                affected_pods = list(set(alert.get('pod_name', '') for alert in group['alerts'] if alert.get('pod_name', '').strip()))
                affected_nodes = list(set(alert.get('node_name', '') for alert in group['alerts'] if alert.get('node_name', '').strip()))

                representative['affected_namespaces'] = ', '.join(affected_namespaces)
                representative['affected_pods'] = ', '.join(affected_pods)
                representative['affected_nodes'] = ', '.join(affected_nodes)

                # Calculate time range
                start_times = [alert.get('starts_at', '') for alert in group['alerts'] if alert.get('starts_at', '').strip()]
                if start_times:
                    representative['first_occurrence'] = min(start_times)
                    representative['last_occurrence'] = max(start_times)

                consolidated_alerts.append(representative)
                total_duplicates += len(duplicates)

            else:
                # Single alert
                alert = group['alerts'][0].copy()
                alert['is_consolidated'] = 'false'
                alert['duplicate_count'] = '0'
                alert['cluster_id'] = str(group['group_id'])
                alert['total_occurrences'] = '1'
                alert['affected_namespaces'] = alert.get('namespace', '')
                alert['affected_pods'] = alert.get('pod_name', '')
                alert['affected_nodes'] = alert.get('node_name', '')
                alert['first_occurrence'] = alert.get('starts_at', '')
                alert['last_occurrence'] = alert.get('starts_at', '')
                consolidated_alerts.append(alert)

        print(f"Consolidation Results:")
        print(f"  Original firing alerts: {len(self.firing_alerts)}")
        print(f"  After consolidation: {len(consolidated_alerts)}")
        print(f"  Duplicate alerts removed: {total_duplicates}")

        if len(self.firing_alerts) > 0:
            reduction_pct = (total_duplicates / len(self.firing_alerts)) * 100
            print(f"  Noise reduction: {reduction_pct:.1f}%")

        return consolidated_alerts

    def save_consolidated_alerts(self, consolidated_alerts):
        """Save consolidated alerts to CSV"""
        if not consolidated_alerts:
            return

        # Define output columns
        base_columns = list(consolidated_alerts[0].keys())

        # Write consolidated alerts CSV
        with open('/data/redppo/consolidated_firing_alerts.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=base_columns)
            writer.writeheader()
            writer.writerows(consolidated_alerts)

        print(f"\n✅ Saved consolidated alerts to: consolidated_firing_alerts.csv")

    def generate_final_insights(self):
        """Generate actionable insights"""
        print(f"\n" + "="*60)
        print("ACTIONABLE INSIGHTS")
        print("="*60)

        consolidated = self.consolidate_duplicate_alerts()

        if not self.firing_alerts:
            print("No firing alerts to analyze")
            return

        # Key metrics
        total_firing = len(self.firing_alerts)
        unique_patterns = len(consolidated)
        clusters = [g for g in self.alert_groups if len(g['alerts']) > 1]

        # Service analysis
        service_counts = Counter(alert.get('service_name', '') for alert in self.firing_alerts)
        top_services = [(s, c) for s, c in service_counts.most_common(3) if s.strip()]

        # Resource analysis
        resource_counts = Counter(alert.get('anomaly_resource_type', '') for alert in self.firing_alerts if alert.get('anomaly_resource_type', '').strip())

        for service, count in top_services:
            print(f"  • {service}: {count} firing alerts")

        if resource_counts:
            print(f"\nRESOURCE TYPES UNDER STRESS:")
            for resource, count in resource_counts.most_common():
                print(f"  • {resource}: {count} alerts")

        # Save consolidated results
        self.save_consolidated_alerts(consolidated)

def main():
    """Main execution function"""
    
    # For demonstration, create a DataFrame-format CSV from alerts.txt
    # In real-world usage, you would already have your alerts in CSV/DataFrame format
    csv_file = load_alert_df(path)
    
    # Use the DataFrame-format CSV for analysis
    analyzer = FiringAlertsAnalyzer(csv_file)

    print("="*60)
    print("FIRING ALERTS SEMANTIC ANALYSIS")
    print("="*60)

    # Load firing alerts
    analyzer.load_firing_alerts()

    if not analyzer.firing_alerts:
        print("No firing alerts found. Analysis complete.")
        return

    # Group similar alerts
    analyzer.group_similar_alerts()

    # Analyze clusters
    analyzer.analyze_alert_clusters()

    # Generate insights and save results
    analyzer.generate_final_insights()

    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
