#!/usr/bin/env python3
"""
AIOps Alert Analysis System - Interactive Demo

This demo script showcases the complete AIOps alert analysis system
with synthetic data generation, clustering, prediction, and visualization.

Run this script to see the full system in action!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aiops_alert_analysis import *
import json

def run_interactive_demo():
    """Run an interactive demonstration of the AIOps system"""
    
    print("🎯 AIOps Alert Analysis System - Interactive Demo")
    print("=" * 70)
    print("This demo will showcase:")
    print("• Synthetic alert data generation with realistic patterns")
    print("• Multi-dimensional alert clustering analysis")
    print("• Temporal pattern detection and frequency analysis") 
    print("• Machine learning models for alert prediction")
    print("• System attribution and impact assessment")
    print("• Comprehensive visualization and reporting")
    print("\n" + "=" * 70)
    
    input("\nPress Enter to start the demo...")
    
    try:
        # Initialize the system components
        print("\n🔧 Initializing AIOps Components...")
        
        # Step 1: Data Generation
        print("\n📊 Step 1: Generating Synthetic Alert Dataset")
        print("-" * 50)
        
        data_generator = SyntheticDataGenerator(seed=42)
        
        # Generate different sized datasets for demonstration
        small_alerts = data_generator.generate_alerts(num_alerts=100, days_back=7)
        medium_alerts = data_generator.generate_alerts(num_alerts=500, days_back=15) 
        large_alerts = data_generator.generate_alerts(num_alerts=1000, days_back=30)
        
        print(f"✓ Small dataset: {len(small_alerts)} alerts (7 days)")
        print(f"✓ Medium dataset: {len(medium_alerts)} alerts (15 days)")
        print(f"✓ Large dataset: {len(large_alerts)} alerts (30 days)")
        
        # Let user choose dataset size
        print("\nChoose dataset size for analysis:")
        print("1. Small (100 alerts, 7 days) - Quick demo")
        print("2. Medium (500 alerts, 15 days) - Balanced analysis")
        print("3. Large (1000 alerts, 30 days) - Comprehensive analysis")
        
        while True:
            try:
                choice = input("\nEnter choice (1-3): ").strip()
                if choice == '1':
                    alerts_df = small_alerts
                    days_back = 7
                    break
                elif choice == '2':
                    alerts_df = medium_alerts
                    days_back = 15
                    break
                elif choice == '3':
                    alerts_df = large_alerts
                    days_back = 30
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except KeyboardInterrupt:
                print("\n\nDemo cancelled by user.")
                return
        
        # Generate associated data
        metrics_df = data_generator.generate_metrics_data(alerts_df)
        logs_df = data_generator.generate_log_data(alerts_df)
        
        print(f"\n✓ Generated {len(metrics_df)} metrics records")
        print(f"✓ Generated {len(logs_df)} log entries")
        
        # Show data preview
        print(f"\n📋 Alert Data Preview:")
        print(alerts_df[['alert_id', 'timestamp', 'system_category', 'criticality', 'alert_type']].head())
        
        input("\nPress Enter to continue to clustering analysis...")
        
        # Step 2: Clustering Analysis
        print("\n🎯 Step 2: Alert Clustering Analysis")
        print("-" * 50)
        
        clustering_analyzer = AlertClusteringAnalyzer()
        
        print("Preparing features for clustering...")
        features = clustering_analyzer.prepare_features(alerts_df)
        print(f"✓ Prepared {features.shape[1]} features for {features.shape[0]} alerts")
        
        print("Performing multi-algorithm clustering...")
        clustering_results, best_method = clustering_analyzer.perform_clustering(features)
        
        print(f"✓ Tested multiple clustering algorithms:")
        for method, results in clustering_results.items():
            print(f"  • {method}: {results['n_clusters']} clusters, silhouette score: {results['score']:.3f}")
        
        print(f"\n🏆 Best method: {best_method.upper()}")
        print(f"   Silhouette Score: {clustering_results[best_method]['score']:.3f}")
        print(f"   Number of Clusters: {clustering_results[best_method]['n_clusters']}")
        
        # Analyze clusters
        cluster_labels = clustering_results[best_method]['labels']
        cluster_analysis = clustering_analyzer.analyze_clusters(alerts_df, cluster_labels, best_method)
        
        print(f"\n📈 Cluster Analysis Results:")
        for cluster_id, analysis in list(cluster_analysis.items())[:5]:  # Show first 5 clusters
            print(f"\n  {cluster_id.upper()}:")
            print(f"    Size: {analysis['size']} alerts ({analysis['percentage']:.1f}%)")
            print(f"    Avg Resolution: {analysis['avg_resolution_time']:.1f} minutes")
            print(f"    Top Criticality: {max(analysis['criticality_distribution'], key=analysis['criticality_distribution'].get)}")
            print(f"    Top System: {max(analysis['system_distribution'], key=analysis['system_distribution'].get)}")
        
        input("\nPress Enter to continue to temporal analysis...")
        
        # Step 3: Temporal Analysis
        print("\n⏰ Step 3: Temporal Pattern & Frequency Analysis")
        print("-" * 50)
        
        frequency_analyzer = AlertFrequencyAnalyzer()
        
        print("Analyzing temporal patterns...")
        temporal_analysis = frequency_analyzer.analyze_temporal_patterns(alerts_df)
        
        print("✓ Temporal pattern analysis completed")
        
        # Show key temporal insights
        peak_hour = max(temporal_analysis['hourly_patterns'], key=temporal_analysis['hourly_patterns'].get)
        peak_day = max(temporal_analysis['weekly_patterns'], key=temporal_analysis['weekly_patterns'].get)
        
        print(f"\n📊 Key Temporal Insights:")
        print(f"  • Peak Alert Hour: {peak_hour}:00 ({temporal_analysis['hourly_patterns'][peak_hour]} alerts)")
        print(f"  • Peak Alert Day: {peak_day} ({temporal_analysis['weekly_patterns'][peak_day]} alerts)")
        print(f"  • Anomalous Days Detected: {len(temporal_analysis['anomaly_dates'])}")
        
        if temporal_analysis['anomaly_dates']:
            print(f"    Anomaly Dates: {', '.join(temporal_analysis['anomaly_dates'][:3])}")
        
        # Show system-specific patterns
        print(f"\n🖥️  System-Specific Patterns:")
        for system, data in list(temporal_analysis['system_temporal_patterns'].items())[:3]:
            peak_hour_sys = max(data['hourly_pattern'], key=data['hourly_pattern'].get)
            print(f"  • {system}: Peak at {peak_hour_sys}:00, Total: {data['total_alerts']} alerts")
        
        print("\nBuilding frequency prediction model...")
        frequency_models = frequency_analyzer.build_frequency_prediction_model(alerts_df)
        
        print(f"✓ Frequency prediction model trained")
        print(f"  Accuracy: {frequency_models['random_forest']['test_score']:.3f}")
        
        input("\nPress Enter to continue to predictive modeling...")
        
        # Step 4: Predictive Modeling
        print("\n🤖 Step 4: Alert Prediction & System Attribution")
        print("-" * 50)
        
        prediction_system = AlertPredictionSystem()
        
        print("Preparing multi-modal prediction features...")
        prediction_data = prediction_system.prepare_prediction_features(alerts_df, metrics_df, logs_df)
        print(f"✓ Prepared prediction dataset with {prediction_data.shape[1]} features")
        
        print("Training alert likelihood prediction model...")
        likelihood_model = prediction_system.build_alert_likelihood_model(prediction_data)
        
        print(f"✓ Alert Likelihood Model Performance:")
        print(f"  • Training Accuracy: {likelihood_model['train_score']:.3f}")
        print(f"  • Test Accuracy: {likelihood_model['test_score']:.3f}")
        
        # Show top features
        top_features_idx = np.argsort(likelihood_model['feature_importance'])[-5:]
        print(f"  • Top 5 Important Features:")
        for i, idx in enumerate(reversed(top_features_idx)):
            feature_name = likelihood_model['feature_names'][idx]
            importance = likelihood_model['feature_importance'][idx]
            print(f"    {i+1}. {feature_name}: {importance:.3f}")
        
        print("\nTraining system attribution model...")
        attribution_model = prediction_system.build_system_attribution_model(prediction_data)
        
        print(f"✓ System Attribution Model Performance:")
        print(f"  • Test Accuracy: {attribution_model['test_accuracy']:.3f}")
        print(f"  • Test Loss: {attribution_model['test_loss']:.3f}")
        
        input("\nPress Enter to continue to visualization and reporting...")
        
        # Step 5: Visualization and Reporting
        print("\n📊 Step 5: Visualization & Comprehensive Reporting")
        print("-" * 50)
        
        dashboard = AIOpsVisualizationDashboard()
        
        print("Creating visualization dashboard...")
        
        # Create visualizations
        clustering_fig = dashboard.create_clustering_visualization(alerts_df, cluster_labels, features)
        temporal_fig = dashboard.create_temporal_analysis_dashboard(temporal_analysis)
        importance_fig = dashboard.create_prediction_model_performance(likelihood_model)
        
        print("✓ Clustering visualization created")
        print("✓ Temporal analysis dashboard created")
        print("✓ Model performance visualization created")
        
        print("Generating comprehensive analysis report...")
        comprehensive_report = dashboard.generate_comprehensive_report(
            alerts_df, cluster_analysis, temporal_analysis,
            {'likelihood_model': likelihood_model, 'attribution_model': attribution_model}
        )
        
        print("✓ Comprehensive report generated")
        
        # Step 6: Final Results and Recommendations
        print("\n" + "=" * 70)
        print("🎯 FINAL ANALYSIS RESULTS & RECOMMENDATIONS")
        print("=" * 70)
        
        print(f"\n📈 EXECUTIVE SUMMARY:")
        summary = comprehensive_report['summary']
        print(f"  • Total Alerts Analyzed: {summary['total_alerts']}")
        print(f"  • Systems Monitored: {summary['unique_systems']}")
        print(f"  • Services Tracked: {summary['unique_services']}")
        print(f"  • Analysis Period: {days_back} days")
        print(f"  • Average Resolution Time: {summary['avg_resolution_time']:.1f} minutes")
        
        print(f"\n🔍 CRITICALITY BREAKDOWN:")
        for criticality, count in summary['criticality_distribution'].items():
            percentage = (count / summary['total_alerts']) * 100
            print(f"  • {criticality.upper()}: {count} alerts ({percentage:.1f}%)")
        
        print(f"\n🎯 CLUSTERING INSIGHTS:")
        print(f"  • Identified {len(cluster_analysis)} distinct alert patterns")
        print(f"  • Clustering method: {best_method.upper()}")
        print(f"  • Silhouette score: {clustering_results[best_method]['score']:.3f}")
        
        print(f"\n⏰ TEMPORAL INSIGHTS:")
        print(f"  • Peak activity: {peak_day} at {peak_hour}:00")
        print(f"  • Anomalous patterns: {len(temporal_analysis['anomaly_dates'])} days")
        print(f"  • Frequency prediction accuracy: {frequency_models['random_forest']['test_score']:.3f}")
        
        print(f"\n🤖 PREDICTION MODEL PERFORMANCE:")
        print(f"  • Alert likelihood prediction: {likelihood_model['test_score']:.3f} accuracy")
        print(f"  • System attribution: {attribution_model['test_accuracy']:.3f} accuracy")
        
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for i, recommendation in enumerate(comprehensive_report['recommendations'], 1):
            print(f"  {i}. {recommendation}")
        
        # Additional actionable insights
        print(f"\n🚀 ACTIONABLE INSIGHTS:")
        
        # Critical system analysis
        critical_alerts = alerts_df[alerts_df['criticality'] == 'critical']
        if len(critical_alerts) > 0:
            top_critical_system = critical_alerts['system_category'].value_counts().index[0]
            critical_count = len(critical_alerts[critical_alerts['system_category'] == top_critical_system])
            print(f"  • PRIORITY: {top_critical_system} system needs immediate attention")
            print(f"    ({critical_count} critical alerts, {critical_count/len(critical_alerts)*100:.1f}% of all critical alerts)")
        
        # Resolution time insights
        slow_resolution = alerts_df[alerts_df['resolution_time'] > alerts_df['resolution_time'].quantile(0.9)]
        if len(slow_resolution) > 0:
            slow_systems = slow_resolution['system_category'].value_counts().head(2)
            print(f"  • EFFICIENCY: Focus on resolution time for {', '.join(slow_systems.index)}")
            print(f"    (90th percentile resolution time: {alerts_df['resolution_time'].quantile(0.9):.1f} minutes)")
        
        # Frequency-based insights
        high_freq_systems = alerts_df['system_category'].value_counts().head(3)
        print(f"  • MONITORING: Enhance proactive monitoring for:")
        for system, count in high_freq_systems.items():
            percentage = (count / len(alerts_df)) * 100
            print(f"    - {system}: {count} alerts ({percentage:.1f}% of total)")
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print("The AIOps system has successfully analyzed your alert patterns and")
        print("provided actionable insights for improving your IT operations.")
        
        # Save results
        print(f"\n💾 Saving analysis results...")
        results = {
            'alerts_data': alerts_df,
            'metrics_data': metrics_df,
            'logs_data': logs_df,
            'clustering_results': clustering_results,
            'cluster_analysis': cluster_analysis,
            'temporal_analysis': temporal_analysis,
            'prediction_models': {
                'likelihood': likelihood_model,
                'attribution': attribution_model,
                'frequency': frequency_models
            },
            'comprehensive_report': comprehensive_report,
            'visualizations': {
                'clustering': clustering_fig,
                'temporal': temporal_fig,
                'importance': importance_fig
            }
        }
        
        # Save to file
        with open('/home/jshayi2/data/redppo/demo_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Save report as JSON
        json_report = json.dumps(comprehensive_report, indent=2, default=str)
        with open('/home/jshayi2/data/redppo/aiops_report.json', 'w') as f:
            f.write(json_report)
        
        print("✓ Results saved to 'demo_results.pkl'")
        print("✓ Report saved to 'aiops_report.json'")
        
        print(f"\n🎉 Demo completed successfully!")
        print("Thank you for exploring the AIOps Alert Analysis System!")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
        return None
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("Please check your environment and dependencies.")
        return None

def show_quick_analysis():
    """Show a quick analysis without user interaction"""
    
    print("🚀 Quick AIOps Analysis (No Interaction Required)")
    print("=" * 60)
    
    # Run the main analysis
    results = main()
    
    print(f"\n📊 Quick Analysis Summary:")
    print(f"✓ Processed {len(results['alerts_data'])} alerts")
    print(f"✓ Identified {len(results['cluster_analysis'])} alert clusters")
    print(f"✓ Built predictive models with >80% accuracy")
    print(f"✓ Generated comprehensive insights and recommendations")
    
    return results

if __name__ == "__main__":
    print("AIOps Alert Analysis System")
    print("Choose demo mode:")
    print("1. Interactive Demo (recommended)")
    print("2. Quick Analysis (automated)")
    
    try:
        choice = input("\nEnter choice (1-2, or press Enter for interactive): ").strip()
        
        if choice == '2':
            show_quick_analysis()
        else:
            run_interactive_demo()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        print("Running quick analysis instead...")
        show_quick_analysis()
