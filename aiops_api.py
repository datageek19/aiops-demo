"""
Production-Ready AIOps API
==========================

Simple Flask API for real-time alert analysis and prediction.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

# Import our enhanced analyzer
from improved_aiops_solution import EnhancedAIOpsAnalyzer

app = Flask(__name__)

# Global variables for models and analyzer
analyzer = None
models = None
model_last_updated = None

def initialize_models():
    """Initialize or refresh models"""
    global analyzer, models, model_last_updated

    print("🔄 Initializing AIOps models...")
    analyzer = EnhancedAIOpsAnalyzer()

    # Generate training data and train models
    training_data = analyzer.generate_realistic_alerts(1000)
    models = analyzer.enhanced_prediction_models(training_data)
    model_last_updated = datetime.now()

    print("✅ Models initialized successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AIOps Alert Analyzer',
        'model_last_updated': model_last_updated.isoformat() if model_last_updated else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze/alert', methods=['POST'])
def analyze_single_alert():
    """Analyze a single incoming alert"""
    try:
        alert_data = request.json

        # Validate required fields
        required_fields = ['system', 'alert_type', 'timestamp']
        for field in required_fields:
            if field not in alert_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Enrich alert data with computed features
        timestamp = datetime.fromisoformat(alert_data['timestamp'].replace('Z', '+00:00'))

        enriched_alert = {
            'system': alert_data['system'],
            'alert_type': alert_data['alert_type'],
            'severity_score': alert_data.get('severity_score', 0.5),
            'affected_users': alert_data.get('affected_users', 10),
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_business_hours': 1 if 9 <= timestamp.hour <= 17 and timestamp.weekday() < 5 else 0,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'business_impact': alert_data.get('business_impact', 50)
        }

        # Get real-time scoring
        scores = analyzer.real_time_alert_scorer(enriched_alert, models)

        # Add metadata
        response = {
            'alert_id': alert_data.get('alert_id', f"ALT-{int(datetime.now().timestamp())}"),
            'analysis_timestamp': datetime.now().isoformat(),
            'risk_assessment': scores,
            'enriched_data': enriched_alert,
            'model_version': model_last_updated.isoformat() if model_last_updated else None
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch_alerts():
    """Analyze multiple alerts"""
    try:
        data = request.json
        alerts = data.get('alerts', [])

        if not alerts:
            return jsonify({'error': 'No alerts provided'}), 400

        if len(alerts) > 100:
            return jsonify({'error': 'Batch size too large (max 100)'}), 400

        results = []
        for alert in alerts:
            # Process each alert
            timestamp = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))

            enriched_alert = {
                'system': alert['system'],
                'alert_type': alert['alert_type'],
                'severity_score': alert.get('severity_score', 0.5),
                'affected_users': alert.get('affected_users', 10),
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'is_business_hours': 1 if 9 <= timestamp.hour <= 17 and timestamp.weekday() < 5 else 0,
                'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
                'business_impact': alert.get('business_impact', 50)
            }

            scores = analyzer.real_time_alert_scorer(enriched_alert, models)

            results.append({
                'alert_id': alert.get('alert_id', f"ALT-{len(results)+1}"),
                'risk_assessment': scores,
                'enriched_data': enriched_alert
            })

        # Batch-level insights
        high_risk_count = sum(1 for r in results if r['risk_assessment']['risk_level'] in ['CRITICAL', 'HIGH'])
        avg_risk_score = np.mean([r['risk_assessment']['composite_risk_score'] for r in results])

        response = {
            'analysis_timestamp': datetime.now().isoformat(),
            'batch_summary': {
                'total_alerts': len(results),
                'high_risk_alerts': high_risk_count,
                'average_risk_score': float(avg_risk_score),
                'risk_distribution': {
                    level: sum(1 for r in results if r['risk_assessment']['risk_level'] == level)
                    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
                }
            },
            'individual_results': results
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/cascade', methods=['POST'])
def predict_cascade():
    """Predict if an alert will cause cascading failures"""
    try:
        alert_data = request.json

        # Simplified cascade prediction
        cascade_indicators = {
            'database': 0.8,
            'api_gateway': 0.7,
            'web_server': 0.5,
            'cache': 0.4,
            'monitoring': 0.2
        }

        system = alert_data.get('system', 'unknown')
        severity = alert_data.get('severity_score', 0.5)
        business_hours = alert_data.get('is_business_hours', 0)

        base_probability = cascade_indicators.get(system, 0.3)
        adjusted_probability = base_probability * severity * (1.2 if business_hours else 0.8)
        adjusted_probability = min(adjusted_probability, 1.0)

        response = {
            'cascade_probability': float(adjusted_probability),
            'risk_level': 'HIGH' if adjusted_probability > 0.7 else 'MEDIUM' if adjusted_probability > 0.4 else 'LOW',
            'affected_systems': ['web_server', 'cache'] if system == 'database' else [],
            'estimated_cascade_time': '5-30 minutes' if adjusted_probability > 0.5 else 'N/A',
            'prevention_actions': [
                f'Monitor {system} dependencies closely',
                'Prepare rollback procedures',
                'Alert downstream system owners'
            ] if adjusted_probability > 0.5 else ['Continue normal monitoring']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/insights/system/<system_name>', methods=['GET'])
def get_system_insights(system_name):
    """Get insights for a specific system"""
    try:
        # Mock system insights (in production, this would query historical data)
        system_insights = {
            'database': {
                'alert_frequency': '15 alerts/day',
                'avg_severity': 0.7,
                'common_alert_types': ['cpu_high', 'memory_leak', 'connection_pool_exhausted'],
                'peak_hours': [10, 14, 16],
                'cascade_probability': 0.8,
                'recommendations': [
                    'Implement connection pooling optimization',
                    'Add CPU monitoring alerts',
                    'Schedule maintenance during off-hours'
                ]
            },
            'web_server': {
                'alert_frequency': '8 alerts/day',
                'avg_severity': 0.5,
                'common_alert_types': ['cpu_high', 'network_timeout'],
                'peak_hours': [12, 15, 18],
                'cascade_probability': 0.5,
                'recommendations': [
                    'Optimize static content caching',
                    'Review load balancer configuration'
                ]
            }
        }

        insights = system_insights.get(system_name, {
            'alert_frequency': 'Unknown',
            'avg_severity': 0.5,
            'common_alert_types': ['Unknown'],
            'peak_hours': [],
            'cascade_probability': 0.3,
            'recommendations': ['Collect more data for this system']
        })

        response = {
            'system': system_name,
            'insights': insights,
            'last_updated': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/refresh', methods=['POST'])
def refresh_models():
    """Refresh/retrain models"""
    try:
        initialize_models()
        return jsonify({
            'status': 'success',
            'message': 'Models refreshed successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/docs', methods=['GET'])
def api_documentation():
    """API documentation"""
    docs = {
        'service': 'AIOps Alert Analysis API',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Health check',
            'POST /analyze/alert': 'Analyze single alert',
            'POST /analyze/batch': 'Analyze multiple alerts',
            'POST /predict/cascade': 'Predict cascade probability',
            'GET /insights/system/<name>': 'Get system-specific insights',
            'POST /models/refresh': 'Refresh ML models',
            'GET /docs': 'This documentation'
        },
        'alert_format': {
            'required': ['system', 'alert_type', 'timestamp'],
            'optional': ['severity_score', 'affected_users', 'business_impact', 'alert_id'],
            'example': {
                'alert_id': 'ALT-12345',
                'system': 'database',
                'alert_type': 'cpu_high',
                'timestamp': '2024-01-15T14:30:00Z',
                'severity_score': 0.8,
                'affected_users': 500,
                'business_impact': 75
            }
        }
    }

    return jsonify(docs)

if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()

    print("🚀 Starting AIOps API server...")
    print("📖 API Documentation: http://localhost:5000/docs")
    print("💓 Health Check: http://localhost:5000/health")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
