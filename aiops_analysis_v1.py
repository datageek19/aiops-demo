
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Simple imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json

print('🚀 Simple AIOps Alert Analysis Solution')
print('=' * 50)

# 1. Generate Simple Synthetic Data
def generate_simple_alerts(n_alerts=500):
    np.random.seed(42)

    systems = ['web_server', 'database', 'api_gateway', 'cache', 'monitoring']
    alert_types = ['cpu_high', 'memory_leak', 'disk_full', 'network_timeout', 'service_down']
    criticalities = ['low', 'medium', 'high', 'critical']

    data = []
    for i in range(n_alerts):
        # Generate realistic patterns
        system = np.random.choice(systems)
        alert_type = np.random.choice(alert_types)

        # Business hours bias
        if np.random.random() < 0.7:
            hour = np.random.randint(9, 18)
        else:
            hour = np.random.randint(0, 24)

        # Criticality based on system and time
        if system in ['database', 'api_gateway'] and hour >= 9 and hour <= 17:
            criticality = np.random.choice(criticalities, p=[0.1, 0.3, 0.4, 0.2])
        else:
            criticality = np.random.choice(criticalities, p=[0.3, 0.4, 0.2, 0.1])

        timestamp = datetime.now() - timedelta(days=30) + timedelta(hours=np.random.randint(0, 720))

        data.append({
            'alert_id': f'ALT-{i+1:04d}',
            'timestamp': timestamp,
            'system': system,
            'alert_type': alert_type,
            'criticality': criticality,
            'hour': hour,
            'day_of_week': timestamp.weekday(),
            'is_business_hours': 1 if 9 <= hour <= 17 else 0,
            'resolution_time': np.random.randint(10, 480)  # minutes
        })

    return pd.DataFrame(data)

# Generate data
alerts_df = generate_simple_alerts(500)
print(f'✅ Generated {len(alerts_df)} alerts')
print('Sample data:')
print(alerts_df.head())

print('\n' + '=' * 50)
print('📊 ALERT ANALYSIS RESULTS')
print('=' * 50)

# 2. Basic Statistics
print('\n1. Basic Alert Statistics:')
print(f'   Total alerts: {len(alerts_df)}')
print(f'   Systems: {alerts_df[\"system\"].nunique()}')
print(f'   Alert types: {alerts_df[\"alert_type\"].nunique()}')
print(f'   Time span: {(alerts_df[\"timestamp\"].max() - alerts_df[\"timestamp\"].min()).days} days')

# 3. System Correlation Analysis
print('\n2. System-Alert Correlation:')
system_alert_crosstab = pd.crosstab(alerts_df['system'], alerts_df['alert_type'])
print(system_alert_crosstab)

# Find most correlated system-alert combinations
correlations = []
for system in alerts_df['system'].unique():
    for alert_type in alerts_df['alert_type'].unique():
        count = len(alerts_df[(alerts_df['system'] == system) & (alerts_df['alert_type'] == alert_type)])
        if count > 0:
            correlations.append({
                'system': system,
                'alert_type': alert_type,
                'count': count,
                'percentage': count / len(alerts_df) * 100
            })

correlations_df = pd.DataFrame(correlations).sort_values('count', ascending=False)
print('\nTop System-Alert Combinations:')
print(correlations_df.head(10))

# 4. Frequency Analysis
print('\n3. Frequency Patterns:')
hourly_freq = alerts_df.groupby('hour').size()
daily_freq = alerts_df.groupby('day_of_week').size()
system_freq = alerts_df['system'].value_counts()

print(f'Peak hour: {hourly_freq.idxmax()}:00 ({hourly_freq.max()} alerts)')
print(f'Peak day: {daily_freq.idxmax()} ({daily_freq.max()} alerts)')
print(f'Most active system: {system_freq.index[0]} ({system_freq.iloc[0]} alerts)')

# 5. Dependency Analysis (Simple Co-occurrence)
print('\n4. System Dependencies (Co-occurrence within 1 hour):')
dependencies = {}
for i, alert in alerts_df.iterrows():
    current_time = alert['timestamp']
    current_system = alert['system']

    # Find alerts within 1 hour
    nearby_alerts = alerts_df[
        (abs((alerts_df['timestamp'] - current_time).dt.total_seconds()) <= 3600) &
        (alerts_df['system'] != current_system)
    ]

    for _, nearby in nearby_alerts.iterrows():
        pair = tuple(sorted([current_system, nearby['system']]))
        dependencies[pair] = dependencies.get(pair, 0) + 1

# Show top dependencies
dep_sorted = sorted(dependencies.items(), key=lambda x: x[1], reverse=True)[:5]
print('Top System Dependencies:')
for (sys1, sys2), count in dep_sorted:
    print(f'   {sys1} ↔ {sys2}: {count} co-occurrences')

# 6. Simple Alert Likelihood Prediction
print('\n5. Alert Likelihood Prediction:')

# Prepare features for ML
feature_cols = ['hour', 'day_of_week', 'is_business_hours']
X = alerts_df[feature_cols]

# Encode system and alert_type
le_system = LabelEncoder()
le_alert_type = LabelEncoder()
X['system_encoded'] = le_system.fit_transform(alerts_df['system'])
X['alert_type_encoded'] = le_alert_type.fit_transform(alerts_df['alert_type'])

# Target: predict if alert will be critical/high
y = (alerts_df['criticality'].isin(['critical', 'high'])).astype(int)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'High-Impact Alert Prediction Accuracy: {accuracy:.3f}')

# Feature importance
feature_importance = dict(zip(X.columns, model.feature_importances_))
print('Feature Importance:')
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f'   {feature}: {importance:.3f}')

# 7. Predictions for Next Alerts
print('\n6. Likelihood Predictions:')

# Predict likelihood for different scenarios
scenarios = [
    {'hour': 14, 'day_of_week': 1, 'is_business_hours': 1, 'system': 'database', 'alert_type': 'cpu_high'},
    {'hour': 2, 'day_of_week': 5, 'is_business_hours': 0, 'system': 'web_server', 'alert_type': 'memory_leak'},
    {'hour': 10, 'day_of_week': 2, 'is_business_hours': 1, 'system': 'api_gateway', 'alert_type': 'service_down'}
]

for i, scenario in enumerate(scenarios):
    # Encode scenario
    scenario_encoded = [
        scenario['hour'],
        scenario['day_of_week'],
        scenario['is_business_hours'],
        le_system.transform([scenario['system']])[0],
        le_alert_type.transform([scenario['alert_type']])[0]
    ]

    probability = model.predict_proba([scenario_encoded])[0][1]
    print(f'Scenario {i+1}: {scenario[\"system\"]} {scenario[\"alert_type\"]} at {scenario[\"hour\"]}:00')
    print(f'   High-impact probability: {probability:.3f} ({\"HIGH\" if probability > 0.5 else \"LOW\"} risk)')

# 8. Simple Recommendations
print('\n7. Actionable Recommendations:')

# System with most critical alerts
critical_by_system = alerts_df[alerts_df['criticality'] == 'critical']['system'].value_counts()
if len(critical_by_system) > 0:
    print(f'• Focus on {critical_by_system.index[0]} system - generates {critical_by_system.iloc[0]} critical alerts')

# Peak time recommendation
print(f'• Schedule maintenance outside peak hour ({hourly_freq.idxmax()}:00)')

# Dependency recommendation
if dep_sorted:
    top_dep = dep_sorted[0]
    print(f'• Monitor {top_dep[0][0]} and {top_dep[0][1]} systems together - they often fail together')

print('\n✅ Analysis Complete!')
print('This solution provides:')
print('• Alert correlation analysis between systems and alert types')
print('• Frequency patterns showing when alerts occur most')
print('• System dependency detection based on co-occurrence')
print('• Machine learning model to predict high-impact alerts')
print('• Actionable recommendations for operations teams')


# ++++++++++++++++++++++++++++++++++++++++++
# more compact implementation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SIMPLE DATA GENERATION
def create_alert_data(n_alerts=1000):
    """Generate realistic alert data"""
    np.random.seed(42)

    systems = ['web_server', 'database', 'api_gateway', 'cache', 'monitoring']
    alert_types = ['cpu_high', 'memory_leak', 'disk_full', 'network_timeout', 'service_down']
    criticalities = ['low', 'medium', 'high', 'critical']

    alerts = []
    for i in range(n_alerts):
        # Create realistic patterns
        system = np.random.choice(systems)
        alert_type = np.random.choice(alert_types)

        # Business hours have more alerts
        if np.random.random() < 0.7:
            hour = np.random.randint(9, 18)
        else:
            hour = np.random.randint(0, 24)

        # Database and API issues are more critical during business hours
        if system in ['database', 'api_gateway'] and 9 <= hour <= 17:
            criticality = np.random.choice(criticalities, p=[0.1, 0.2, 0.4, 0.3])
        else:
            criticality = np.random.choice(criticalities, p=[0.4, 0.3, 0.2, 0.1])

        timestamp = datetime.now() - timedelta(days=30) + timedelta(hours=np.random.randint(0, 720))

        alerts.append({
            'alert_id': f'ALT-{i+1:04d}',
            'timestamp': timestamp,
            'system': system,
            'alert_type': alert_type,
            'criticality': criticality,
            'hour': hour,
            'day_of_week': timestamp.weekday(),
            'is_business_hours': 1 if 9 <= hour <= 17 else 0
        })

    return pd.DataFrame(alerts)

# 2. CORRELATION ANALYSIS
def analyze_correlations(df):
    """Find how alerts are correlated with systems"""

    print("🔍 ALERT CORRELATION ANALYSIS")
    print("=" * 40)

    # System vs Alert Type correlation
    correlation_matrix = pd.crosstab(df['system'], df['alert_type'])
    print("\nSystem-Alert Type Matrix:")
    print(correlation_matrix)

    # Find strongest correlations
    correlations = []
    for system in df['system'].unique():
        for alert_type in df['alert_type'].unique():
            count = len(df[(df['system'] == system) & (df['alert_type'] == alert_type)])
            total_system = len(df[df['system'] == system])
            if total_system > 0:
                percentage = count / total_system * 100
                correlations.append({
                    'system': system,
                    'alert_type': alert_type,
                    'count': count,
                    'percentage': percentage
                })

    corr_df = pd.DataFrame(correlations).sort_values('percentage', ascending=False)
    print("\nTop System-Alert Correlations:")
    print(corr_df.head(10))

    return corr_df

# 3. FREQUENCY ANALYSIS
def analyze_frequency(df):
    """Analyze how often alerts occur"""

    print("\n⏰ FREQUENCY ANALYSIS")
    print("=" * 40)

    # Hourly patterns
    hourly_freq = df.groupby('hour').size()
    print(f"Peak hour: {hourly_freq.idxmax()}:00 with {hourly_freq.max()} alerts")

    # Daily patterns
    daily_freq = df.groupby('day_of_week').size()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"Peak day: {days[daily_freq.idxmax()]} with {daily_freq.max()} alerts")

    # System frequency
    system_freq = df['system'].value_counts()
    print(f"Most active system: {system_freq.index[0]} ({system_freq.iloc[0]} alerts)")

    # Business hours vs off-hours
    business_alerts = len(df[df['is_business_hours'] == 1])
    total_alerts = len(df)
    print(f"Business hours alerts: {business_alerts} ({business_alerts/total_alerts*100:.1f}%)")

    return {
        'hourly': hourly_freq,
        'daily': daily_freq,
        'system': system_freq
    }

# 4. DEPENDENCY ANALYSIS
def analyze_dependencies(df):
    """Find which systems depend on each other"""

    print("\n🔗 SYSTEM DEPENDENCY ANALYSIS")
    print("=" * 40)

    # Find alerts that happen close together (within 1 hour)
    dependencies = {}

    for i, alert in df.iterrows():
        current_time = alert['timestamp']
        current_system = alert['system']

        # Find nearby alerts (within 1 hour)
        nearby = df[
            (abs((df['timestamp'] - current_time).dt.total_seconds()) <= 3600) &
            (df['system'] != current_system)
        ]

        for _, nearby_alert in nearby.iterrows():
            pair = tuple(sorted([current_system, nearby_alert['system']]))
            dependencies[pair] = dependencies.get(pair, 0) + 1

    # Sort by frequency
    dep_sorted = sorted(dependencies.items(), key=lambda x: x[1], reverse=True)

    print("System Dependencies (co-occurring within 1 hour):")
    for (sys1, sys2), count in dep_sorted[:10]:
        print(f"  {sys1} ↔ {sys2}: {count} times")

    return dep_sorted

# 5. ALERT LIKELIHOOD PREDICTION
def predict_alert_likelihood(df):
    """Build model to predict high-impact alerts"""

    print("\n🤖 ALERT LIKELIHOOD PREDICTION")
    print("=" * 40)

    # Prepare features
    features = ['hour', 'day_of_week', 'is_business_hours']
    X = df[features].copy()

    # Encode categorical features
    le_system = LabelEncoder()
    le_alert_type = LabelEncoder()

    X['system_encoded'] = le_system.fit_transform(df['system'])
    X['alert_type_encoded'] = le_alert_type.fit_transform(df['alert_type'])

    # Target: predict critical/high alerts
    y = df['criticality'].isin(['critical', 'high']).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.3f}")

    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    print("\nFeature Importance:")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {imp:.3f}")

    return model, le_system, le_alert_type

# 6. MAIN EXECUTION
def main():
    print("🚀 Simple AIOps Alert Analysis")
    print("=" * 50)

    # Generate data
    df = create_alert_data(1000)
    print(f"✅ Generated {len(df)} alerts")

    # Run analyses
    correlations = analyze_correlations(df)
    frequencies = analyze_frequency(df)
    dependencies = analyze_dependencies(df)
    model, le_system, le_alert_type = predict_alert_likelihood(df)

    # Summary recommendations
    print("\n💡 KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 40)

    # Most problematic system
    critical_by_system = df[df['criticality'] == 'critical']['system'].value_counts()
    if len(critical_by_system) > 0:
        print(f"1. Focus on '{critical_by_system.index[0]}' - generates most critical alerts")

    # Peak time
    peak_hour = frequencies['hourly'].idxmax()
    print(f"2. Peak alert time is {peak_hour}:00 - avoid maintenance then")

    # System dependencies
    if dependencies:
        top_dep = dependencies[0]
        print(f"3. '{top_dep[0][0]}' and '{top_dep[0][1]}' often fail together - monitor both")

    # Business impact
    business_ratio = len(df[df['is_business_hours'] == 1]) / len(df)
    print(f"4. {business_ratio*100:.1f}% of alerts occur during business hours")

    print("\n✅ Analysis complete!")

    return df, model, correlations, frequencies, dependencies

# Run the analysis
if __name__ == "__main__":
    alerts_df, prediction_model, correlations, frequencies, dependencies = main()

# ++++++++++++++++++++++++++++++++++++++++++

# *******************************************************
#  sample graph data

# *******************************************************
[{'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'ocpe-storecloudconfig',
  'source_node_id': '0',
  'source_properties': '{\n'
                       '  tenantId: aks-res-clusters,\n'
                       '  environment: qa2,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: ocpe-storecloudconfig,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: ocpe-qa2\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'epe_acceptance',
  'target_node_id': '2348',
  'target_properties': '{\n'
                       '  tenantId: aks-res-clusters,\n'
                       '  connection_string: [],\n'
                       '  environment: qa2,\n'
                       '  updated_at: 2025-09-23T11:55:08.516Z,\n'
                       '  db_system: mongodb,\n'
                       '  name: epe_acceptance,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:18.438Z,\n'
                       '  type: database,\n'
                       '  namespace: ocpe-qa2\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'ocmo-3pmtoorchestrator',
  'source_node_id': '1',
  'source_properties': '{\n'
                       '  tenantId: aks-res-clusters,\n'
                       '  environment: perf1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: ocmo-3pmtoorchestrator,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: ocmo-perf1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'order_info',
  'target_node_id': '2356',
  'target_properties': '{\n'
                       '  tenantId: aks-res-clusters,\n'
                       '  connection_string: [],\n'
                       '  environment: perf1,\n'
                       '  updated_at: 2025-09-23T11:55:08.516Z,\n'
                       '  db_system: mongodb,\n'
                       '  name: order_info,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:18.438Z,\n'
                       '  type: database,\n'
                       '  namespace: ocmo-perf1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'menfpt-calendar-service',
  'source_node_id': '2',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: menfpt-calendar-service,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: menfpt-qa1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'menfpt_qa1',
  'target_node_id': '3',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mssql://menfpt-nonprod-qa1-westus-sqlsrv-01.database.windows.net:1433],\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-29T11:10:29.186Z,\n'
                       '  db_system: microsoft,\n'
                       '  name: menfpt_qa1,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: database,\n'
                       '  namespace: menfpt-qa1\n'
                       '}'},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Service',
  'source_name': 'menfpt_qa1',
  'source_node_id': '3',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mssql://menfpt-nonprod-qa1-westus-sqlsrv-01.database.windows.net:1433],\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-29T11:10:29.186Z,\n'
                       '  db_system: microsoft,\n'
                       '  name: menfpt_qa1,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: database,\n'
                       '  namespace: menfpt-qa1\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': '{\n'
                             '  request_protocol: http,\n'
                             '  updated_at: 2025-09-25T18:36:07.981Z,\n'
                             '  created_at: 2025-09-25T18:36:07.981Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'menfpt-forecast-service',
  'source_node_id': '4',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: menfpt-forecast-service,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: menfpt-qa1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'menfpt-calendar-service',
  'target_node_id': '2',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: menfpt-calendar-service,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: menfpt-qa1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'menfpt-forecast-service',
  'source_node_id': '4',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: menfpt-forecast-service,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: menfpt-qa1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'menfpt_qa1',
  'target_node_id': '3',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mssql://menfpt-nonprod-qa1-westus-sqlsrv-01.database.windows.net:1433],\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-29T11:10:29.186Z,\n'
                       '  db_system: microsoft,\n'
                       '  name: menfpt_qa1,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: database,\n'
                       '  namespace: menfpt-qa1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'mebp-comp-audit-api',
  'source_node_id': '5',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: perf1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: mebp-comp-audit-api,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: mebp-perf1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'UPP_PERF',
  'target_node_id': '3206',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mongodb://pl-0-westus-azure.27pnw.mongodb.net:1025],\n'
                       '  environment: perf1,\n'
                       '  updated_at: 2025-09-27T09:19:17.984Z,\n'
                       '  db_system: mongodb,\n'
                       '  name: UPP_PERF,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:19.450Z,\n'
                       '  type: database,\n'
                       '  namespace: mebp-perf1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'mebp-item-hierarchy-api',
  'source_node_id': '6',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: mebp-item-hierarchy-api,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: mebp-qa1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'UPP_QA',
  'target_node_id': '3213',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mongodb://pl-0-westus-azure.pcdi3.mongodb.net:1026],\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-23T11:55:08.516Z,\n'
                       '  db_system: mongodb,\n'
                       '  name: UPP_QA,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:19.450Z,\n'
                       '  type: database,\n'
                       '  namespace: mebp-qa1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'mebp-pricegroupitem-cost-api',
  'source_node_id': '7',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: stage,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: mebp-pricegroupitem-cost-api,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: mebp-stage\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'UPP_UAT',
  'target_node_id': '3222',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mongodb://pl-0-westus-azure.pcdi3.mongodb.net:1026],\n'
                       '  environment: stage,\n'
                       '  updated_at: 2025-09-23T11:55:08.516Z,\n'
                       '  db_system: mongodb,\n'
                       '  name: UPP_UAT,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:19.450Z,\n'
                       '  type: database,\n'
                       '  namespace: mebp-stage\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'meitem-master',
  'source_node_id': '8',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: meitem-master,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: meitem-qa1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'meitem-common',
  'target_node_id': '3239',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mongodb://pl-0-westus-azure.derni.mongodb.net:1025, '
                       'mongodb://pl-0-westus-azure.w7dgd.mongodb.net:1025, '
                       'mongodb://pl-0-westus-azure.16hz5.mongodb.net:1026],\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-26T21:51:06.592Z,\n'
                       '  db_system: mongodb,\n'
                       '  name: meitem-common,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:19.450Z,\n'
                       '  type: database,\n'
                       '  namespace: meitem-qa1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'meitem-master',
  'source_node_id': '8',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: meitem-master,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: meitem-qa1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'meitem-qa',
  'target_node_id': '3258',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  connection_string: '
                       '[mongodb://pl-0-westus-azure.derni.mongodb.net:1025],\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-23T11:55:09.710Z,\n'
                       '  db_system: mongodb,\n'
                       '  name: meitem-qa,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:19.796Z,\n'
                       '  type: database,\n'
                       '  namespace: meitem-qa1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  request_protocol: http,\n'
                             '  updated_at: 2025-09-16T11:06:48.926Z,\n'
                             '  created_at: 2025-09-16T11:06:48.926Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'meitem-master',
  'source_node_id': '8',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: meitem-master,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: meitem-qa1\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'mevend-user-profile',
  'target_node_id': '3389',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: qa1,\n'
                       '  updated_at: 2025-09-04T11:12:19.796Z,\n'
                       '  name: mevend-user-profile,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:12:19.796Z,\n'
                       '  type: service,\n'
                       '  namespace: meitem-qa1\n'
                       '}'},
 {'relationship_properties': '{\n'
                             '  request_protocol: http,\n'
                             '  updated_at: 2025-09-04T11:18:18.633Z,\n'
                             '  created_at: 2025-09-04T11:18:18.633Z\n'
                             '}',
  'relationship_type': 'CALLS',
  'source_label': 'Service',
  'source_name': 'meupp-org',
  'source_node_id': '9',
  'source_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: dev,\n'
                       '  updated_at: 2025-09-04T11:18:18.221Z,\n'
                       '  name: meupp-org,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:18:18.221Z,\n'
                       '  type: service,\n'
                       '  namespace: meitem-dev\n'
                       '}',
  'target_label': 'Service',
  'target_name': 'meitem-service',
  'target_node_id': '1812',
  'target_properties': '{\n'
                       '  tenantId: aks-tru-clusters,\n'
                       '  environment: dev,\n'
                       '  updated_at: 2025-09-30T01:42:08.280Z,\n'
                       '  name: meitem-service,\n'
                       '  cluster: ,\n'
                       '  created_at: 2025-09-04T11:09:18.786Z,\n'
                       '  type: service,\n'
                       '  namespace: meitem-dev\n'
                       '}'},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'source_node_id': '10',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'source_node_id': '11',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'source_node_id': '12',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-akssb-nonprod-westus-cluster-01',
  'source_node_id': '13',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'source_node_id': '14',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'source_node_id': '15',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'source_node_id': '16',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'source_node_id': '17',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'source_node_id': '18',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'source_node_id': '19',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'source_node_id': '20',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'source_node_id': '21',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'source_node_id': '22',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': nan,
  'relationship_type': nan,
  'source_label': 'Cluster',
  'source_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'source_node_id': '23',
  'source_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}',
  'target_label': nan,
  'target_name': nan,
  'target_node_id': nan,
  'target_properties': nan},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aais-perf1',
  'source_node_id': '24',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572388,\n'
                       '  createdAt: 1757951421423,\n'
                       '  name: aais-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aais-perf1',
  'source_node_id': '24',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572388,\n'
                       '  createdAt: 1757951421423,\n'
                       '  name: aais-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aapn-perf1',
  'source_node_id': '25',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572467,\n'
                       '  createdAt: 1757951421704,\n'
                       '  name: aapn-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aapn-perf1',
  'source_node_id': '25',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572467,\n'
                       '  createdAt: 1757951421704,\n'
                       '  name: aapn-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'afap-dev',
  'source_node_id': '26',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593057,\n'
                       '  createdAt: 1757951421743,\n'
                       '  name: afap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'akv2k8s',
  'source_node_id': '27',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593087,\n'
                       '  createdAt: 1757951421785,\n'
                       '  name: akv2k8s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'appdynamics',
  'source_node_id': '28',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592863,\n'
                       '  createdAt: 1757951421825,\n'
                       '  name: appdynamics\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'calico-system',
  'source_node_id': '29',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593114,\n'
                       '  createdAt: 1757951422007,\n'
                       '  name: calico-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsm-perf1',
  'source_node_id': '30',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573303,\n'
                       '  createdAt: 1757951422064,\n'
                       '  name: emsm-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsm-perf1',
  'source_node_id': '30',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573303,\n'
                       '  createdAt: 1757951422064,\n'
                       '  name: emsm-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev1',
  'source_node_id': '31',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591390,\n'
                       '  createdAt: 1757951422130,\n'
                       '  name: escf-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-perf1',
  'source_node_id': '32',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593144,\n'
                       '  createdAt: 1757951422169,\n'
                       '  name: esob-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'gatekeeper-system',
  'source_node_id': '33',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593274,\n'
                       '  createdAt: 1757951422207,\n'
                       '  name: gatekeeper-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irol-qa1',
  'source_node_id': '34',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573871,\n'
                       '  createdAt: 1757951422244,\n'
                       '  name: irol-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irol-qa1',
  'source_node_id': '34',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573871,\n'
                       '  createdAt: 1757951422244,\n'
                       '  name: irol-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'istio-system',
  'source_node_id': '35',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593300,\n'
                       '  createdAt: 1757951422289,\n'
                       '  name: istio-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-dev',
  'source_node_id': '36',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579640,\n'
                       '  createdAt: 1757951422326,\n'
                       '  name: itsv-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-dev',
  'source_node_id': '36',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579640,\n'
                       '  createdAt: 1757951422326,\n'
                       '  name: itsv-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-perf1',
  'source_node_id': '37',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590504,\n'
                       '  createdAt: 1757951422364,\n'
                       '  name: itsv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-perf1',
  'source_node_id': '37',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590504,\n'
                       '  createdAt: 1757951422364,\n'
                       '  name: itsv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-perf1',
  'source_node_id': '37',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590504,\n'
                       '  createdAt: 1757951422364,\n'
                       '  name: itsv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-perf1',
  'source_node_id': '37',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590504,\n'
                       '  createdAt: 1757951422364,\n'
                       '  name: itsv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-perf1',
  'source_node_id': '37',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590504,\n'
                       '  createdAt: 1757951422364,\n'
                       '  name: itsv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-perf1',
  'source_node_id': '37',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590504,\n'
                       '  createdAt: 1757951422364,\n'
                       '  name: itsv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-perf1',
  'source_node_id': '37',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590504,\n'
                       '  createdAt: 1757951422364,\n'
                       '  name: itsv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itsv-qa1',
  'source_node_id': '38',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572178,\n'
                       '  createdAt: 1757951422403,\n'
                       '  name: itsv-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'kured',
  'source_node_id': '39',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572205,\n'
                       '  createdAt: 1757951422485,\n'
                       '  name: kured\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-02',
  'target_node_id': '15',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571581,\n'
                       '  createdAt: 1757951421050,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-03',
  'target_node_id': '16',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571607,\n'
                       '  createdAt: 1757951421094,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-04',
  'target_node_id': '17',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571632,\n'
                       '  createdAt: 1757951421136,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-05',
  'target_node_id': '18',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571658,\n'
                       '  createdAt: 1757951421174,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'tigera-operator',
  'source_node_id': '40',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593326,\n'
                       '  createdAt: 1757951422522,\n'
                       '  name: tigera-operator\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'turbo',
  'source_node_id': '41',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575749,\n'
                       '  createdAt: 1757951422559,\n'
                       '  name: turbo\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'turbo',
  'source_node_id': '41',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575749,\n'
                       '  createdAt: 1757951422559,\n'
                       '  name: turbo\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'twistlock',
  'source_node_id': '42',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591987,\n'
                       '  createdAt: 1757951422596,\n'
                       '  name: twistlock\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'twistlock',
  'source_node_id': '42',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591987,\n'
                       '  createdAt: 1757951422596,\n'
                       '  name: twistlock\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'twistlock',
  'source_node_id': '42',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591987,\n'
                       '  createdAt: 1757951422596,\n'
                       '  name: twistlock\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'twistlock',
  'source_node_id': '42',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591987,\n'
                       '  createdAt: 1757951422596,\n'
                       '  name: twistlock\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'twistlock',
  'source_node_id': '42',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591987,\n'
                       '  createdAt: 1757951422596,\n'
                       '  name: twistlock\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'twistlock',
  'source_node_id': '42',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591987,\n'
                       '  createdAt: 1757951422596,\n'
                       '  name: twistlock\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'twistlock',
  'source_node_id': '42',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591987,\n'
                       '  createdAt: 1757951422596,\n'
                       '  name: twistlock\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'velero',
  'source_node_id': '43',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593027,\n'
                       '  createdAt: 1757951422634,\n'
                       '  name: velero\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aabz-dev',
  'source_node_id': '44',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572336,\n'
                       '  createdAt: 1757951422674,\n'
                       '  name: aabz-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aais-dev',
  'source_node_id': '45',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572362,\n'
                       '  createdAt: 1757951422714,\n'
                       '  name: aais-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aais-qa1',
  'source_node_id': '46',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572416,\n'
                       '  createdAt: 1757951422793,\n'
                       '  name: aais-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aapn-dev',
  'source_node_id': '47',
  'source_properties': '{\n'
                       '  updatedAt: 1758632576885,\n'
                       '  createdAt: 1757951422832,\n'
                       '  name: aapn-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aapn-dev',
  'source_node_id': '47',
  'source_properties': '{\n'
                       '  updatedAt: 1758632576885,\n'
                       '  createdAt: 1757951422832,\n'
                       '  name: aapn-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aapn-qa1',
  'source_node_id': '48',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572497,\n'
                       '  createdAt: 1757951422904,\n'
                       '  name: aapn-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'bico-dev',
  'source_node_id': '49',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572601,\n'
                       '  createdAt: 1757951423059,\n'
                       '  name: bico-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dewp-dev',
  'source_node_id': '50',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572653,\n'
                       '  createdAt: 1757951423134,\n'
                       '  name: dewp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dewp-perf1',
  'source_node_id': '51',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572678,\n'
                       '  createdAt: 1757951423171,\n'
                       '  name: dewp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dewp-qa1',
  'source_node_id': '52',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572704,\n'
                       '  createdAt: 1757951423207,\n'
                       '  name: dewp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'doct-dev',
  'source_node_id': '53',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572730,\n'
                       '  createdAt: 1757951423247,\n'
                       '  name: doct-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'doct-perf1',
  'source_node_id': '54',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572758,\n'
                       '  createdAt: 1757951423284,\n'
                       '  name: doct-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'doct-qa1',
  'source_node_id': '55',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572784,\n'
                       '  createdAt: 1757951423321,\n'
                       '  name: doct-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dprd-qa1',
  'source_node_id': '56',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572811,\n'
                       '  createdAt: 1757951423359,\n'
                       '  name: dprd-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxam-dev',
  'source_node_id': '57',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572836,\n'
                       '  createdAt: 1757951423398,\n'
                       '  name: dxam-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxam-qa1',
  'source_node_id': '58',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572863,\n'
                       '  createdAt: 1757951423434,\n'
                       '  name: dxam-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxinaz-dev',
  'source_node_id': '59',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572888,\n'
                       '  createdAt: 1757951423475,\n'
                       '  name: dxinaz-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxinaz-perf1',
  'source_node_id': '60',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572913,\n'
                       '  createdAt: 1757951423513,\n'
                       '  name: dxinaz-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxinaz-qa1',
  'source_node_id': '61',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572939,\n'
                       '  createdAt: 1757951423557,\n'
                       '  name: dxinaz-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxinp-dev',
  'source_node_id': '62',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572965,\n'
                       '  createdAt: 1757951423596,\n'
                       '  name: dxinp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxinp-qa1',
  'source_node_id': '63',
  'source_properties': '{\n'
                       '  updatedAt: 1758632572990,\n'
                       '  createdAt: 1757951423635,\n'
                       '  name: dxinp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxpci-dev',
  'source_node_id': '64',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573016,\n'
                       '  createdAt: 1757951423671,\n'
                       '  name: dxpci-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxpci-qa1',
  'source_node_id': '65',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573042,\n'
                       '  createdAt: 1757951423710,\n'
                       '  name: dxpci-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxpf-dev',
  'source_node_id': '66',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573068,\n'
                       '  createdAt: 1757951423748,\n'
                       '  name: dxpf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxpf-qa1',
  'source_node_id': '67',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573094,\n'
                       '  createdAt: 1757951423786,\n'
                       '  name: dxpf-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxt1-dev',
  'source_node_id': '68',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573119,\n'
                       '  createdAt: 1757951423823,\n'
                       '  name: dxt1-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxt1-qa1',
  'source_node_id': '69',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573145,\n'
                       '  createdAt: 1757951423860,\n'
                       '  name: dxt1-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'eiot-dev',
  'source_node_id': '70',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573170,\n'
                       '  createdAt: 1757951423900,\n'
                       '  name: eiot-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'eiot-qa1',
  'source_node_id': '71',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573199,\n'
                       '  createdAt: 1757951423937,\n'
                       '  name: eiot-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'empm-dev',
  'source_node_id': '72',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573227,\n'
                       '  createdAt: 1757951423974,\n'
                       '  name: empm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'empm-qa1',
  'source_node_id': '73',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573252,\n'
                       '  createdAt: 1757951424014,\n'
                       '  name: empm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsm-dev',
  'source_node_id': '74',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573277,\n'
                       '  createdAt: 1757951424049,\n'
                       '  name: emsm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsm-qa1',
  'source_node_id': '75',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573328,\n'
                       '  createdAt: 1757951424119,\n'
                       '  name: emsm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsm-qa2',
  'source_node_id': '76',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573355,\n'
                       '  createdAt: 1757951424154,\n'
                       '  name: emsm-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsm-stage',
  'source_node_id': '77',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573381,\n'
                       '  createdAt: 1757951424189,\n'
                       '  name: emsm-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fdaf-dev',
  'source_node_id': '78',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573433,\n'
                       '  createdAt: 1757951424367,\n'
                       '  name: fdaf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fdaf-qa1',
  'source_node_id': '79',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573460,\n'
                       '  createdAt: 1757951424412,\n'
                       '  name: fdaf-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fdfru-dev',
  'source_node_id': '80',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573485,\n'
                       '  createdAt: 1757951424450,\n'
                       '  name: fdfru-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fdfru-qa1',
  'source_node_id': '81',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573510,\n'
                       '  createdAt: 1757951424484,\n'
                       '  name: fdfru-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fltps-dev',
  'source_node_id': '82',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573539,\n'
                       '  createdAt: 1757951424519,\n'
                       '  name: fltps-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fltps-qa1',
  'source_node_id': '83',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573567,\n'
                       '  createdAt: 1757951424554,\n'
                       '  name: fltps-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fosp-dev',
  'source_node_id': '84',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573602,\n'
                       '  createdAt: 1757951424591,\n'
                       '  name: fosp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fosp-qa1',
  'source_node_id': '85',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573636,\n'
                       '  createdAt: 1757951424656,\n'
                       '  name: fosp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'foss-dev',
  'source_node_id': '86',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573672,\n'
                       '  createdAt: 1757951424691,\n'
                       '  name: foss-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'foss-qa1',
  'source_node_id': '87',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573704,\n'
                       '  createdAt: 1757951424727,\n'
                       '  name: foss-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irmkd-dev',
  'source_node_id': '88',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573761,\n'
                       '  createdAt: 1757951424804,\n'
                       '  name: irmkd-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irmkd-qa1',
  'source_node_id': '89',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573790,\n'
                       '  createdAt: 1757951424843,\n'
                       '  name: irmkd-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irol-dev',
  'source_node_id': '90',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573815,\n'
                       '  createdAt: 1757951424878,\n'
                       '  name: irol-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irol-perf1',
  'source_node_id': '91',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573843,\n'
                       '  createdAt: 1757951424912,\n'
                       '  name: irol-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irol-stage',
  'source_node_id': '92',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573898,\n'
                       '  createdAt: 1757951424985,\n'
                       '  name: irol-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irpi-dev',
  'source_node_id': '93',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573922,\n'
                       '  createdAt: 1757951425020,\n'
                       '  name: irpi-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irpi-qa1',
  'source_node_id': '94',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573948,\n'
                       '  createdAt: 1757951425057,\n'
                       '  name: irpi-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irpp-dev',
  'source_node_id': '95',
  'source_properties': '{\n'
                       '  updatedAt: 1758632573974,\n'
                       '  createdAt: 1757951425091,\n'
                       '  name: irpp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irpp-qa1',
  'source_node_id': '96',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574000,\n'
                       '  createdAt: 1757951425125,\n'
                       '  name: irpp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irsu-dev',
  'source_node_id': '97',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574025,\n'
                       '  createdAt: 1757951425163,\n'
                       '  name: irsu-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irsu-perf1',
  'source_node_id': '98',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574050,\n'
                       '  createdAt: 1757951425241,\n'
                       '  name: irsu-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irsu-qa1',
  'source_node_id': '99',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574080,\n'
                       '  createdAt: 1757951425278,\n'
                       '  name: irsu-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irwo-dev',
  'source_node_id': '100',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574108,\n'
                       '  createdAt: 1757951425325,\n'
                       '  name: irwo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irwo-qa1',
  'source_node_id': '101',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574134,\n'
                       '  createdAt: 1757951425360,\n'
                       '  name: irwo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'itds-dev',
  'source_node_id': '102',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574186,\n'
                       '  createdAt: 1757951425448,\n'
                       '  name: itds-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meam-dev',
  'source_node_id': '103',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574211,\n'
                       '  createdAt: 1757951425493,\n'
                       '  name: meam-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meam-qa1',
  'source_node_id': '104',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574237,\n'
                       '  createdAt: 1757951425538,\n'
                       '  name: meam-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meamf-dev',
  'source_node_id': '105',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574262,\n'
                       '  createdAt: 1757951425571,\n'
                       '  name: meamf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meamf-qa1',
  'source_node_id': '106',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574292,\n'
                       '  createdAt: 1757951425607,\n'
                       '  name: meamf-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megn-dev',
  'source_node_id': '107',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574319,\n'
                       '  createdAt: 1757951425644,\n'
                       '  name: megn-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megn-qa1',
  'source_node_id': '108',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574346,\n'
                       '  createdAt: 1757951425681,\n'
                       '  name: megn-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megna-dev',
  'source_node_id': '109',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574374,\n'
                       '  createdAt: 1757951425743,\n'
                       '  name: megna-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megna-qa1',
  'source_node_id': '110',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574399,\n'
                       '  createdAt: 1757951425788,\n'
                       '  name: megna-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megnp-dev',
  'source_node_id': '111',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574425,\n'
                       '  createdAt: 1757951425824,\n'
                       '  name: megnp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megnp-qa1',
  'source_node_id': '112',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574451,\n'
                       '  createdAt: 1757951425859,\n'
                       '  name: megnp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megnu-dev',
  'source_node_id': '113',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574477,\n'
                       '  createdAt: 1757951425896,\n'
                       '  name: megnu-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megnu-qa1',
  'source_node_id': '114',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574508,\n'
                       '  createdAt: 1757951425931,\n'
                       '  name: megnu-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megnw-dev',
  'source_node_id': '115',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574534,\n'
                       '  createdAt: 1757951425966,\n'
                       '  name: megnw-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'megnw-qa1',
  'source_node_id': '116',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574561,\n'
                       '  createdAt: 1757951426002,\n'
                       '  name: megnw-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mepl-dev',
  'source_node_id': '117',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574588,\n'
                       '  createdAt: 1757951426036,\n'
                       '  name: mepl-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mepl-qa1',
  'source_node_id': '118',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574614,\n'
                       '  createdAt: 1757951426072,\n'
                       '  name: mepl-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meui-dev',
  'source_node_id': '119',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574640,\n'
                       '  createdAt: 1757951426107,\n'
                       '  name: meui-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocec-perf1',
  'source_node_id': '120',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574666,\n'
                       '  createdAt: 1757951426145,\n'
                       '  name: ocec-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocec-qa1',
  'source_node_id': '121',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574692,\n'
                       '  createdAt: 1757951426183,\n'
                       '  name: ocec-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocec-stage',
  'source_node_id': '122',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574718,\n'
                       '  createdAt: 1757951426217,\n'
                       '  name: ocec-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsg-dev',
  'source_node_id': '123',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574744,\n'
                       '  createdAt: 1757951426252,\n'
                       '  name: ocsg-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsg-qa1',
  'source_node_id': '124',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574770,\n'
                       '  createdAt: 1757951426286,\n'
                       '  name: ocsg-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocvl-dev',
  'source_node_id': '125',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574796,\n'
                       '  createdAt: 1757951426320,\n'
                       '  name: ocvl-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocvl-perf1',
  'source_node_id': '126',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574823,\n'
                       '  createdAt: 1757951426356,\n'
                       '  name: ocvl-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocvl-qa1',
  'source_node_id': '127',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574848,\n'
                       '  createdAt: 1757951426390,\n'
                       '  name: ocvl-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppms-dev',
  'source_node_id': '128',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574874,\n'
                       '  createdAt: 1757951426425,\n'
                       '  name: ppms-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppms-perf1',
  'source_node_id': '129',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574900,\n'
                       '  createdAt: 1757951426464,\n'
                       '  name: ppms-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppms-qa1',
  'source_node_id': '130',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574930,\n'
                       '  createdAt: 1757951426510,\n'
                       '  name: ppms-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppms-qa2',
  'source_node_id': '131',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574960,\n'
                       '  createdAt: 1757951426548,\n'
                       '  name: ppms-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppms-stage',
  'source_node_id': '132',
  'source_properties': '{\n'
                       '  updatedAt: 1758632574985,\n'
                       '  createdAt: 1757951426587,\n'
                       '  name: ppms-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psfc-dev',
  'source_node_id': '133',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575013,\n'
                       '  createdAt: 1757951426622,\n'
                       '  name: psfc-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psfc-qa1',
  'source_node_id': '134',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575039,\n'
                       '  createdAt: 1757951426658,\n'
                       '  name: psfc-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psfs-dev',
  'source_node_id': '135',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575065,\n'
                       '  createdAt: 1757951426697,\n'
                       '  name: psfs-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psfs-qa1',
  'source_node_id': '136',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575091,\n'
                       '  createdAt: 1757951426741,\n'
                       '  name: psfs-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'scum-dev',
  'source_node_id': '137',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575117,\n'
                       '  createdAt: 1757951426776,\n'
                       '  name: scum-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'scur-dev',
  'source_node_id': '138',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575172,\n'
                       '  createdAt: 1757951426811,\n'
                       '  name: scur-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'scur-perf1',
  'source_node_id': '139',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575197,\n'
                       '  createdAt: 1757951426850,\n'
                       '  name: scur-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'scur-qa1',
  'source_node_id': '140',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575223,\n'
                       '  createdAt: 1757951426884,\n'
                       '  name: scur-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdav6-dev',
  'source_node_id': '141',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575249,\n'
                       '  createdAt: 1757951426918,\n'
                       '  name: sdav6-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdav6-perf1',
  'source_node_id': '142',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575274,\n'
                       '  createdAt: 1757951426952,\n'
                       '  name: sdav6-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdav6-qa1',
  'source_node_id': '143',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575300,\n'
                       '  createdAt: 1757951427014,\n'
                       '  name: sdav6-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sddo-dev',
  'source_node_id': '144',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575330,\n'
                       '  createdAt: 1757951427049,\n'
                       '  name: sddo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sddo-qa1',
  'source_node_id': '145',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575355,\n'
                       '  createdAt: 1757951427084,\n'
                       '  name: sddo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdog-dev',
  'source_node_id': '146',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575381,\n'
                       '  createdAt: 1757951427119,\n'
                       '  name: sdog-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdog-qa1',
  'source_node_id': '147',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575406,\n'
                       '  createdAt: 1757951427155,\n'
                       '  name: sdog-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdovm-dev',
  'source_node_id': '148',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575433,\n'
                       '  createdAt: 1757951427189,\n'
                       '  name: sdovm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdovm-qa1',
  'source_node_id': '149',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575458,\n'
                       '  createdAt: 1757951427224,\n'
                       '  name: sdovm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdovs-dev',
  'source_node_id': '150',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575489,\n'
                       '  createdAt: 1757951427260,\n'
                       '  name: sdovs-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdovs-qa1',
  'source_node_id': '151',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575515,\n'
                       '  createdAt: 1757951427297,\n'
                       '  name: sdovs-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sp2k-dev',
  'source_node_id': '152',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575540,\n'
                       '  createdAt: 1757951427332,\n'
                       '  name: sp2k-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sp2k-qa1',
  'source_node_id': '153',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575565,\n'
                       '  createdAt: 1757951427365,\n'
                       '  name: sp2k-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ssat-dev',
  'source_node_id': '154',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575591,\n'
                       '  createdAt: 1757951427401,\n'
                       '  name: ssat-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ssat-qa1',
  'source_node_id': '155',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575617,\n'
                       '  createdAt: 1757951427436,\n'
                       '  name: ssat-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ssia-dev',
  'source_node_id': '156',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575646,\n'
                       '  createdAt: 1757951427473,\n'
                       '  name: ssia-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ssia-qa1',
  'source_node_id': '157',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575673,\n'
                       '  createdAt: 1757951427511,\n'
                       '  name: ssia-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ssia-stage',
  'source_node_id': '158',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575697,\n'
                       '  createdAt: 1757951427545,\n'
                       '  name: ssia-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wm01ws-dev',
  'source_node_id': '159',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575824,\n'
                       '  createdAt: 1757951427716,\n'
                       '  name: wm01ws-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wm01ws-qa1',
  'source_node_id': '160',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575849,\n'
                       '  createdAt: 1757951427750,\n'
                       '  name: wm01ws-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmpq-dev',
  'source_node_id': '161',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575875,\n'
                       '  createdAt: 1757951427784,\n'
                       '  name: wmpq-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmpq-qa1',
  'source_node_id': '162',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575900,\n'
                       '  createdAt: 1757951427820,\n'
                       '  name: wmpq-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wpcu-dev',
  'source_node_id': '163',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575924,\n'
                       '  createdAt: 1757951427855,\n'
                       '  name: wpcu-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wpcu-qa1',
  'source_node_id': '164',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575949,\n'
                       '  createdAt: 1757951427889,\n'
                       '  name: wpcu-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'zz012-dev',
  'source_node_id': '165',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575974,\n'
                       '  createdAt: 1757951427923,\n'
                       '  name: zz012-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'zz015-dev5',
  'source_node_id': '166',
  'source_properties': '{\n'
                       '  updatedAt: 1758632575999,\n'
                       '  createdAt: 1757951427958,\n'
                       '  name: zz015-dev5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dafi-dev',
  'source_node_id': '167',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577015,\n'
                       '  createdAt: 1757951428201,\n'
                       '  name: dafi-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dafi-qa1',
  'source_node_id': '168',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577040,\n'
                       '  createdAt: 1757951428237,\n'
                       '  name: dafi-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dafo-dev',
  'source_node_id': '169',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577066,\n'
                       '  createdAt: 1757951428285,\n'
                       '  name: dafo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dafo-qa1',
  'source_node_id': '170',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577092,\n'
                       '  createdAt: 1757951428319,\n'
                       '  name: dafo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dird-dev',
  'source_node_id': '171',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577117,\n'
                       '  createdAt: 1757951428353,\n'
                       '  name: dird-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dird-perf1',
  'source_node_id': '172',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577143,\n'
                       '  createdAt: 1757951428388,\n'
                       '  name: dird-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dird-qa1',
  'source_node_id': '173',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577168,\n'
                       '  createdAt: 1757951428423,\n'
                       '  name: dird-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dird-qa2',
  'source_node_id': '174',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577193,\n'
                       '  createdAt: 1757951428456,\n'
                       '  name: dird-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dird-stage',
  'source_node_id': '175',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577219,\n'
                       '  createdAt: 1757951428491,\n'
                       '  name: dird-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dirm-dev',
  'source_node_id': '176',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577244,\n'
                       '  createdAt: 1757951428529,\n'
                       '  name: dirm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dirm-perf1',
  'source_node_id': '177',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577269,\n'
                       '  createdAt: 1757951428587,\n'
                       '  name: dirm-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dirm-perf1',
  'source_node_id': '177',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577269,\n'
                       '  createdAt: 1757951428587,\n'
                       '  name: dirm-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dirm-qa1',
  'source_node_id': '178',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577294,\n'
                       '  createdAt: 1757951428635,\n'
                       '  name: dirm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dirm-qa2',
  'source_node_id': '179',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577320,\n'
                       '  createdAt: 1757951428717,\n'
                       '  name: dirm-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dirm-stage',
  'source_node_id': '180',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577346,\n'
                       '  createdAt: 1757951428795,\n'
                       '  name: dirm-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dmep-dev',
  'source_node_id': '181',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577374,\n'
                       '  createdAt: 1757951428874,\n'
                       '  name: dmep-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dmep-dev2',
  'source_node_id': '182',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577399,\n'
                       '  createdAt: 1757951428954,\n'
                       '  name: dmep-dev2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dmep-perf1',
  'source_node_id': '183',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577424,\n'
                       '  createdAt: 1757951429015,\n'
                       '  name: dmep-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dmep-qa1',
  'source_node_id': '184',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577449,\n'
                       '  createdAt: 1757951429054,\n'
                       '  name: dmep-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dsdsi-dev',
  'source_node_id': '185',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577478,\n'
                       '  createdAt: 1757951429094,\n'
                       '  name: dsdsi-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dsdsi-qa1',
  'source_node_id': '186',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577504,\n'
                       '  createdAt: 1757951429129,\n'
                       '  name: dsdsi-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dssb-dev',
  'source_node_id': '187',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577529,\n'
                       '  createdAt: 1757951429179,\n'
                       '  name: dssb-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dssb-dev',
  'source_node_id': '187',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577529,\n'
                       '  createdAt: 1757951429179,\n'
                       '  name: dssb-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dssb-qa1',
  'source_node_id': '188',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577554,\n'
                       '  createdAt: 1757951429213,\n'
                       '  name: dssb-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dssb-qa1',
  'source_node_id': '188',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577554,\n'
                       '  createdAt: 1757951429213,\n'
                       '  name: dssb-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dssb-stage',
  'source_node_id': '189',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577583,\n'
                       '  createdAt: 1757951429248,\n'
                       '  name: dssb-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dssb-stage',
  'source_node_id': '189',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577583,\n'
                       '  createdAt: 1757951429248,\n'
                       '  name: dssb-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxat-dev',
  'source_node_id': '190',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577609,\n'
                       '  createdAt: 1757951429283,\n'
                       '  name: dxat-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxat-qa1',
  'source_node_id': '191',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577634,\n'
                       '  createdAt: 1757951429319,\n'
                       '  name: dxat-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxcm-dev',
  'source_node_id': '192',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577659,\n'
                       '  createdAt: 1757951429378,\n'
                       '  name: dxcm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxcm-qa1',
  'source_node_id': '193',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577684,\n'
                       '  createdAt: 1757951429420,\n'
                       '  name: dxcm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxcx-dev',
  'source_node_id': '194',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577709,\n'
                       '  createdAt: 1757951429503,\n'
                       '  name: dxcx-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxcx-qa1',
  'source_node_id': '195',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577735,\n'
                       '  createdAt: 1757951429537,\n'
                       '  name: dxcx-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxina-dev',
  'source_node_id': '196',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577760,\n'
                       '  createdAt: 1757951429570,\n'
                       '  name: dxina-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxina-qa1',
  'source_node_id': '197',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577784,\n'
                       '  createdAt: 1757951429603,\n'
                       '  name: dxina-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxpc-dev',
  'source_node_id': '198',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577809,\n'
                       '  createdAt: 1757951429637,\n'
                       '  name: dxpc-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxpc-qa1',
  'source_node_id': '199',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577835,\n'
                       '  createdAt: 1757951429671,\n'
                       '  name: dxpc-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxrp-dev',
  'source_node_id': '200',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577860,\n'
                       '  createdAt: 1757951429704,\n'
                       '  name: dxrp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxrp-qa1',
  'source_node_id': '201',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577885,\n'
                       '  createdAt: 1757951429738,\n'
                       '  name: dxrp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dxrp-qa2',
  'source_node_id': '202',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577911,\n'
                       '  createdAt: 1757951429771,\n'
                       '  name: dxrp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emcj-dev',
  'source_node_id': '203',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577937,\n'
                       '  createdAt: 1757951429805,\n'
                       '  name: emcj-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emcj-qa1',
  'source_node_id': '204',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577962,\n'
                       '  createdAt: 1757951429839,\n'
                       '  name: emcj-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emcn-dev',
  'source_node_id': '205',
  'source_properties': '{\n'
                       '  updatedAt: 1758632577989,\n'
                       '  createdAt: 1757951429875,\n'
                       '  name: emcn-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emcn-perf1',
  'source_node_id': '206',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578014,\n'
                       '  createdAt: 1757951429915,\n'
                       '  name: emcn-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emcn-qa1',
  'source_node_id': '207',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578039,\n'
                       '  createdAt: 1757951429981,\n'
                       '  name: emcn-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emcn-qa2',
  'source_node_id': '208',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578065,\n'
                       '  createdAt: 1757951430031,\n'
                       '  name: emcn-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emcn-stage',
  'source_node_id': '209',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578090,\n'
                       '  createdAt: 1757951430133,\n'
                       '  name: emcn-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emiv-dev',
  'source_node_id': '210',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578115,\n'
                       '  createdAt: 1757951430168,\n'
                       '  name: emiv-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emiv-perf1',
  'source_node_id': '211',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578140,\n'
                       '  createdAt: 1757951430487,\n'
                       '  name: emiv-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emiv-qa1',
  'source_node_id': '212',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578165,\n'
                       '  createdAt: 1757951430523,\n'
                       '  name: emiv-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emiv-qa2',
  'source_node_id': '213',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578194,\n'
                       '  createdAt: 1757951430557,\n'
                       '  name: emiv-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emiv-stage',
  'source_node_id': '214',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578221,\n'
                       '  createdAt: 1757951430595,\n'
                       '  name: emiv-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emju-dev',
  'source_node_id': '215',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578246,\n'
                       '  createdAt: 1757951430630,\n'
                       '  name: emju-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emju-perf1',
  'source_node_id': '216',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578271,\n'
                       '  createdAt: 1757951430665,\n'
                       '  name: emju-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emju-perf1',
  'source_node_id': '216',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578271,\n'
                       '  createdAt: 1757951430665,\n'
                       '  name: emju-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emju-qa1',
  'source_node_id': '217',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578296,\n'
                       '  createdAt: 1757951430700,\n'
                       '  name: emju-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emju-qa2',
  'source_node_id': '218',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578321,\n'
                       '  createdAt: 1757951430757,\n'
                       '  name: emju-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'empg-dev',
  'source_node_id': '219',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578346,\n'
                       '  createdAt: 1757951430806,\n'
                       '  name: empg-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'empg-qa1',
  'source_node_id': '220',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578371,\n'
                       '  createdAt: 1757951430851,\n'
                       '  name: empg-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsp-dev',
  'source_node_id': '221',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578401,\n'
                       '  createdAt: 1757951430930,\n'
                       '  name: emsp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsp-perf1',
  'source_node_id': '222',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578432,\n'
                       '  createdAt: 1757951430982,\n'
                       '  name: emsp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsp-qa1',
  'source_node_id': '223',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578466,\n'
                       '  createdAt: 1757951431086,\n'
                       '  name: emsp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsp-qa2',
  'source_node_id': '224',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578492,\n'
                       '  createdAt: 1757951431155,\n'
                       '  name: emsp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emsp-stage',
  'source_node_id': '225',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578525,\n'
                       '  createdAt: 1757951431202,\n'
                       '  name: emsp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emum-dev',
  'source_node_id': '226',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578652,\n'
                       '  createdAt: 1757951431336,\n'
                       '  name: emum-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emum-perf1',
  'source_node_id': '227',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578682,\n'
                       '  createdAt: 1757951431411,\n'
                       '  name: emum-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emum-qa1',
  'source_node_id': '228',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578707,\n'
                       '  createdAt: 1757951431461,\n'
                       '  name: emum-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emum-qa2',
  'source_node_id': '229',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578732,\n'
                       '  createdAt: 1757951431528,\n'
                       '  name: emum-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'emum-stage',
  'source_node_id': '230',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578758,\n'
                       '  createdAt: 1757951431563,\n'
                       '  name: emum-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esgh-dev',
  'source_node_id': '231',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592140,\n'
                       '  createdAt: 1757951431640,\n'
                       '  name: esgh-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esgh-dev',
  'source_node_id': '231',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592140,\n'
                       '  createdAt: 1757951431640,\n'
                       '  name: esgh-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esgh-dev',
  'source_node_id': '231',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592140,\n'
                       '  createdAt: 1757951431640,\n'
                       '  name: esgh-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'flot-dev',
  'source_node_id': '232',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578839,\n'
                       '  createdAt: 1757951431776,\n'
                       '  name: flot-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'flot-qa1',
  'source_node_id': '233',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578864,\n'
                       '  createdAt: 1757951431810,\n'
                       '  name: flot-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'flts-dev',
  'source_node_id': '234',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578890,\n'
                       '  createdAt: 1757951431849,\n'
                       '  name: flts-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'flts-perf1',
  'source_node_id': '235',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578916,\n'
                       '  createdAt: 1757951431889,\n'
                       '  name: flts-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'flts-qa1',
  'source_node_id': '236',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578943,\n'
                       '  createdAt: 1757951431933,\n'
                       '  name: flts-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fm01-dev',
  'source_node_id': '237',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578967,\n'
                       '  createdAt: 1757951431968,\n'
                       '  name: fm01-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fm01-perf1',
  'source_node_id': '238',
  'source_properties': '{\n'
                       '  updatedAt: 1758632578993,\n'
                       '  createdAt: 1757951432003,\n'
                       '  name: fm01-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fm01-qa1',
  'source_node_id': '239',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579018,\n'
                       '  createdAt: 1757951432039,\n'
                       '  name: fm01-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fmpo-dev',
  'source_node_id': '240',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579043,\n'
                       '  createdAt: 1757951432074,\n'
                       '  name: fmpo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fmpo-qa1',
  'source_node_id': '241',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579071,\n'
                       '  createdAt: 1757951432172,\n'
                       '  name: fmpo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fo01-dev',
  'source_node_id': '242',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579096,\n'
                       '  createdAt: 1757951432213,\n'
                       '  name: fo01-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fo01-perf',
  'source_node_id': '243',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579121,\n'
                       '  createdAt: 1757951432302,\n'
                       '  name: fo01-perf\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fo01-qa1',
  'source_node_id': '244',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579149,\n'
                       '  createdAt: 1757951432342,\n'
                       '  name: fo01-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'fo01-qa2',
  'source_node_id': '245',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579176,\n'
                       '  createdAt: 1757951432399,\n'
                       '  name: fo01-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irbo-dev',
  'source_node_id': '246',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579226,\n'
                       '  createdAt: 1757951432704,\n'
                       '  name: irbo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irbo-perf1',
  'source_node_id': '247',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579254,\n'
                       '  createdAt: 1757951432782,\n'
                       '  name: irbo-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irbo-qa1',
  'source_node_id': '248',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579280,\n'
                       '  createdAt: 1757951432828,\n'
                       '  name: irbo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irdf-dev',
  'source_node_id': '249',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579305,\n'
                       '  createdAt: 1757951432869,\n'
                       '  name: irdf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irft-dev',
  'source_node_id': '250',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579332,\n'
                       '  createdAt: 1757951432916,\n'
                       '  name: irft-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irft-qa1',
  'source_node_id': '251',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579358,\n'
                       '  createdAt: 1757951432970,\n'
                       '  name: irft-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irvo-dev',
  'source_node_id': '252',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579383,\n'
                       '  createdAt: 1757951433078,\n'
                       '  name: irvo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irvo-perf1',
  'source_node_id': '253',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579410,\n'
                       '  createdAt: 1757951433123,\n'
                       '  name: irvo-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irvo-qa1',
  'source_node_id': '254',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579435,\n'
                       '  createdAt: 1757951433188,\n'
                       '  name: irvo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irwo2-dev',
  'source_node_id': '255',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579465,\n'
                       '  createdAt: 1757951433223,\n'
                       '  name: irwo2-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'irwo2-qa1',
  'source_node_id': '256',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579490,\n'
                       '  createdAt: 1757951433257,\n'
                       '  name: irwo2-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'is01-dev',
  'source_node_id': '257',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579514,\n'
                       '  createdAt: 1757951433291,\n'
                       '  name: is01-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'is01-perf',
  'source_node_id': '258',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579540,\n'
                       '  createdAt: 1757951433397,\n'
                       '  name: is01-perf\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'is01-qa1',
  'source_node_id': '259',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579565,\n'
                       '  createdAt: 1757951433592,\n'
                       '  name: is01-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'is01-qa2',
  'source_node_id': '260',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579589,\n'
                       '  createdAt: 1757951433846,\n'
                       '  name: is01-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'is01-qa2',
  'source_node_id': '260',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579589,\n'
                       '  createdAt: 1757951433846,\n'
                       '  name: is01-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mafg-dev',
  'source_node_id': '261',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579665,\n'
                       '  createdAt: 1757951434292,\n'
                       '  name: mafg-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mafg-qa1',
  'source_node_id': '262',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579690,\n'
                       '  createdAt: 1757951434433,\n'
                       '  name: mafg-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mafg-stage',
  'source_node_id': '263',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579715,\n'
                       '  createdAt: 1757951434491,\n'
                       '  name: mafg-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mebp-dev',
  'source_node_id': '264',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579741,\n'
                       '  createdAt: 1757951434530,\n'
                       '  name: mebp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mebp-perf1',
  'source_node_id': '265',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579766,\n'
                       '  createdAt: 1757951434575,\n'
                       '  name: mebp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mebp-perf2',
  'source_node_id': '266',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579792,\n'
                       '  createdAt: 1757951434617,\n'
                       '  name: mebp-perf2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mebp-qa1',
  'source_node_id': '267',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579817,\n'
                       '  createdAt: 1757951434662,\n'
                       '  name: mebp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mebp-qa6',
  'source_node_id': '268',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579842,\n'
                       '  createdAt: 1757951434697,\n'
                       '  name: mebp-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mebp-stage',
  'source_node_id': '269',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579867,\n'
                       '  createdAt: 1757951434739,\n'
                       '  name: mebp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecea-dev',
  'source_node_id': '270',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579892,\n'
                       '  createdAt: 1757951434780,\n'
                       '  name: mecea-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecea-perf1',
  'source_node_id': '271',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579918,\n'
                       '  createdAt: 1757951434861,\n'
                       '  name: mecea-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecea-qa1',
  'source_node_id': '272',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579945,\n'
                       '  createdAt: 1757951434922,\n'
                       '  name: mecea-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecea-stage',
  'source_node_id': '273',
  'source_properties': '{\n'
                       '  updatedAt: 1758632579979,\n'
                       '  createdAt: 1757951434964,\n'
                       '  name: mecea-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-dev',
  'source_node_id': '274',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580004,\n'
                       '  createdAt: 1757951434999,\n'
                       '  name: mecost-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-dev2',
  'source_node_id': '275',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580030,\n'
                       '  createdAt: 1757951435046,\n'
                       '  name: mecost-dev2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-perf1',
  'source_node_id': '276',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580055,\n'
                       '  createdAt: 1757951435081,\n'
                       '  name: mecost-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-qa1',
  'source_node_id': '277',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580081,\n'
                       '  createdAt: 1757951435145,\n'
                       '  name: mecost-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-qa2',
  'source_node_id': '278',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580106,\n'
                       '  createdAt: 1757951435177,\n'
                       '  name: mecost-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-qa3',
  'source_node_id': '279',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580135,\n'
                       '  createdAt: 1757951435223,\n'
                       '  name: mecost-qa3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-qa6',
  'source_node_id': '280',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580161,\n'
                       '  createdAt: 1757951435269,\n'
                       '  name: mecost-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecost-stage',
  'source_node_id': '281',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580186,\n'
                       '  createdAt: 1757951435306,\n'
                       '  name: mecost-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecsa-dev',
  'source_node_id': '282',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580212,\n'
                       '  createdAt: 1757951435360,\n'
                       '  name: mecsa-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mecsa-qa1',
  'source_node_id': '283',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580236,\n'
                       '  createdAt: 1757951435434,\n'
                       '  name: mecsa-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'medi-dev',
  'source_node_id': '284',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580261,\n'
                       '  createdAt: 1757951435588,\n'
                       '  name: medi-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'medi-qa1',
  'source_node_id': '285',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580286,\n'
                       '  createdAt: 1757951435625,\n'
                       '  name: medi-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meimt-dev',
  'source_node_id': '286',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580311,\n'
                       '  createdAt: 1757951435659,\n'
                       '  name: meimt-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meimt-perf1',
  'source_node_id': '287',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580336,\n'
                       '  createdAt: 1757951435692,\n'
                       '  name: meimt-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meimt-qa1',
  'source_node_id': '288',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580365,\n'
                       '  createdAt: 1757951435726,\n'
                       '  name: meimt-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meimt-qa6',
  'source_node_id': '289',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580395,\n'
                       '  createdAt: 1757951435761,\n'
                       '  name: meimt-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meimt-stage',
  'source_node_id': '290',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580421,\n'
                       '  createdAt: 1757951435796,\n'
                       '  name: meimt-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-dev',
  'source_node_id': '291',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580447,\n'
                       '  createdAt: 1757951435828,\n'
                       '  name: meitem-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-dev2',
  'source_node_id': '292',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580478,\n'
                       '  createdAt: 1757951435864,\n'
                       '  name: meitem-dev2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-dev3',
  'source_node_id': '293',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580503,\n'
                       '  createdAt: 1757951435899,\n'
                       '  name: meitem-dev3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-dev4',
  'source_node_id': '294',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580528,\n'
                       '  createdAt: 1757951435935,\n'
                       '  name: meitem-dev4\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-perf1',
  'source_node_id': '295',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580912,\n'
                       '  createdAt: 1757951435968,\n'
                       '  name: meitem-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-perf2',
  'source_node_id': '296',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580940,\n'
                       '  createdAt: 1757951436003,\n'
                       '  name: meitem-perf2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-qa1',
  'source_node_id': '297',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580965,\n'
                       '  createdAt: 1757951436047,\n'
                       '  name: meitem-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-qa2',
  'source_node_id': '298',
  'source_properties': '{\n'
                       '  updatedAt: 1758632580991,\n'
                       '  createdAt: 1757951436093,\n'
                       '  name: meitem-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-qa3',
  'source_node_id': '299',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581019,\n'
                       '  createdAt: 1757951436234,\n'
                       '  name: meitem-qa3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-qa4',
  'source_node_id': '300',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581046,\n'
                       '  createdAt: 1757951436280,\n'
                       '  name: meitem-qa4\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-qa5',
  'source_node_id': '301',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581073,\n'
                       '  createdAt: 1757951436314,\n'
                       '  name: meitem-qa5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-qa6',
  'source_node_id': '302',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581098,\n'
                       '  createdAt: 1757951436354,\n'
                       '  name: meitem-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meitem-stage',
  'source_node_id': '303',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581124,\n'
                       '  createdAt: 1757951436413,\n'
                       '  name: meitem-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memi-dev',
  'source_node_id': '304',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581149,\n'
                       '  createdAt: 1757951436475,\n'
                       '  name: memi-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memi-qa1',
  'source_node_id': '305',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581180,\n'
                       '  createdAt: 1757951436527,\n'
                       '  name: memi-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memiu-dev',
  'source_node_id': '306',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581206,\n'
                       '  createdAt: 1757951436566,\n'
                       '  name: memiu-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memiu-qa1',
  'source_node_id': '307',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581231,\n'
                       '  createdAt: 1757951436612,\n'
                       '  name: memiu-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-dev',
  'source_node_id': '308',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581257,\n'
                       '  createdAt: 1757951436702,\n'
                       '  name: memsp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-dev2',
  'source_node_id': '309',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581282,\n'
                       '  createdAt: 1757951436768,\n'
                       '  name: memsp-dev2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-dev3',
  'source_node_id': '310',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581309,\n'
                       '  createdAt: 1757951436805,\n'
                       '  name: memsp-dev3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-dev4',
  'source_node_id': '311',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581338,\n'
                       '  createdAt: 1757951436906,\n'
                       '  name: memsp-dev4\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-perf1',
  'source_node_id': '312',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581364,\n'
                       '  createdAt: 1757951437004,\n'
                       '  name: memsp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-perf2',
  'source_node_id': '313',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581394,\n'
                       '  createdAt: 1757951437176,\n'
                       '  name: memsp-perf2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-qa1',
  'source_node_id': '314',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581420,\n'
                       '  createdAt: 1757951437290,\n'
                       '  name: memsp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-qa2',
  'source_node_id': '315',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581445,\n'
                       '  createdAt: 1757951437336,\n'
                       '  name: memsp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-qa3',
  'source_node_id': '316',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581471,\n'
                       '  createdAt: 1757951437477,\n'
                       '  name: memsp-qa3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-qa4',
  'source_node_id': '317',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581497,\n'
                       '  createdAt: 1757951437599,\n'
                       '  name: memsp-qa4\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-qa5',
  'source_node_id': '318',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581522,\n'
                       '  createdAt: 1757951437670,\n'
                       '  name: memsp-qa5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-qa6',
  'source_node_id': '319',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581548,\n'
                       '  createdAt: 1757951437808,\n'
                       '  name: memsp-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'memsp-stage',
  'source_node_id': '320',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581574,\n'
                       '  createdAt: 1757951437933,\n'
                       '  name: memsp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menfpt-dev',
  'source_node_id': '321',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581600,\n'
                       '  createdAt: 1757951437989,\n'
                       '  name: menfpt-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menfpt-perf1',
  'source_node_id': '322',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581625,\n'
                       '  createdAt: 1757951438023,\n'
                       '  name: menfpt-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menfpt-qa1',
  'source_node_id': '323',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581650,\n'
                       '  createdAt: 1757951438059,\n'
                       '  name: menfpt-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menfpt-qa2',
  'source_node_id': '324',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581675,\n'
                       '  createdAt: 1757951438093,\n'
                       '  name: menfpt-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menfpt-qa6',
  'source_node_id': '325',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581700,\n'
                       '  createdAt: 1757951438129,\n'
                       '  name: menfpt-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menfpt-stage',
  'source_node_id': '326',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581726,\n'
                       '  createdAt: 1757951438163,\n'
                       '  name: menfpt-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menu-dev',
  'source_node_id': '327',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581751,\n'
                       '  createdAt: 1757951438201,\n'
                       '  name: menu-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'menu-qa1',
  'source_node_id': '328',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581784,\n'
                       '  createdAt: 1757951438235,\n'
                       '  name: menu-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mesmr-dev',
  'source_node_id': '329',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581830,\n'
                       '  createdAt: 1757951438271,\n'
                       '  name: mesmr-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'messp-dev',
  'source_node_id': '330',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581856,\n'
                       '  createdAt: 1757951438308,\n'
                       '  name: messp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'messp-qa1',
  'source_node_id': '331',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581886,\n'
                       '  createdAt: 1757951438342,\n'
                       '  name: messp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mest-dev',
  'source_node_id': '332',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581911,\n'
                       '  createdAt: 1757951438382,\n'
                       '  name: mest-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mestk-dev',
  'source_node_id': '333',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581937,\n'
                       '  createdAt: 1757951438415,\n'
                       '  name: mestk-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mestk-qa1',
  'source_node_id': '334',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581964,\n'
                       '  createdAt: 1757951438448,\n'
                       '  name: mestk-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mestk-qa2',
  'source_node_id': '335',
  'source_properties': '{\n'
                       '  updatedAt: 1758632581989,\n'
                       '  createdAt: 1757951438481,\n'
                       '  name: mestk-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mestk-stage',
  'source_node_id': '336',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582014,\n'
                       '  createdAt: 1757951438514,\n'
                       '  name: mestk-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meup-dev',
  'source_node_id': '337',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582041,\n'
                       '  createdAt: 1757951438549,\n'
                       '  name: meup-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meup-qa1',
  'source_node_id': '338',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582067,\n'
                       '  createdAt: 1757951438601,\n'
                       '  name: meup-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meupp-dev',
  'source_node_id': '339',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582091,\n'
                       '  createdAt: 1757951438634,\n'
                       '  name: meupp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meupp-perf1',
  'source_node_id': '340',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582119,\n'
                       '  createdAt: 1757951438666,\n'
                       '  name: meupp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meupp-perf2',
  'source_node_id': '341',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582145,\n'
                       '  createdAt: 1757951438699,\n'
                       '  name: meupp-perf2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meupp-qa1',
  'source_node_id': '342',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582170,\n'
                       '  createdAt: 1757951438757,\n'
                       '  name: meupp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meupp-qa2',
  'source_node_id': '343',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582196,\n'
                       '  createdAt: 1757951438838,\n'
                       '  name: meupp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meupp-qa6',
  'source_node_id': '344',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582221,\n'
                       '  createdAt: 1757951438933,\n'
                       '  name: meupp-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'meupp-stage',
  'source_node_id': '345',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582246,\n'
                       '  createdAt: 1757951438979,\n'
                       '  name: meupp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevend-dev',
  'source_node_id': '346',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582272,\n'
                       '  createdAt: 1757951439038,\n'
                       '  name: mevend-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevend-dev2',
  'source_node_id': '347',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582298,\n'
                       '  createdAt: 1757951439074,\n'
                       '  name: mevend-dev2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevend-perf1',
  'source_node_id': '348',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582325,\n'
                       '  createdAt: 1757951439126,\n'
                       '  name: mevend-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevend-qa1',
  'source_node_id': '349',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582425,\n'
                       '  createdAt: 1757951439161,\n'
                       '  name: mevend-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevend-qa5',
  'source_node_id': '350',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582525,\n'
                       '  createdAt: 1757951439195,\n'
                       '  name: mevend-qa5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevend-qa6',
  'source_node_id': '351',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582625,\n'
                       '  createdAt: 1757951439227,\n'
                       '  name: mevend-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevend-stage',
  'source_node_id': '352',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582650,\n'
                       '  createdAt: 1757951439260,\n'
                       '  name: mevend-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevir-dev',
  'source_node_id': '353',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582678,\n'
                       '  createdAt: 1757951439293,\n'
                       '  name: mevir-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevir-perf1',
  'source_node_id': '354',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582703,\n'
                       '  createdAt: 1757951439326,\n'
                       '  name: mevir-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevir-perf2',
  'source_node_id': '355',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582729,\n'
                       '  createdAt: 1757951439358,\n'
                       '  name: mevir-perf2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevir-qa1',
  'source_node_id': '356',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582755,\n'
                       '  createdAt: 1757951439399,\n'
                       '  name: mevir-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevir-qa6',
  'source_node_id': '357',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582780,\n'
                       '  createdAt: 1757951439439,\n'
                       '  name: mevir-qa6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'mevir-stage',
  'source_node_id': '358',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582805,\n'
                       '  createdAt: 1757951439473,\n'
                       '  name: mevir-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocas-dev',
  'source_node_id': '359',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582834,\n'
                       '  createdAt: 1757951439506,\n'
                       '  name: ocas-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocas-perf1',
  'source_node_id': '360',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582858,\n'
                       '  createdAt: 1757951439541,\n'
                       '  name: ocas-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocas-qa1',
  'source_node_id': '361',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582884,\n'
                       '  createdAt: 1757951439574,\n'
                       '  name: ocas-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocas-qa2',
  'source_node_id': '362',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582909,\n'
                       '  createdAt: 1757951439607,\n'
                       '  name: ocas-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocas-stage',
  'source_node_id': '363',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582934,\n'
                       '  createdAt: 1757951439639,\n'
                       '  name: ocas-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocat-dev',
  'source_node_id': '364',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582959,\n'
                       '  createdAt: 1757951439675,\n'
                       '  name: ocat-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocat-perf1',
  'source_node_id': '365',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582984,\n'
                       '  createdAt: 1757951439763,\n'
                       '  name: ocat-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocat-perf1',
  'source_node_id': '365',
  'source_properties': '{\n'
                       '  updatedAt: 1758632582984,\n'
                       '  createdAt: 1757951439763,\n'
                       '  name: ocat-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocat-qa1',
  'source_node_id': '366',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583010,\n'
                       '  createdAt: 1757951439825,\n'
                       '  name: ocat-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocat-qa2',
  'source_node_id': '367',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583035,\n'
                       '  createdAt: 1757951439867,\n'
                       '  name: ocat-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocat-stage',
  'source_node_id': '368',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583060,\n'
                       '  createdAt: 1757951439900,\n'
                       '  name: ocat-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocbc-dev',
  'source_node_id': '369',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583088,\n'
                       '  createdAt: 1757951440002,\n'
                       '  name: ocbc-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocbc-perf1',
  'source_node_id': '370',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583113,\n'
                       '  createdAt: 1757951440054,\n'
                       '  name: ocbc-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocbc-qa1',
  'source_node_id': '371',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583141,\n'
                       '  createdAt: 1757951440111,\n'
                       '  name: ocbc-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occl-dev',
  'source_node_id': '372',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583168,\n'
                       '  createdAt: 1757951440170,\n'
                       '  name: occl-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occl-perf1',
  'source_node_id': '373',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583193,\n'
                       '  createdAt: 1757951440234,\n'
                       '  name: occl-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occl-qa1',
  'source_node_id': '374',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583218,\n'
                       '  createdAt: 1757951440287,\n'
                       '  name: occl-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occl-qa2',
  'source_node_id': '375',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583245,\n'
                       '  createdAt: 1757951440335,\n'
                       '  name: occl-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occl-stage',
  'source_node_id': '376',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583271,\n'
                       '  createdAt: 1757951440373,\n'
                       '  name: occl-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occo-dev',
  'source_node_id': '377',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583296,\n'
                       '  createdAt: 1757951440408,\n'
                       '  name: occo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occo-qa1',
  'source_node_id': '378',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583351,\n'
                       '  createdAt: 1757951440471,\n'
                       '  name: occo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occp-dev',
  'source_node_id': '379',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583415,\n'
                       '  createdAt: 1757951440505,\n'
                       '  name: occp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occp-perf1',
  'source_node_id': '380',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583441,\n'
                       '  createdAt: 1757951440537,\n'
                       '  name: occp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occp-perf1',
  'source_node_id': '380',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583441,\n'
                       '  createdAt: 1757951440537,\n'
                       '  name: occp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occp-qa1',
  'source_node_id': '381',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583467,\n'
                       '  createdAt: 1757951440571,\n'
                       '  name: occp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occp-qa2',
  'source_node_id': '382',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583493,\n'
                       '  createdAt: 1757951440605,\n'
                       '  name: occp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occp-stage',
  'source_node_id': '383',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583521,\n'
                       '  createdAt: 1757951440638,\n'
                       '  name: occp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occs-dev',
  'source_node_id': '384',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583546,\n'
                       '  createdAt: 1757951440671,\n'
                       '  name: occs-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'occs-qa1',
  'source_node_id': '385',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583572,\n'
                       '  createdAt: 1757951440710,\n'
                       '  name: occs-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdp-dev',
  'source_node_id': '386',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583597,\n'
                       '  createdAt: 1757951440743,\n'
                       '  name: ocdp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdp-qa1',
  'source_node_id': '387',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583622,\n'
                       '  createdAt: 1757951440774,\n'
                       '  name: ocdp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdr-dev1',
  'source_node_id': '388',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583647,\n'
                       '  createdAt: 1757951440808,\n'
                       '  name: ocdr-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdr-perf1',
  'source_node_id': '389',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583672,\n'
                       '  createdAt: 1757951440849,\n'
                       '  name: ocdr-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdr-qa1',
  'source_node_id': '390',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583696,\n'
                       '  createdAt: 1757951440882,\n'
                       '  name: ocdr-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocds-dev',
  'source_node_id': '391',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583721,\n'
                       '  createdAt: 1757951440913,\n'
                       '  name: ocds-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocds-perf1',
  'source_node_id': '392',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583746,\n'
                       '  createdAt: 1757951440948,\n'
                       '  name: ocds-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocds-qa1',
  'source_node_id': '393',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583772,\n'
                       '  createdAt: 1757951440981,\n'
                       '  name: ocds-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocds-qa2',
  'source_node_id': '394',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583797,\n'
                       '  createdAt: 1757951441015,\n'
                       '  name: ocds-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocds-stage',
  'source_node_id': '395',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583823,\n'
                       '  createdAt: 1757951441048,\n'
                       '  name: ocds-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdt-dev',
  'source_node_id': '396',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583850,\n'
                       '  createdAt: 1757951441084,\n'
                       '  name: ocdt-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdt-perf1',
  'source_node_id': '397',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583876,\n'
                       '  createdAt: 1757951441117,\n'
                       '  name: ocdt-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdt-perf1',
  'source_node_id': '397',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583876,\n'
                       '  createdAt: 1757951441117,\n'
                       '  name: ocdt-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdt-qa1',
  'source_node_id': '398',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583901,\n'
                       '  createdAt: 1757951441150,\n'
                       '  name: ocdt-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdt-qa2',
  'source_node_id': '399',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583928,\n'
                       '  createdAt: 1757951441186,\n'
                       '  name: ocdt-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocdt-stage',
  'source_node_id': '400',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583953,\n'
                       '  createdAt: 1757951441218,\n'
                       '  name: ocdt-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oceg-dev',
  'source_node_id': '401',
  'source_properties': '{\n'
                       '  updatedAt: 1758632583979,\n'
                       '  createdAt: 1757951441251,\n'
                       '  name: oceg-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oceg-perf1',
  'source_node_id': '402',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584005,\n'
                       '  createdAt: 1757951441284,\n'
                       '  name: oceg-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oceg-qa1',
  'source_node_id': '403',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584031,\n'
                       '  createdAt: 1757951441318,\n'
                       '  name: oceg-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oceg-qa2',
  'source_node_id': '404',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584057,\n'
                       '  createdAt: 1757951441351,\n'
                       '  name: oceg-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocim-dev',
  'source_node_id': '405',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584082,\n'
                       '  createdAt: 1757951441384,\n'
                       '  name: ocim-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocim-qa1',
  'source_node_id': '406',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584110,\n'
                       '  createdAt: 1757951441430,\n'
                       '  name: ocim-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocim-qa2',
  'source_node_id': '407',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584135,\n'
                       '  createdAt: 1757951441474,\n'
                       '  name: ocim-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocim-stage',
  'source_node_id': '408',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584166,\n'
                       '  createdAt: 1757951441506,\n'
                       '  name: ocim-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocis-dev',
  'source_node_id': '409',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584191,\n'
                       '  createdAt: 1757951441542,\n'
                       '  name: ocis-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocis-perf1',
  'source_node_id': '410',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584216,\n'
                       '  createdAt: 1757951441576,\n'
                       '  name: ocis-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocis-perf1',
  'source_node_id': '410',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584216,\n'
                       '  createdAt: 1757951441576,\n'
                       '  name: ocis-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocis-qa1',
  'source_node_id': '411',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584241,\n'
                       '  createdAt: 1757951441610,\n'
                       '  name: ocis-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocis-qa2',
  'source_node_id': '412',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584268,\n'
                       '  createdAt: 1757951441646,\n'
                       '  name: ocis-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocis-stage',
  'source_node_id': '413',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584293,\n'
                       '  createdAt: 1757951441688,\n'
                       '  name: ocis-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocmc-dev',
  'source_node_id': '414',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584319,\n'
                       '  createdAt: 1757951441728,\n'
                       '  name: ocmc-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocmc-perf1',
  'source_node_id': '415',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584347,\n'
                       '  createdAt: 1757951441760,\n'
                       '  name: ocmc-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocmc-qa1',
  'source_node_id': '416',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584375,\n'
                       '  createdAt: 1757951441794,\n'
                       '  name: ocmc-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocmc-qa2',
  'source_node_id': '417',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584411,\n'
                       '  createdAt: 1757951441827,\n'
                       '  name: ocmc-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocmc-stage',
  'source_node_id': '418',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584437,\n'
                       '  createdAt: 1757951441860,\n'
                       '  name: ocmc-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocom-dev',
  'source_node_id': '419',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584464,\n'
                       '  createdAt: 1757951441893,\n'
                       '  name: ocom-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocom-perf1',
  'source_node_id': '420',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584490,\n'
                       '  createdAt: 1757951441927,\n'
                       '  name: ocom-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocom-qa1',
  'source_node_id': '421',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584516,\n'
                       '  createdAt: 1757951441961,\n'
                       '  name: ocom-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocom-qa2',
  'source_node_id': '422',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584542,\n'
                       '  createdAt: 1757951441995,\n'
                       '  name: ocom-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocom-stage',
  'source_node_id': '423',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584567,\n'
                       '  createdAt: 1757951442028,\n'
                       '  name: ocom-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpi-dev',
  'source_node_id': '424',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584593,\n'
                       '  createdAt: 1757951442062,\n'
                       '  name: ocpi-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpi-perf1',
  'source_node_id': '425',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584618,\n'
                       '  createdAt: 1757951442095,\n'
                       '  name: ocpi-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpi-qa1',
  'source_node_id': '426',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584646,\n'
                       '  createdAt: 1757951442131,\n'
                       '  name: ocpi-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpi-qa2',
  'source_node_id': '427',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584671,\n'
                       '  createdAt: 1757951442166,\n'
                       '  name: ocpi-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpi-stage',
  'source_node_id': '428',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584697,\n'
                       '  createdAt: 1757951442200,\n'
                       '  name: ocpi-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpr-dev',
  'source_node_id': '429',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584722,\n'
                       '  createdAt: 1757951442234,\n'
                       '  name: ocpr-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpr-perf1',
  'source_node_id': '430',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584750,\n'
                       '  createdAt: 1757951442266,\n'
                       '  name: ocpr-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpr-qa1',
  'source_node_id': '431',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584777,\n'
                       '  createdAt: 1757951442305,\n'
                       '  name: ocpr-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpr-qa2',
  'source_node_id': '432',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584805,\n'
                       '  createdAt: 1757951442337,\n'
                       '  name: ocpr-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocpr-stage',
  'source_node_id': '433',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584831,\n'
                       '  createdAt: 1757951442369,\n'
                       '  name: ocpr-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocrf-dev',
  'source_node_id': '434',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584857,\n'
                       '  createdAt: 1757951442406,\n'
                       '  name: ocrf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsp-dev',
  'source_node_id': '435',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584883,\n'
                       '  createdAt: 1757951442439,\n'
                       '  name: ocsp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsp-perf1',
  'source_node_id': '436',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584909,\n'
                       '  createdAt: 1757951442478,\n'
                       '  name: ocsp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsp-perf1',
  'source_node_id': '436',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584909,\n'
                       '  createdAt: 1757951442478,\n'
                       '  name: ocsp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsp-qa1',
  'source_node_id': '437',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584935,\n'
                       '  createdAt: 1757951442514,\n'
                       '  name: ocsp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsp-qa2',
  'source_node_id': '438',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584961,\n'
                       '  createdAt: 1757951442550,\n'
                       '  name: ocsp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocsp-stage',
  'source_node_id': '439',
  'source_properties': '{\n'
                       '  updatedAt: 1758632584986,\n'
                       '  createdAt: 1757951442582,\n'
                       '  name: ocsp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octs-dev',
  'source_node_id': '440',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585012,\n'
                       '  createdAt: 1757951442619,\n'
                       '  name: octs-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octs-perf1',
  'source_node_id': '441',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585041,\n'
                       '  createdAt: 1757951442653,\n'
                       '  name: octs-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octs-perf1',
  'source_node_id': '441',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585041,\n'
                       '  createdAt: 1757951442653,\n'
                       '  name: octs-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octs-qa1',
  'source_node_id': '442',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585069,\n'
                       '  createdAt: 1757951442686,\n'
                       '  name: octs-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octs-qa2',
  'source_node_id': '443',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585096,\n'
                       '  createdAt: 1757951442719,\n'
                       '  name: octs-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octs-stage',
  'source_node_id': '444',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585122,\n'
                       '  createdAt: 1757951442754,\n'
                       '  name: octs-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octx-dev',
  'source_node_id': '445',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585148,\n'
                       '  createdAt: 1757951442787,\n'
                       '  name: octx-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octx-perf1',
  'source_node_id': '446',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585173,\n'
                       '  createdAt: 1757951442821,\n'
                       '  name: octx-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octx-qa1',
  'source_node_id': '447',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585199,\n'
                       '  createdAt: 1757951442857,\n'
                       '  name: octx-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octx-qa2',
  'source_node_id': '448',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585228,\n'
                       '  createdAt: 1757951442890,\n'
                       '  name: octx-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'octx-stage',
  'source_node_id': '449',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585253,\n'
                       '  createdAt: 1757951442923,\n'
                       '  name: octx-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocus-dev',
  'source_node_id': '450',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585278,\n'
                       '  createdAt: 1757951442957,\n'
                       '  name: ocus-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocus-perf1',
  'source_node_id': '451',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585303,\n'
                       '  createdAt: 1757951442992,\n'
                       '  name: ocus-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocus-qa1',
  'source_node_id': '452',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585328,\n'
                       '  createdAt: 1757951443033,\n'
                       '  name: ocus-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocus-qa2',
  'source_node_id': '453',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585353,\n'
                       '  createdAt: 1757951443068,\n'
                       '  name: ocus-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ocus-stage',
  'source_node_id': '454',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585379,\n'
                       '  createdAt: 1757951443102,\n'
                       '  name: ocus-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oscp-dev',
  'source_node_id': '455',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585405,\n'
                       '  createdAt: 1757951443134,\n'
                       '  name: oscp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oscp-perf1',
  'source_node_id': '456',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585432,\n'
                       '  createdAt: 1757951443167,\n'
                       '  name: oscp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oscp-qa1',
  'source_node_id': '457',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585458,\n'
                       '  createdAt: 1757951443203,\n'
                       '  name: oscp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oscp-qa2',
  'source_node_id': '458',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585483,\n'
                       '  createdAt: 1757951443236,\n'
                       '  name: oscp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oscp-stage',
  'source_node_id': '459',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585509,\n'
                       '  createdAt: 1757951443268,\n'
                       '  name: oscp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osdp-dev',
  'source_node_id': '460',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585533,\n'
                       '  createdAt: 1757951443301,\n'
                       '  name: osdp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osdp-perf1',
  'source_node_id': '461',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585559,\n'
                       '  createdAt: 1757951443332,\n'
                       '  name: osdp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osdp-perf1',
  'source_node_id': '461',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585559,\n'
                       '  createdAt: 1757951443332,\n'
                       '  name: osdp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osdp-qa1',
  'source_node_id': '462',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585585,\n'
                       '  createdAt: 1757951443365,\n'
                       '  name: osdp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osdp-qa2',
  'source_node_id': '463',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585610,\n'
                       '  createdAt: 1757951443397,\n'
                       '  name: osdp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osdp-stage',
  'source_node_id': '464',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585635,\n'
                       '  createdAt: 1757951443430,\n'
                       '  name: osdp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfc-dev',
  'source_node_id': '465',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585665,\n'
                       '  createdAt: 1757951443462,\n'
                       '  name: osfc-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfc-perf1',
  'source_node_id': '466',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585690,\n'
                       '  createdAt: 1757951443495,\n'
                       '  name: osfc-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfc-qa1',
  'source_node_id': '467',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585716,\n'
                       '  createdAt: 1757951443529,\n'
                       '  name: osfc-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfc-qa2',
  'source_node_id': '468',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585741,\n'
                       '  createdAt: 1757951443568,\n'
                       '  name: osfc-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfc-stage',
  'source_node_id': '469',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585766,\n'
                       '  createdAt: 1757951443601,\n'
                       '  name: osfc-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfi-perf1',
  'source_node_id': '470',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585796,\n'
                       '  createdAt: 1757951443634,\n'
                       '  name: osfi-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfi-qa1',
  'source_node_id': '471',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585822,\n'
                       '  createdAt: 1757951443666,\n'
                       '  name: osfi-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfi-qa2',
  'source_node_id': '472',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585848,\n'
                       '  createdAt: 1757951443698,\n'
                       '  name: osfi-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osfi-stage',
  'source_node_id': '473',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585874,\n'
                       '  createdAt: 1757951443732,\n'
                       '  name: osfi-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oslm-qa1',
  'source_node_id': '474',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585899,\n'
                       '  createdAt: 1757951443763,\n'
                       '  name: oslm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrf-dev',
  'source_node_id': '475',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585924,\n'
                       '  createdAt: 1757951443795,\n'
                       '  name: osrf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrf-perf1',
  'source_node_id': '476',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585949,\n'
                       '  createdAt: 1757951443828,\n'
                       '  name: osrf-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrf-qa1',
  'source_node_id': '477',
  'source_properties': '{\n'
                       '  updatedAt: 1758632585974,\n'
                       '  createdAt: 1757951443860,\n'
                       '  name: osrf-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrf-qa2',
  'source_node_id': '478',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586000,\n'
                       '  createdAt: 1757951443899,\n'
                       '  name: osrf-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrf-stage',
  'source_node_id': '479',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586025,\n'
                       '  createdAt: 1757951443932,\n'
                       '  name: osrf-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrr-dev',
  'source_node_id': '480',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586051,\n'
                       '  createdAt: 1757951443964,\n'
                       '  name: osrr-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrr-perf1',
  'source_node_id': '481',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586078,\n'
                       '  createdAt: 1757951443996,\n'
                       '  name: osrr-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrr-qa1',
  'source_node_id': '482',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586105,\n'
                       '  createdAt: 1757951444028,\n'
                       '  name: osrr-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrr-qa2',
  'source_node_id': '483',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586129,\n'
                       '  createdAt: 1757951444060,\n'
                       '  name: osrr-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osrr-stage',
  'source_node_id': '484',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586155,\n'
                       '  createdAt: 1757951444092,\n'
                       '  name: osrr-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ossr-dev',
  'source_node_id': '485',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586183,\n'
                       '  createdAt: 1757951444140,\n'
                       '  name: ossr-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ossr-dev1',
  'source_node_id': '486',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586208,\n'
                       '  createdAt: 1757951444171,\n'
                       '  name: ossr-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ossr-perf1',
  'source_node_id': '487',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586235,\n'
                       '  createdAt: 1757951444207,\n'
                       '  name: ossr-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ossr-perf1',
  'source_node_id': '487',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586235,\n'
                       '  createdAt: 1757951444207,\n'
                       '  name: ossr-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ossr-qa1',
  'source_node_id': '488',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586263,\n'
                       '  createdAt: 1757951444246,\n'
                       '  name: ossr-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ossr-qa2',
  'source_node_id': '489',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586288,\n'
                       '  createdAt: 1757951444281,\n'
                       '  name: ossr-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ossr-stage',
  'source_node_id': '490',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586313,\n'
                       '  createdAt: 1757951444313,\n'
                       '  name: ossr-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pabs-dev',
  'source_node_id': '491',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586339,\n'
                       '  createdAt: 1757951444346,\n'
                       '  name: pabs-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pabs-perf1',
  'source_node_id': '492',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586363,\n'
                       '  createdAt: 1757951444380,\n'
                       '  name: pabs-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pabs-perf1',
  'source_node_id': '492',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586363,\n'
                       '  createdAt: 1757951444380,\n'
                       '  name: pabs-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pabs-qa1',
  'source_node_id': '493',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586392,\n'
                       '  createdAt: 1757951444412,\n'
                       '  name: pabs-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pabs-qa2',
  'source_node_id': '494',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586421,\n'
                       '  createdAt: 1757951444444,\n'
                       '  name: pabs-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pabs-stage',
  'source_node_id': '495',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586447,\n'
                       '  createdAt: 1757951444476,\n'
                       '  name: pabs-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppap-dev',
  'source_node_id': '496',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586472,\n'
                       '  createdAt: 1757951444507,\n'
                       '  name: ppap-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppap-perf1',
  'source_node_id': '497',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586506,\n'
                       '  createdAt: 1757951444539,\n'
                       '  name: ppap-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppap-qa1',
  'source_node_id': '498',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586538,\n'
                       '  createdAt: 1757951444584,\n'
                       '  name: ppap-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppap-stage',
  'source_node_id': '499',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586563,\n'
                       '  createdAt: 1757951444617,\n'
                       '  name: ppap-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppmp-dev',
  'source_node_id': '500',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586589,\n'
                       '  createdAt: 1757951444649,\n'
                       '  name: ppmp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppmp-perf1',
  'source_node_id': '501',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586615,\n'
                       '  createdAt: 1757951444682,\n'
                       '  name: ppmp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppmp-perf1',
  'source_node_id': '501',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586615,\n'
                       '  createdAt: 1757951444682,\n'
                       '  name: ppmp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppmp-qa1',
  'source_node_id': '502',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586648,\n'
                       '  createdAt: 1757951444714,\n'
                       '  name: ppmp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ppmp-stage',
  'source_node_id': '503',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586673,\n'
                       '  createdAt: 1757951444745,\n'
                       '  name: ppmp-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ps01-dev',
  'source_node_id': '504',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586699,\n'
                       '  createdAt: 1757951444827,\n'
                       '  name: ps01-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ps01-perf1',
  'source_node_id': '505',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586724,\n'
                       '  createdAt: 1757951444862,\n'
                       '  name: ps01-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ps01-perf1',
  'source_node_id': '505',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586724,\n'
                       '  createdAt: 1757951444862,\n'
                       '  name: ps01-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ps01-qa1',
  'source_node_id': '506',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586750,\n'
                       '  createdAt: 1757951444896,\n'
                       '  name: ps01-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ps01-qa1',
  'source_node_id': '506',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586750,\n'
                       '  createdAt: 1757951444896,\n'
                       '  name: ps01-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psbr-dev',
  'source_node_id': '507',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586776,\n'
                       '  createdAt: 1757951444929,\n'
                       '  name: psbr-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psbr-perf1',
  'source_node_id': '508',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586802,\n'
                       '  createdAt: 1757951444962,\n'
                       '  name: psbr-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psbr-qa1',
  'source_node_id': '509',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586827,\n'
                       '  createdAt: 1757951445005,\n'
                       '  name: psbr-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psep-dev',
  'source_node_id': '510',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586853,\n'
                       '  createdAt: 1757951445041,\n'
                       '  name: psep-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psep-qa1',
  'source_node_id': '511',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586878,\n'
                       '  createdAt: 1757951445077,\n'
                       '  name: psep-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pspm-dev',
  'source_node_id': '512',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586902,\n'
                       '  createdAt: 1757951445108,\n'
                       '  name: pspm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pspm-qa1',
  'source_node_id': '513',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586937,\n'
                       '  createdAt: 1757951445140,\n'
                       '  name: pspm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pssc-dev',
  'source_node_id': '514',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586966,\n'
                       '  createdAt: 1757951445171,\n'
                       '  name: pssc-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'pssc-qa1',
  'source_node_id': '515',
  'source_properties': '{\n'
                       '  updatedAt: 1758632586993,\n'
                       '  createdAt: 1757951445202,\n'
                       '  name: pssc-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psti-dev',
  'source_node_id': '516',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587018,\n'
                       '  createdAt: 1757951445233,\n'
                       '  name: psti-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'psti-qa1',
  'source_node_id': '517',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587043,\n'
                       '  createdAt: 1757951445266,\n'
                       '  name: psti-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rcdo-dev',
  'source_node_id': '518',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587068,\n'
                       '  createdAt: 1757951445298,\n'
                       '  name: rcdo-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rcdo-qa1',
  'source_node_id': '519',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587095,\n'
                       '  createdAt: 1757951445331,\n'
                       '  name: rcdo-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ri01w-dev',
  'source_node_id': '520',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587120,\n'
                       '  createdAt: 1757951445363,\n'
                       '  name: ri01w-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ri01w-qa1',
  'source_node_id': '521',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587145,\n'
                       '  createdAt: 1757951445395,\n'
                       '  name: ri01w-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxds-dev',
  'source_node_id': '522',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587171,\n'
                       '  createdAt: 1757951445430,\n'
                       '  name: rxds-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxds-perf1',
  'source_node_id': '523',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587196,\n'
                       '  createdAt: 1757951445461,\n'
                       '  name: rxds-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxds-qa1',
  'source_node_id': '524',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587221,\n'
                       '  createdAt: 1757951445493,\n'
                       '  name: rxds-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxds-qa2',
  'source_node_id': '525',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587253,\n'
                       '  createdAt: 1757951445525,\n'
                       '  name: rxds-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxni-dev',
  'source_node_id': '526',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587308,\n'
                       '  createdAt: 1757951445557,\n'
                       '  name: rxni-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxni-perf1',
  'source_node_id': '527',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587333,\n'
                       '  createdAt: 1757951445590,\n'
                       '  name: rxni-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxni-qa1',
  'source_node_id': '528',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587359,\n'
                       '  createdAt: 1757951445623,\n'
                       '  name: rxni-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rxni-qa2',
  'source_node_id': '529',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587384,\n'
                       '  createdAt: 1757951445656,\n'
                       '  name: rxni-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sact-dev',
  'source_node_id': '530',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587435,\n'
                       '  createdAt: 1757951445689,\n'
                       '  name: sact-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sact-qa1',
  'source_node_id': '531',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587460,\n'
                       '  createdAt: 1757951445722,\n'
                       '  name: sact-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdar-dev',
  'source_node_id': '532',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587485,\n'
                       '  createdAt: 1757951445754,\n'
                       '  name: sdar-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdar-perf1',
  'source_node_id': '533',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587511,\n'
                       '  createdAt: 1757951445786,\n'
                       '  name: sdar-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdar-qa1',
  'source_node_id': '534',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587537,\n'
                       '  createdAt: 1757951445823,\n'
                       '  name: sdar-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdov-dev',
  'source_node_id': '535',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587562,\n'
                       '  createdAt: 1757951445855,\n'
                       '  name: sdov-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sdov-qa1',
  'source_node_id': '536',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587587,\n'
                       '  createdAt: 1757951445889,\n'
                       '  name: sdov-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sims-dev',
  'source_node_id': '537',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587612,\n'
                       '  createdAt: 1757951445922,\n'
                       '  name: sims-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sims-perf',
  'source_node_id': '538',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587640,\n'
                       '  createdAt: 1757951445954,\n'
                       '  name: sims-perf\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sims-qa1',
  'source_node_id': '539',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587667,\n'
                       '  createdAt: 1757951445985,\n'
                       '  name: sims-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sims-qa2',
  'source_node_id': '540',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587697,\n'
                       '  createdAt: 1757951446017,\n'
                       '  name: sims-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sims-qa2',
  'source_node_id': '540',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587697,\n'
                       '  createdAt: 1757951446017,\n'
                       '  name: sims-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-eastus-cluster-01',
  'target_node_id': '7054',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571478,\n'
                       '  createdAt: 1758621516558,\n'
                       '  name: esco-aksba2-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sosm-dev',
  'source_node_id': '541',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587722,\n'
                       '  createdAt: 1757951446051,\n'
                       '  name: sosm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sosm-perf1',
  'source_node_id': '542',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587748,\n'
                       '  createdAt: 1757951446083,\n'
                       '  name: sosm-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sosm-qa1',
  'source_node_id': '543',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587775,\n'
                       '  createdAt: 1757951446116,\n'
                       '  name: sosm-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sosm-qa2',
  'source_node_id': '544',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587800,\n'
                       '  createdAt: 1757951446150,\n'
                       '  name: sosm-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sosm-stage',
  'source_node_id': '545',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587825,\n'
                       '  createdAt: 1757951446181,\n'
                       '  name: sosm-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svcg-dev',
  'source_node_id': '546',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587850,\n'
                       '  createdAt: 1757951446219,\n'
                       '  name: svcg-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svcg-qa1',
  'source_node_id': '547',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587875,\n'
                       '  createdAt: 1757951446251,\n'
                       '  name: svcg-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svcp-dev',
  'source_node_id': '548',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587901,\n'
                       '  createdAt: 1757951446283,\n'
                       '  name: svcp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svcp-perf1',
  'source_node_id': '549',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587926,\n'
                       '  createdAt: 1757951446316,\n'
                       '  name: svcp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svcp-qa1',
  'source_node_id': '550',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587951,\n'
                       '  createdAt: 1757951446348,\n'
                       '  name: svcp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svcp-qa2',
  'source_node_id': '551',
  'source_properties': '{\n'
                       '  updatedAt: 1758632587976,\n'
                       '  createdAt: 1757951446379,\n'
                       '  name: svcp-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svdx-dev',
  'source_node_id': '552',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588001,\n'
                       '  createdAt: 1757951446411,\n'
                       '  name: svdx-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svdx-perf1',
  'source_node_id': '553',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588026,\n'
                       '  createdAt: 1757951446442,\n'
                       '  name: svdx-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'svdx-qa1',
  'source_node_id': '554',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588051,\n'
                       '  createdAt: 1757951446476,\n'
                       '  name: svdx-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'test-debug',
  'source_node_id': '555',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588084,\n'
                       '  createdAt: 1757951446508,\n'
                       '  name: test-debug\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'vs01-dev',
  'source_node_id': '556',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588184,\n'
                       '  createdAt: 1757951446637,\n'
                       '  name: vs01-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'vs01-qa1',
  'source_node_id': '557',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588209,\n'
                       '  createdAt: 1757951446669,\n'
                       '  name: vs01-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'vs01-stage',
  'source_node_id': '558',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588235,\n'
                       '  createdAt: 1757951446701,\n'
                       '  name: vs01-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wcax-qa1',
  'source_node_id': '559',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588260,\n'
                       '  createdAt: 1757951446734,\n'
                       '  name: wcax-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wcax-qa2',
  'source_node_id': '560',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588286,\n'
                       '  createdAt: 1757951446771,\n'
                       '  name: wcax-qa2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wcax-stage',
  'source_node_id': '561',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588314,\n'
                       '  createdAt: 1757951446806,\n'
                       '  name: wcax-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmas-dev',
  'source_node_id': '562',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588342,\n'
                       '  createdAt: 1757951446837,\n'
                       '  name: wmas-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmas-qa1',
  'source_node_id': '563',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588369,\n'
                       '  createdAt: 1757951446869,\n'
                       '  name: wmas-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmont-dev',
  'source_node_id': '564',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588394,\n'
                       '  createdAt: 1757951446901,\n'
                       '  name: wmont-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmont-qa1',
  'source_node_id': '565',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588419,\n'
                       '  createdAt: 1757951446933,\n'
                       '  name: wmont-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmpq2-dev',
  'source_node_id': '566',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588447,\n'
                       '  createdAt: 1757951446967,\n'
                       '  name: wmpq2-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmpq2-qa1',
  'source_node_id': '567',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588479,\n'
                       '  createdAt: 1757951446999,\n'
                       '  name: wmpq2-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmvp-dev',
  'source_node_id': '568',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588505,\n'
                       '  createdAt: 1757951447031,\n'
                       '  name: wmvp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wmvp-qa1',
  'source_node_id': '569',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588532,\n'
                       '  createdAt: 1757951447067,\n'
                       '  name: wmvp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wrin-dev',
  'source_node_id': '570',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588562,\n'
                       '  createdAt: 1757951447101,\n'
                       '  name: wrin-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wrin-perf1',
  'source_node_id': '571',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588589,\n'
                       '  createdAt: 1757951447132,\n'
                       '  name: wrin-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wrin-qa1',
  'source_node_id': '572',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588613,\n'
                       '  createdAt: 1757951447164,\n'
                       '  name: wrin-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wrin-stage',
  'source_node_id': '573',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588640,\n'
                       '  createdAt: 1757951447195,\n'
                       '  name: wrin-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wrot-dev',
  'source_node_id': '574',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588665,\n'
                       '  createdAt: 1757951447227,\n'
                       '  name: wrot-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wrot-perf1',
  'source_node_id': '575',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588689,\n'
                       '  createdAt: 1757951447262,\n'
                       '  name: wrot-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'wrot-qa1',
  'source_node_id': '576',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588715,\n'
                       '  createdAt: 1757951447292,\n'
                       '  name: wrot-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'asase00',
  'source_node_id': '577',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588816,\n'
                       '  createdAt: 1757951447385,\n'
                       '  name: asase00\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'bookinfo',
  'source_node_id': '578',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588841,\n'
                       '  createdAt: 1757951447417,\n'
                       '  name: bookinfo\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'dataprotection-microsoft',
  'source_node_id': '579',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588866,\n'
                       '  createdAt: 1757951447449,\n'
                       '  name: dataprotection-microsoft\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esaxo-perf1',
  'source_node_id': '580',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591337,\n'
                       '  createdAt: 1757951447481,\n'
                       '  name: esaxo-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esaxo-perf1',
  'source_node_id': '580',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591337,\n'
                       '  createdAt: 1757951447481,\n'
                       '  name: esaxo-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esaxo-perf1',
  'source_node_id': '580',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591337,\n'
                       '  createdAt: 1757951447481,\n'
                       '  name: esaxo-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ias1-dev',
  'source_node_id': '581',
  'source_properties': '{\n'
                       '  updatedAt: 1758632588993,\n'
                       '  createdAt: 1757951447612,\n'
                       '  name: ias1-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'ionru00',
  'source_node_id': '582',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589018,\n'
                       '  createdAt: 1757951447644,\n'
                       '  name: ionru00\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'k6-operator-system',
  'source_node_id': '583',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589068,\n'
                       '  createdAt: 1757951447707,\n'
                       '  name: k6-operator-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'kafka',
  'source_node_id': '584',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589093,\n'
                       '  createdAt: 1757951447742,\n'
                       '  name: kafka\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'nmana11',
  'source_node_id': '585',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589119,\n'
                       '  createdAt: 1757951447774,\n'
                       '  name: nmana11\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'nmana11-dev',
  'source_node_id': '586',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589144,\n'
                       '  createdAt: 1757951447805,\n'
                       '  name: nmana11-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'opa-exporter',
  'source_node_id': '587',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589169,\n'
                       '  createdAt: 1757951447838,\n'
                       '  name: opa-exporter\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'rlitr00-dev',
  'source_node_id': '588',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589195,\n'
                       '  createdAt: 1757951447873,\n'
                       '  name: rlitr00-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sd00003',
  'source_node_id': '589',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589220,\n'
                       '  createdAt: 1757951447904,\n'
                       '  name: sd00003\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'sguru22-03',
  'source_node_id': '590',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589245,\n'
                       '  createdAt: 1757951447936,\n'
                       '  name: sguru22-03\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'test-pilot',
  'source_node_id': '591',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589270,\n'
                       '  createdAt: 1757951447967,\n'
                       '  name: test-pilot\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'topologytest',
  'source_node_id': '592',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589295,\n'
                       '  createdAt: 1757951447997,\n'
                       '  name: topologytest\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'vmadi00',
  'source_node_id': '593',
  'source_properties': '{\n'
                       '  updatedAt: 1758632589351,\n'
                       '  createdAt: 1757951448059,\n'
                       '  name: vmadi00\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-akssb-nonprod-westus-cluster-01',
  'target_node_id': '13',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571531,\n'
                       '  createdAt: 1757951420970,\n'
                       '  name: esco-akssb-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aamp-perf1',
  'source_node_id': '594',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590610,\n'
                       '  createdAt: 1757951448095,\n'
                       '  name: aamp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-eastus-cluster-01',
  'target_node_id': '14',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571556,\n'
                       '  createdAt: 1757951421007,\n'
                       '  name: esco-aksshared-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aamp-perf1',
  'source_node_id': '594',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590610,\n'
                       '  createdAt: 1757951448095,\n'
                       '  name: aamp-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aamp-dev',
  'source_node_id': '595',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590583,\n'
                       '  createdAt: 1757951449779,\n'
                       '  name: aamp-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'aamp-qa1',
  'source_node_id': '596',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590635,\n'
                       '  createdAt: 1757951449846,\n'
                       '  name: aamp-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'egcg-dev',
  'source_node_id': '597',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590777,\n'
                       '  createdAt: 1757951450018,\n'
                       '  name: egcg-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'egcg-stage',
  'source_node_id': '598',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590803,\n'
                       '  createdAt: 1757951450057,\n'
                       '  name: egcg-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'egkc-dev',
  'source_node_id': '599',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590829,\n'
                       '  createdAt: 1757951450096,\n'
                       '  name: egkc-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'egkc-stage',
  'source_node_id': '600',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590855,\n'
                       '  createdAt: 1757951450137,\n'
                       '  name: egkc-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esaa-dev',
  'source_node_id': '601',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590881,\n'
                       '  createdAt: 1757951450169,\n'
                       '  name: esaa-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esaq-dev',
  'source_node_id': '602',
  'source_properties': '{\n'
                       '  updatedAt: 1758632590907,\n'
                       '  createdAt: 1757951450210,\n'
                       '  name: esaq-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esax-dev',
  'source_node_id': '603',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591232,\n'
                       '  createdAt: 1757951450242,\n'
                       '  name: esax-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esax-qa1',
  'source_node_id': '604',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591258,\n'
                       '  createdAt: 1757951450274,\n'
                       '  name: esax-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esaxo-dev1',
  'source_node_id': '605',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591284,\n'
                       '  createdAt: 1757951450306,\n'
                       '  name: esaxo-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esaxo-dev2',
  'source_node_id': '606',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591311,\n'
                       '  createdAt: 1757951450341,\n'
                       '  name: esaxo-dev2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-dev',
  'source_node_id': '607',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591363,\n'
                       '  createdAt: 1757951450406,\n'
                       '  name: escf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'escf-qa1',
  'source_node_id': '608',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591416,\n'
                       '  createdAt: 1757951450468,\n'
                       '  name: escf-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esff-dev',
  'source_node_id': '609',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591442,\n'
                       '  createdAt: 1757951450522,\n'
                       '  name: esff-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esff-int',
  'source_node_id': '610',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591467,\n'
                       '  createdAt: 1757951450554,\n'
                       '  name: esff-int\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esff-qa1',
  'source_node_id': '611',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591493,\n'
                       '  createdAt: 1757951450585,\n'
                       '  name: esff-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esff-stg',
  'source_node_id': '612',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591518,\n'
                       '  createdAt: 1757951450616,\n'
                       '  name: esff-stg\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esgh-dev1',
  'source_node_id': '613',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591570,\n'
                       '  createdAt: 1757951450694,\n'
                       '  name: esgh-dev1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esgh-dev2',
  'source_node_id': '614',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591599,\n'
                       '  createdAt: 1757951450728,\n'
                       '  name: esgh-dev2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esgh-dev3',
  'source_node_id': '615',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591624,\n'
                       '  createdAt: 1757951450766,\n'
                       '  name: esgh-dev3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esgh-stage',
  'source_node_id': '616',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591651,\n'
                       '  createdAt: 1757951450800,\n'
                       '  name: esgh-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esghd-dev',
  'source_node_id': '617',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591676,\n'
                       '  createdAt: 1757951450836,\n'
                       '  name: esghd-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esghd-stage',
  'source_node_id': '618',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591702,\n'
                       '  createdAt: 1757951450870,\n'
                       '  name: esghd-stage\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobf-dev',
  'source_node_id': '619',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592291,\n'
                       '  createdAt: 1757951450935,\n'
                       '  name: esobf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobf-dev',
  'source_node_id': '619',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592291,\n'
                       '  createdAt: 1757951450935,\n'
                       '  name: esobf-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobf-perf1',
  'source_node_id': '620',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592316,\n'
                       '  createdAt: 1757951450968,\n'
                       '  name: esobf-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobf-perf1',
  'source_node_id': '620',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592316,\n'
                       '  createdAt: 1757951450968,\n'
                       '  name: esobf-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'iacspm-dev',
  'source_node_id': '621',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591832,\n'
                       '  createdAt: 1757951451031,\n'
                       '  name: iacspm-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'oauth2-proxy',
  'source_node_id': '622',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591885,\n'
                       '  createdAt: 1757951451101,\n'
                       '  name: oauth2-proxy\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'osca-dev',
  'source_node_id': '623',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591911,\n'
                       '  createdAt: 1757951451133,\n'
                       '  name: osca-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'scob-qa1',
  'source_node_id': '624',
  'source_properties': '{\n'
                       '  updatedAt: 1758632591936,\n'
                       '  createdAt: 1757951451167,\n'
                       '  name: scob-qa1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-01',
  'target_node_id': '19',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571683,\n'
                       '  createdAt: 1757951421209,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'app-live-view-connector',
  'source_node_id': '625',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592089,\n'
                       '  createdAt: 1757951451373,\n'
                       '  name: app-live-view-connector\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esob-dev',
  'source_node_id': '626',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592165,\n'
                       '  createdAt: 1757951451504,\n'
                       '  name: esob-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esoba-dev',
  'source_node_id': '627',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592216,\n'
                       '  createdAt: 1757951451571,\n'
                       '  name: esoba-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobai-dev',
  'source_node_id': '628',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592241,\n'
                       '  createdAt: 1757951451604,\n'
                       '  name: esobai-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'flux-system',
  'source_node_id': '629',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592341,\n'
                       '  createdAt: 1757951451709,\n'
                       '  name: flux-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'github-dev',
  'source_node_id': '630',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592392,\n'
                       '  createdAt: 1757951451774,\n'
                       '  name: github-dev\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'scan-link-system',
  'source_node_id': '631',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592444,\n'
                       '  createdAt: 1757951451838,\n'
                       '  name: scan-link-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'source-system',
  'source_node_id': '632',
  'source_properties': '{\n'
                       '  updatedAt: 1758632592469,\n'
                       '  createdAt: 1757951451871,\n'
                       '  name: source-system\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-02',
  'target_node_id': '20',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571710,\n'
                       '  createdAt: 1757951421247,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-02\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobb-perf1',
  'source_node_id': '633',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593198,\n'
                       '  createdAt: 1757951452177,\n'
                       '  name: esobb-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-03',
  'target_node_id': '21',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571735,\n'
                       '  createdAt: 1757951421305,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-03\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobb-perf1',
  'source_node_id': '633',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593198,\n'
                       '  createdAt: 1757951452177,\n'
                       '  name: esobb-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esoba-perf1',
  'source_node_id': '634',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593172,\n'
                       '  createdAt: 1757951452841,\n'
                       '  name: esoba-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobc-perf1',
  'source_node_id': '635',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593223,\n'
                       '  createdAt: 1757951452919,\n'
                       '  name: esobc-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-04',
  'target_node_id': '22',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571761,\n'
                       '  createdAt: 1757951421342,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-04\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobc-perf1',
  'source_node_id': '635',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593223,\n'
                       '  createdAt: 1757951452919,\n'
                       '  name: esobc-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Namespace',
  'source_name': 'esobd-perf1',
  'source_node_id': '636',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593248,\n'
                       '  createdAt: 1757951452958,\n'
                       '  name: esobd-perf1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksshared-nonprod-westus-cluster-05',
  'target_node_id': '23',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571786,\n'
                       '  createdAt: 1757951421385,\n'
                       '  name: esco-aksshared-nonprod-westus-cluster-05\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-40976780-vmss00007m',
  'source_node_id': '637',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593354,\n'
                       '  createdAt: 1757951453101,\n'
                       '  name: aks-escoba1np01-40976780-vmss00007m\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-40976780-vmss00008j',
  'source_node_id': '638',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593381,\n'
                       '  createdAt: 1757951453260,\n'
                       '  name: aks-escoba1np01-40976780-vmss00008j\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-40976780-vmss00008y',
  'source_node_id': '639',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593412,\n'
                       '  createdAt: 1757951453293,\n'
                       '  name: aks-escoba1np01-40976780-vmss00008y\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-20500933-vmss00003n',
  'source_node_id': '640',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593441,\n'
                       '  createdAt: 1757951453325,\n'
                       '  name: aks-escoba1np02-20500933-vmss00003n\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-20500933-vmss00003o',
  'source_node_id': '641',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593467,\n'
                       '  createdAt: 1757951453356,\n'
                       '  name: aks-escoba1np02-20500933-vmss00003o\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-20500933-vmss00003s',
  'source_node_id': '642',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593509,\n'
                       '  createdAt: 1757951453388,\n'
                       '  name: aks-escoba1np02-20500933-vmss00003s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-40976780-vmss00008z',
  'source_node_id': '643',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593547,\n'
                       '  createdAt: 1757951453423,\n'
                       '  name: aks-escoba1np01-40976780-vmss00008z\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-40976780-vmss000095',
  'source_node_id': '644',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593573,\n'
                       '  createdAt: 1757951453462,\n'
                       '  name: aks-escoba1np01-40976780-vmss000095\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-40976780-vmss000096',
  'source_node_id': '645',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593599,\n'
                       '  createdAt: 1757951453494,\n'
                       '  name: aks-escoba1np01-40976780-vmss000096\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1npsys-39259977-vmss00003k',
  'source_node_id': '646',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593624,\n'
                       '  createdAt: 1757951453524,\n'
                       '  name: aks-escoba1npsys-39259977-vmss00003k\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-eastus-cluster-01',
  'target_node_id': '10',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571428,\n'
                       '  createdAt: 1757951420666,\n'
                       '  name: esco-aksba1-nonprod-eastus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss00037r',
  'source_node_id': '647',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593649,\n'
                       '  createdAt: 1757951453556,\n'
                       '  name: aks-escoba1np01-28933280-vmss00037r\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss00041w',
  'source_node_id': '648',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593675,\n'
                       '  createdAt: 1757951453613,\n'
                       '  name: aks-escoba1np01-28933280-vmss00041w\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0004xh',
  'source_node_id': '649',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593726,\n'
                       '  createdAt: 1757951453651,\n'
                       '  name: aks-escoba1np01-28933280-vmss0004xh\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0005fz',
  'source_node_id': '650',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593754,\n'
                       '  createdAt: 1757951453682,\n'
                       '  name: aks-escoba1np01-28933280-vmss0005fz\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss000623',
  'source_node_id': '651',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593779,\n'
                       '  createdAt: 1757951453714,\n'
                       '  name: aks-escoba1np01-28933280-vmss000623\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0007hl',
  'source_node_id': '652',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593805,\n'
                       '  createdAt: 1757951453745,\n'
                       '  name: aks-escoba1np01-28933280-vmss0007hl\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0007hs',
  'source_node_id': '653',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596517,\n'
                       '  createdAt: 1757951453781,\n'
                       '  name: aks-escoba1np01-28933280-vmss0007hs\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0002p6',
  'source_node_id': '654',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593831,\n'
                       '  createdAt: 1757951453816,\n'
                       '  name: aks-escoba1np02-13277147-vmss0002p6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0002ye',
  'source_node_id': '655',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593855,\n'
                       '  createdAt: 1757951453849,\n'
                       '  name: aks-escoba1np02-13277147-vmss0002ye\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0003pe',
  'source_node_id': '656',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593880,\n'
                       '  createdAt: 1757951453880,\n'
                       '  name: aks-escoba1np02-13277147-vmss0003pe\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0003px',
  'source_node_id': '657',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593905,\n'
                       '  createdAt: 1757951453913,\n'
                       '  name: aks-escoba1np02-13277147-vmss0003px\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004ew',
  'source_node_id': '658',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593931,\n'
                       '  createdAt: 1757951453947,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004ew\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004tk',
  'source_node_id': '659',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593955,\n'
                       '  createdAt: 1757951453979,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004tk\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss0001ry',
  'source_node_id': '660',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593981,\n'
                       '  createdAt: 1757951454011,\n'
                       '  name: aks-escoba1np03-25879678-vmss0001ry\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss000251',
  'source_node_id': '661',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594017,\n'
                       '  createdAt: 1757951454041,\n'
                       '  name: aks-escoba1np03-25879678-vmss000251\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss00025a',
  'source_node_id': '662',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594052,\n'
                       '  createdAt: 1757951454072,\n'
                       '  name: aks-escoba1np03-25879678-vmss00025a\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss00026f',
  'source_node_id': '663',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594078,\n'
                       '  createdAt: 1757951454104,\n'
                       '  name: aks-escoba1np03-25879678-vmss00026f\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss00026i',
  'source_node_id': '664',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594104,\n'
                       '  createdAt: 1757951454136,\n'
                       '  name: aks-escoba1np03-25879678-vmss00026i\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss0002tx',
  'source_node_id': '665',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594130,\n'
                       '  createdAt: 1757951454167,\n'
                       '  name: aks-escoba1np03-25879678-vmss0002tx\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0000no',
  'source_node_id': '666',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594157,\n'
                       '  createdAt: 1757951454198,\n'
                       '  name: aks-escoba1np04-28230666-vmss0000no\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss00013p',
  'source_node_id': '667',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594183,\n'
                       '  createdAt: 1757951454227,\n'
                       '  name: aks-escoba1np04-28230666-vmss00013p\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss00015m',
  'source_node_id': '668',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594208,\n'
                       '  createdAt: 1757951454260,\n'
                       '  name: aks-escoba1np04-28230666-vmss00015m\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0001uq',
  'source_node_id': '669',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594235,\n'
                       '  createdAt: 1757951454291,\n'
                       '  name: aks-escoba1np04-28230666-vmss0001uq\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss00035a',
  'source_node_id': '670',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594260,\n'
                       '  createdAt: 1757951454323,\n'
                       '  name: aks-escoba1np04-28230666-vmss00035a\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00000y',
  'source_node_id': '671',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594313,\n'
                       '  createdAt: 1757951454354,\n'
                       '  name: aks-escoba1np05-29664909-vmss00000y\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss000074',
  'source_node_id': '672',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594350,\n'
                       '  createdAt: 1757951454391,\n'
                       '  name: aks-escoba1np05-29664909-vmss000074\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000ix',
  'source_node_id': '673',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594376,\n'
                       '  createdAt: 1757951454427,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000ix\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000k7',
  'source_node_id': '674',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594402,\n'
                       '  createdAt: 1757951454468,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000k7\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00027k',
  'source_node_id': '675',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594457,\n'
                       '  createdAt: 1757951454499,\n'
                       '  name: aks-escoba1np05-29664909-vmss00027k\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00030b',
  'source_node_id': '676',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594483,\n'
                       '  createdAt: 1757951454534,\n'
                       '  name: aks-escoba1np05-29664909-vmss00030b\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss000333',
  'source_node_id': '677',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594509,\n'
                       '  createdAt: 1757951454566,\n'
                       '  name: aks-escoba1np05-29664909-vmss000333\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0004cr',
  'source_node_id': '678',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594534,\n'
                       '  createdAt: 1757951454597,\n'
                       '  name: aks-escoba1np05-29664909-vmss0004cr\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00050w',
  'source_node_id': '679',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597030,\n'
                       '  createdAt: 1757951454628,\n'
                       '  name: aks-escoba1np05-29664909-vmss00050w\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00052p',
  'source_node_id': '680',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594560,\n'
                       '  createdAt: 1757951454659,\n'
                       '  name: aks-escoba1np05-29664909-vmss00052p\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0001el',
  'source_node_id': '681',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594586,\n'
                       '  createdAt: 1757951454697,\n'
                       '  name: aks-escoba1np01-28933280-vmss0001el\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0001k1',
  'source_node_id': '682',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594612,\n'
                       '  createdAt: 1757951454740,\n'
                       '  name: aks-escoba1np01-28933280-vmss0001k1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0002hr',
  'source_node_id': '683',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594641,\n'
                       '  createdAt: 1757951454779,\n'
                       '  name: aks-escoba1np01-28933280-vmss0002hr\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss00033g',
  'source_node_id': '684',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594666,\n'
                       '  createdAt: 1757951454823,\n'
                       '  name: aks-escoba1np01-28933280-vmss00033g\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss00037l',
  'source_node_id': '685',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594693,\n'
                       '  createdAt: 1757951454861,\n'
                       '  name: aks-escoba1np01-28933280-vmss00037l\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss000384',
  'source_node_id': '686',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594718,\n'
                       '  createdAt: 1757951454898,\n'
                       '  name: aks-escoba1np01-28933280-vmss000384\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0003d5',
  'source_node_id': '687',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594744,\n'
                       '  createdAt: 1757951454932,\n'
                       '  name: aks-escoba1np01-28933280-vmss0003d5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0003u4',
  'source_node_id': '688',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594770,\n'
                       '  createdAt: 1757951454967,\n'
                       '  name: aks-escoba1np01-28933280-vmss0003u4\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss00041l',
  'source_node_id': '689',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594796,\n'
                       '  createdAt: 1757951455000,\n'
                       '  name: aks-escoba1np01-28933280-vmss00041l\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss000437',
  'source_node_id': '690',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594824,\n'
                       '  createdAt: 1757951455034,\n'
                       '  name: aks-escoba1np01-28933280-vmss000437\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0004k5',
  'source_node_id': '691',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594852,\n'
                       '  createdAt: 1757951455066,\n'
                       '  name: aks-escoba1np01-28933280-vmss0004k5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0004lg',
  'source_node_id': '692',
  'source_properties': '{\n'
                       '  updatedAt: 1758632593700,\n'
                       '  createdAt: 1757951455098,\n'
                       '  name: aks-escoba1np01-28933280-vmss0004lg\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0007hd',
  'source_node_id': '693',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594879,\n'
                       '  createdAt: 1757951455129,\n'
                       '  name: aks-escoba1np01-28933280-vmss0007hd\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0001or',
  'source_node_id': '694',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594905,\n'
                       '  createdAt: 1757951455162,\n'
                       '  name: aks-escoba1np02-13277147-vmss0001or\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0002lx',
  'source_node_id': '695',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594931,\n'
                       '  createdAt: 1757951455195,\n'
                       '  name: aks-escoba1np02-13277147-vmss0002lx\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0002og',
  'source_node_id': '696',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594963,\n'
                       '  createdAt: 1757951455300,\n'
                       '  name: aks-escoba1np02-13277147-vmss0002og\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0002t1',
  'source_node_id': '697',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594993,\n'
                       '  createdAt: 1757951455373,\n'
                       '  name: aks-escoba1np02-13277147-vmss0002t1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0002xj',
  'source_node_id': '698',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595020,\n'
                       '  createdAt: 1757951455410,\n'
                       '  name: aks-escoba1np02-13277147-vmss0002xj\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0003om',
  'source_node_id': '699',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595045,\n'
                       '  createdAt: 1757951455444,\n'
                       '  name: aks-escoba1np02-13277147-vmss0003om\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0003vn',
  'source_node_id': '700',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595072,\n'
                       '  createdAt: 1757951455477,\n'
                       '  name: aks-escoba1np02-13277147-vmss0003vn\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0003vs',
  'source_node_id': '701',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595104,\n'
                       '  createdAt: 1757951455508,\n'
                       '  name: aks-escoba1np02-13277147-vmss0003vs\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0003w8',
  'source_node_id': '702',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595130,\n'
                       '  createdAt: 1757951455541,\n'
                       '  name: aks-escoba1np02-13277147-vmss0003w8\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00041x',
  'source_node_id': '703',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595155,\n'
                       '  createdAt: 1757951455574,\n'
                       '  name: aks-escoba1np02-13277147-vmss00041x\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss000438',
  'source_node_id': '704',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595181,\n'
                       '  createdAt: 1757951455606,\n'
                       '  name: aks-escoba1np02-13277147-vmss000438\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00045y',
  'source_node_id': '705',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595206,\n'
                       '  createdAt: 1757951455642,\n'
                       '  name: aks-escoba1np02-13277147-vmss00045y\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss000464',
  'source_node_id': '706',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595234,\n'
                       '  createdAt: 1757951455675,\n'
                       '  name: aks-escoba1np02-13277147-vmss000464\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004ls',
  'source_node_id': '707',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595262,\n'
                       '  createdAt: 1757951455707,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004ls\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004m6',
  'source_node_id': '708',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595292,\n'
                       '  createdAt: 1757951455739,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004m6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004sg',
  'source_node_id': '709',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595318,\n'
                       '  createdAt: 1757951455770,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004sg\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004sn',
  'source_node_id': '710',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595344,\n'
                       '  createdAt: 1757951455802,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004sn\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004tv',
  'source_node_id': '711',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595370,\n'
                       '  createdAt: 1757951455834,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004tv\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss0001iu',
  'source_node_id': '712',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595396,\n'
                       '  createdAt: 1757951455866,\n'
                       '  name: aks-escoba1np03-25879678-vmss0001iu\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss000227',
  'source_node_id': '713',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595422,\n'
                       '  createdAt: 1757951455898,\n'
                       '  name: aks-escoba1np03-25879678-vmss000227\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss00024q',
  'source_node_id': '714',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595447,\n'
                       '  createdAt: 1757951455929,\n'
                       '  name: aks-escoba1np03-25879678-vmss00024q\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss000258',
  'source_node_id': '715',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595473,\n'
                       '  createdAt: 1757951455961,\n'
                       '  name: aks-escoba1np03-25879678-vmss000258\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss00025c',
  'source_node_id': '716',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595500,\n'
                       '  createdAt: 1757951455994,\n'
                       '  name: aks-escoba1np03-25879678-vmss00025c\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss00025l',
  'source_node_id': '717',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595532,\n'
                       '  createdAt: 1757951456026,\n'
                       '  name: aks-escoba1np03-25879678-vmss00025l\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss000265',
  'source_node_id': '718',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595559,\n'
                       '  createdAt: 1757951456058,\n'
                       '  name: aks-escoba1np03-25879678-vmss000265\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss0002ei',
  'source_node_id': '719',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595584,\n'
                       '  createdAt: 1757951456091,\n'
                       '  name: aks-escoba1np03-25879678-vmss0002ei\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss0002tt',
  'source_node_id': '720',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595613,\n'
                       '  createdAt: 1757951456124,\n'
                       '  name: aks-escoba1np03-25879678-vmss0002tt\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0000fn',
  'source_node_id': '721',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595642,\n'
                       '  createdAt: 1757951456156,\n'
                       '  name: aks-escoba1np04-28230666-vmss0000fn\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0000rf',
  'source_node_id': '722',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595669,\n'
                       '  createdAt: 1757951456187,\n'
                       '  name: aks-escoba1np04-28230666-vmss0000rf\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0000w8',
  'source_node_id': '723',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595695,\n'
                       '  createdAt: 1757951456219,\n'
                       '  name: aks-escoba1np04-28230666-vmss0000w8\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0000w9',
  'source_node_id': '724',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595719,\n'
                       '  createdAt: 1757951456250,\n'
                       '  name: aks-escoba1np04-28230666-vmss0000w9\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss00011x',
  'source_node_id': '725',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595747,\n'
                       '  createdAt: 1757951456282,\n'
                       '  name: aks-escoba1np04-28230666-vmss00011x\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0001vk',
  'source_node_id': '726',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595773,\n'
                       '  createdAt: 1757951456316,\n'
                       '  name: aks-escoba1np04-28230666-vmss0001vk\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss000338',
  'source_node_id': '727',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595799,\n'
                       '  createdAt: 1757951456349,\n'
                       '  name: aks-escoba1np04-28230666-vmss000338\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00000f',
  'source_node_id': '728',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595824,\n'
                       '  createdAt: 1757951456381,\n'
                       '  name: aks-escoba1np05-29664909-vmss00000f\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000df',
  'source_node_id': '729',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595850,\n'
                       '  createdAt: 1757951456415,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000df\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000ew',
  'source_node_id': '730',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595880,\n'
                       '  createdAt: 1757951456447,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000ew\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000f5',
  'source_node_id': '731',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595906,\n'
                       '  createdAt: 1757951456481,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000f5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000vh',
  'source_node_id': '732',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594431,\n'
                       '  createdAt: 1757951456517,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000vh\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00047o',
  'source_node_id': '733',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595932,\n'
                       '  createdAt: 1757951456548,\n'
                       '  name: aks-escoba1np05-29664909-vmss00047o\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00050v',
  'source_node_id': '734',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595960,\n'
                       '  createdAt: 1757951456589,\n'
                       '  name: aks-escoba1np05-29664909-vmss00050v\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00052j',
  'source_node_id': '735',
  'source_properties': '{\n'
                       '  updatedAt: 1758632595986,\n'
                       '  createdAt: 1757951456629,\n'
                       '  name: aks-escoba1np05-29664909-vmss00052j\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss000532',
  'source_node_id': '736',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596012,\n'
                       '  createdAt: 1757951456658,\n'
                       '  name: aks-escoba1np05-29664909-vmss000532\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss00033j',
  'source_node_id': '737',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596039,\n'
                       '  createdAt: 1757951456690,\n'
                       '  name: aks-escoba1np01-28933280-vmss00033j\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0007h5',
  'source_node_id': '738',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596066,\n'
                       '  createdAt: 1757951456723,\n'
                       '  name: aks-escoba1np01-28933280-vmss0007h5\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00040e',
  'source_node_id': '739',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596092,\n'
                       '  createdAt: 1757951456755,\n'
                       '  name: aks-escoba1np02-13277147-vmss00040e\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004qn',
  'source_node_id': '740',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596117,\n'
                       '  createdAt: 1757951456788,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004qn\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00069g',
  'source_node_id': '741',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596145,\n'
                       '  createdAt: 1757951456820,\n'
                       '  name: aks-escoba1np02-13277147-vmss00069g\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss00018z',
  'source_node_id': '742',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596178,\n'
                       '  createdAt: 1757951456853,\n'
                       '  name: aks-escoba1np04-28230666-vmss00018z\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss0001vg',
  'source_node_id': '743',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596204,\n'
                       '  createdAt: 1757951456884,\n'
                       '  name: aks-escoba1np04-28230666-vmss0001vg\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00052r',
  'source_node_id': '744',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596230,\n'
                       '  createdAt: 1757951456916,\n'
                       '  name: aks-escoba1np05-29664909-vmss00052r\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0007hk',
  'source_node_id': '745',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596281,\n'
                       '  createdAt: 1757951456951,\n'
                       '  name: aks-escoba1np01-28933280-vmss0007hk\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss000453',
  'source_node_id': '746',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596306,\n'
                       '  createdAt: 1757951456981,\n'
                       '  name: aks-escoba1np02-13277147-vmss000453\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00068u',
  'source_node_id': '747',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596331,\n'
                       '  createdAt: 1757951457018,\n'
                       '  name: aks-escoba1np02-13277147-vmss00068u\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss00035b',
  'source_node_id': '748',
  'source_properties': '{\n'
                       '  updatedAt: 1758632594287,\n'
                       '  createdAt: 1757951457057,\n'
                       '  name: aks-escoba1np04-28230666-vmss00035b\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss00035r',
  'source_node_id': '749',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596904,\n'
                       '  createdAt: 1757951457089,\n'
                       '  name: aks-escoba1np04-28230666-vmss00035r\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0004c8',
  'source_node_id': '750',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596384,\n'
                       '  createdAt: 1757951457119,\n'
                       '  name: aks-escoba1np05-29664909-vmss0004c8\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00053b',
  'source_node_id': '751',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596256,\n'
                       '  createdAt: 1757951457152,\n'
                       '  name: aks-escoba1np05-29664909-vmss00053b\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0003wi',
  'source_node_id': '752',
  'source_properties': '{\n'
                       '  updatedAt: 1758036290319,\n'
                       '  createdAt: 1757951457182,\n'
                       '  name: aks-escoba1np01-28933280-vmss0003wi\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0003x2',
  'source_node_id': '753',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596439,\n'
                       '  createdAt: 1757951457213,\n'
                       '  name: aks-escoba1np01-28933280-vmss0003x2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0007h0',
  'source_node_id': '754',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596464,\n'
                       '  createdAt: 1757951457245,\n'
                       '  name: aks-escoba1np01-28933280-vmss0007h0\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np01-28933280-vmss0007h7',
  'source_node_id': '755',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596489,\n'
                       '  createdAt: 1757951457278,\n'
                       '  name: aks-escoba1np01-28933280-vmss0007h7\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00045e',
  'source_node_id': '756',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596621,\n'
                       '  createdAt: 1757951457309,\n'
                       '  name: aks-escoba1np02-13277147-vmss00045e\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss0004ey',
  'source_node_id': '757',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596647,\n'
                       '  createdAt: 1757951457356,\n'
                       '  name: aks-escoba1np02-13277147-vmss0004ey\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00063y',
  'source_node_id': '758',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596672,\n'
                       '  createdAt: 1757951457387,\n'
                       '  name: aks-escoba1np02-13277147-vmss00063y\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00066i',
  'source_node_id': '759',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596698,\n'
                       '  createdAt: 1757951457418,\n'
                       '  name: aks-escoba1np02-13277147-vmss00066i\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00068z',
  'source_node_id': '760',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596724,\n'
                       '  createdAt: 1757951457449,\n'
                       '  name: aks-escoba1np02-13277147-vmss00068z\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np02-13277147-vmss00069f',
  'source_node_id': '761',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596749,\n'
                       '  createdAt: 1757951457481,\n'
                       '  name: aks-escoba1np02-13277147-vmss00069f\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss000183',
  'source_node_id': '762',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596825,\n'
                       '  createdAt: 1757951457514,\n'
                       '  name: aks-escoba1np03-25879678-vmss000183\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np03-25879678-vmss0002u3',
  'source_node_id': '763',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596853,\n'
                       '  createdAt: 1757951457544,\n'
                       '  name: aks-escoba1np03-25879678-vmss0002u3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np04-28230666-vmss000358',
  'source_node_id': '764',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596878,\n'
                       '  createdAt: 1757951457577,\n'
                       '  name: aks-escoba1np04-28230666-vmss000358\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000fz',
  'source_node_id': '765',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596929,\n'
                       '  createdAt: 1757951457614,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000fz\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000k4',
  'source_node_id': '766',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596954,\n'
                       '  createdAt: 1757951457646,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000k4\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0000kc',
  'source_node_id': '767',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596979,\n'
                       '  createdAt: 1757951457677,\n'
                       '  name: aks-escoba1np05-29664909-vmss0000kc\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0002wz',
  'source_node_id': '768',
  'source_properties': '{\n'
                       '  updatedAt: 1758632596358,\n'
                       '  createdAt: 1757951457714,\n'
                       '  name: aks-escoba1np05-29664909-vmss0002wz\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss0004of',
  'source_node_id': '769',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597005,\n'
                       '  createdAt: 1757951457747,\n'
                       '  name: aks-escoba1np05-29664909-vmss0004of\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss000514',
  'source_node_id': '770',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597056,\n'
                       '  createdAt: 1757951457779,\n'
                       '  name: aks-escoba1np05-29664909-vmss000514\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00052s',
  'source_node_id': '771',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597081,\n'
                       '  createdAt: 1757951457809,\n'
                       '  name: aks-escoba1np05-29664909-vmss00052s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00052q',
  'source_node_id': '772',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597185,\n'
                       '  createdAt: 1757951457842,\n'
                       '  name: aks-escoba1np05-29664909-vmss00052q\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00052t',
  'source_node_id': '773',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597107,\n'
                       '  createdAt: 1757951457877,\n'
                       '  name: aks-escoba1np05-29664909-vmss00052t\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00052u',
  'source_node_id': '774',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597212,\n'
                       '  createdAt: 1757951457909,\n'
                       '  name: aks-escoba1np05-29664909-vmss00052u\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1np05-29664909-vmss00053s',
  'source_node_id': '775',
  'source_properties': '{\n'
                       '  updatedAt: 1757954048325,\n'
                       '  createdAt: 1757951457940,\n'
                       '  name: aks-escoba1np05-29664909-vmss00053s\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1npsys-38620034-vmss00000n',
  'source_node_id': '776',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597237,\n'
                       '  createdAt: 1757951457987,\n'
                       '  name: aks-escoba1npsys-38620034-vmss00000n\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1npsys-38620034-vmss00000q',
  'source_node_id': '777',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597265,\n'
                       '  createdAt: 1757951458021,\n'
                       '  name: aks-escoba1npsys-38620034-vmss00000q\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba1npsys-38620034-vmss00000r',
  'source_node_id': '778',
  'source_properties': '{\n'
                       '  updatedAt: 1758632597298,\n'
                       '  createdAt: 1757951458052,\n'
                       '  name: aks-escoba1npsys-38620034-vmss00000r\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba1-nonprod-westus-cluster-01',
  'target_node_id': '11',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571453,\n'
                       '  createdAt: 1757951420895,\n'
                       '  name: esco-aksba1-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np03-28707821-vmss0000bv',
  'source_node_id': '779',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598550,\n'
                       '  createdAt: 1757951458088,\n'
                       '  name: aks-escoba2np03-28707821-vmss0000bv\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss00003a',
  'source_node_id': '780',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598577,\n'
                       '  createdAt: 1757951458121,\n'
                       '  name: aks-escoba2np01-12560122-vmss00003a\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss00003b',
  'source_node_id': '781',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598603,\n'
                       '  createdAt: 1757951458152,\n'
                       '  name: aks-escoba2np01-12560122-vmss00003b\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss00003e',
  'source_node_id': '782',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598629,\n'
                       '  createdAt: 1757951458186,\n'
                       '  name: aks-escoba2np01-12560122-vmss00003e\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss00003f',
  'source_node_id': '783',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598655,\n'
                       '  createdAt: 1757951458216,\n'
                       '  name: aks-escoba2np01-12560122-vmss00003f\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss00003m',
  'source_node_id': '784',
  'source_properties': '{\n'
                       '  updatedAt: 1758183690301,\n'
                       '  createdAt: 1757951458247,\n'
                       '  name: aks-escoba2np01-12560122-vmss00003m\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss00006h',
  'source_node_id': '785',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598681,\n'
                       '  createdAt: 1757951458277,\n'
                       '  name: aks-escoba2np01-12560122-vmss00006h\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss00009v',
  'source_node_id': '786',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598707,\n'
                       '  createdAt: 1757951458313,\n'
                       '  name: aks-escoba2np01-12560122-vmss00009v\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000bf',
  'source_node_id': '787',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598734,\n'
                       '  createdAt: 1757951458344,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000bf\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000bn',
  'source_node_id': '788',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598759,\n'
                       '  createdAt: 1757951458378,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000bn\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000c7',
  'source_node_id': '789',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598785,\n'
                       '  createdAt: 1757951458408,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000c7\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000cb',
  'source_node_id': '790',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598812,\n'
                       '  createdAt: 1757951458439,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000cb\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000ce',
  'source_node_id': '791',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598845,\n'
                       '  createdAt: 1757951458478,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000ce\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000cy',
  'source_node_id': '792',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598871,\n'
                       '  createdAt: 1757951458510,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000cy\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000da',
  'source_node_id': '793',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598897,\n'
                       '  createdAt: 1757951458561,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000da\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000db',
  'source_node_id': '794',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598924,\n'
                       '  createdAt: 1757951458597,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000db\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000df',
  'source_node_id': '795',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598949,\n'
                       '  createdAt: 1757951458673,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000df\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000ei',
  'source_node_id': '796',
  'source_properties': '{\n'
                       '  updatedAt: 1758632598975,\n'
                       '  createdAt: 1757951458706,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000ei\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000eo',
  'source_node_id': '797',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599000,\n'
                       '  createdAt: 1757951458738,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000eo\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000ep',
  'source_node_id': '798',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599030,\n'
                       '  createdAt: 1757951458769,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000ep\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000es',
  'source_node_id': '799',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599056,\n'
                       '  createdAt: 1757951458801,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000es\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000ex',
  'source_node_id': '800',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599084,\n'
                       '  createdAt: 1757951458832,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000ex\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000f1',
  'source_node_id': '801',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599109,\n'
                       '  createdAt: 1757951458866,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000f1\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000f2',
  'source_node_id': '802',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599134,\n'
                       '  createdAt: 1757951458900,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000f2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np01-12560122-vmss0000f6',
  'source_node_id': '803',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599160,\n'
                       '  createdAt: 1757951458932,\n'
                       '  name: aks-escoba2np01-12560122-vmss0000f6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss00003d',
  'source_node_id': '804',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599186,\n'
                       '  createdAt: 1757951458963,\n'
                       '  name: aks-escoba2np02-27420733-vmss00003d\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss00009d',
  'source_node_id': '805',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599211,\n'
                       '  createdAt: 1757951458993,\n'
                       '  name: aks-escoba2np02-27420733-vmss00009d\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000av',
  'source_node_id': '806',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599238,\n'
                       '  createdAt: 1757951459030,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000av\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000b6',
  'source_node_id': '807',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599264,\n'
                       '  createdAt: 1757951459060,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000b6\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000cg',
  'source_node_id': '808',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599289,\n'
                       '  createdAt: 1757951459092,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000cg\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000dd',
  'source_node_id': '809',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599318,\n'
                       '  createdAt: 1757951459123,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000dd\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000dh',
  'source_node_id': '810',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599343,\n'
                       '  createdAt: 1757951459158,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000dh\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000ef',
  'source_node_id': '811',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599368,\n'
                       '  createdAt: 1757951459189,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000ef\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000ez',
  'source_node_id': '812',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599393,\n'
                       '  createdAt: 1757951459219,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000ez\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000f2',
  'source_node_id': '813',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599419,\n'
                       '  createdAt: 1757951459252,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000f2\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000fn',
  'source_node_id': '814',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599444,\n'
                       '  createdAt: 1757951459282,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000fn\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000fy',
  'source_node_id': '815',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599470,\n'
                       '  createdAt: 1757951459314,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000fy\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000g4',
  'source_node_id': '816',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599495,\n'
                       '  createdAt: 1757951459349,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000g4\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000gj',
  'source_node_id': '817',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599521,\n'
                       '  createdAt: 1757951459381,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000gj\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000gt',
  'source_node_id': '818',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599547,\n'
                       '  createdAt: 1757951459413,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000gt\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000h3',
  'source_node_id': '819',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599572,\n'
                       '  createdAt: 1757951459444,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000h3\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000hb',
  'source_node_id': '820',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599598,\n'
                       '  createdAt: 1757951459477,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000hb\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000hc',
  'source_node_id': '821',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599623,\n'
                       '  createdAt: 1757951459508,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000hc\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000hj',
  'source_node_id': '822',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599648,\n'
                       '  createdAt: 1757951459538,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000hj\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000hk',
  'source_node_id': '823',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599674,\n'
                       '  createdAt: 1757951459568,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000hk\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000hm',
  'source_node_id': '824',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599699,\n'
                       '  createdAt: 1757951459599,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000hm\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'},
 {'relationship_properties': '{}',
  'relationship_type': 'BELONGS_TO',
  'source_label': 'Node',
  'source_name': 'aks-escoba2np02-27420733-vmss0000ho',
  'source_node_id': '825',
  'source_properties': '{\n'
                       '  updatedAt: 1758632599725,\n'
                       '  createdAt: 1757951459631,\n'
                       '  name: aks-escoba2np02-27420733-vmss0000ho\n'
                       '}',
  'target_label': 'Cluster',
  'target_name': 'esco-aksba2-nonprod-westus-cluster-01',
  'target_node_id': '12',
  'target_properties': '{\n'
                       '  updatedAt: 1758632571506,\n'
                       '  createdAt: 1757951420933,\n'
                       '  name: esco-aksba2-nonprod-westus-cluster-01\n'
                       '}'}]



