
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


