"""
anomaly_detection.py
---------------------
Trains an Isolation Forest model to flag incidents as Normal or Anomaly,
based purely on the numeric system metrics (cpu, memory, response time,
error count, network latency).
"""

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import NUMERIC_COLUMNS


def train_anomaly_model(df, contamination=0.18):
    """
    Train an Isolation Forest on the numeric feature columns.

    Returns the fitted model, the fitted scaler, and the dataframe with
    two new columns added: 'anomaly_label' and 'anomaly_score'.
    """
    X = df[NUMERIC_COLUMNS].values

    # Scale features so no single metric (e.g. response_time in ms)
    # dominates purely because of its numeric range.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_scaled)

    # -1 = anomaly, 1 = normal (sklearn convention) -> map to readable labels
    raw_predictions = model.predict(X_scaled)
    df = df.copy()
    df["anomaly_label"] = ["Anomaly" if p == -1 else "Normal" for p in raw_predictions]

    # decision_function: higher = more normal, lower/negative = more anomalous.
    # We flip the sign so a HIGHER anomaly_score means MORE anomalous,
    # which is more intuitive to show in the UI.
    df["anomaly_score"] = -model.decision_function(X_scaled)

    return model, scaler, df


def explain_anomaly(row, df):
    """
    Return a short human-readable explanation for why a given incident
    row might be considered abnormal, by comparing it to the dataset's
    average ("normal") values.
    """
    reasons = []
    averages = df[NUMERIC_COLUMNS].mean()

    checks = {
        "cpu_usage": ("CPU usage", "%"),
        "memory_usage": ("Memory usage", "%"),
        "response_time": ("Response time", "ms"),
        "error_count": ("Error count", "errors"),
        "network_latency": ("Network latency", "ms"),
    }

    for col, (label, unit) in checks.items():
        value = row[col]
        avg = averages[col]
        # Flag anything notably (30%+) above the dataset average
        if avg > 0 and value > avg * 1.3:
            reasons.append(
                f"{label} is {value:.1f}{unit}, noticeably higher than the "
                f"typical average of {avg:.1f}{unit}."
            )

    if not reasons:
        reasons.append("Metrics are close to typical ranges for this system.")

    return reasons


def predict_single(model, scaler, feature_row):
    """
    Predict anomaly status for a single new incident (used if you want to
    score a brand-new record instead of picking one from the dataset).
    feature_row must be a list/array in NUMERIC_COLUMNS order.
    """
    X_scaled = scaler.transform([feature_row])
    label = model.predict(X_scaled)[0]
    score = -model.decision_function(X_scaled)[0]
    return ("Anomaly" if label == -1 else "Normal"), score
