"""
generate_dataset.py
--------------------
Creates a synthetic IT incident dataset (data/incidents.csv) with ~1000 rows.

The dataset contains normal and abnormal (anomalous) incidents so that
Isolation Forest has something meaningful to detect, and it contains
realistic error messages + resolutions so that the similarity search
and LLM recommendation steps have good material to work with.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Make results repeatable
random.seed(42)
np.random.seed(42)

NUM_RECORDS = 1000
CATEGORIES = ["Database", "Network", "API", "Server", "Application"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]

# Each category has a set of realistic error messages paired with a
# realistic resolution. This is what powers the "similar incident search"
# and the AI recommendation later on.
INCIDENT_TEMPLATES = {
    "Database": [
        ("Database connection timeout while executing query on orders table",
         "Restarted the database connection pool and increased max connection timeout."),
        ("Checkout service unable to connect to database",
         "Identified exhausted connection pool; scaled DB connections and restarted service."),
        ("Deadlock detected on inventory table during batch update",
         "Rewrote the batch update to process smaller chunks to avoid lock contention."),
        ("High replication lag on read-replica database",
         "Throttled write traffic temporarily and resized the replica instance."),
        ("Database CPU usage spiked to 98 percent during nightly job",
         "Rescheduled the nightly job and added an index to speed up the slow query."),
        ("Payment API database connection timeout",
         "Increased connection pool size and added retry logic with exponential backoff."),
    ],
    "Network": [
        ("Packet loss detected between application server and load balancer",
         "Replaced a faulty network interface card on the affected host."),
        ("DNS resolution failures for internal service discovery",
         "Restarted internal DNS resolver and cleared stale cache entries."),
        ("High network latency between data center regions",
         "Rerouted traffic through backup link while ISP resolved a fiber cut."),
        ("VPN tunnel dropped intermittently for remote office",
         "Updated VPN gateway firmware and adjusted MTU settings."),
        ("Firewall blocking legitimate traffic to API gateway",
         "Corrected an overly restrictive firewall rule introduced in last deployment."),
        ("Network latency spike affecting checkout service",
         "Identified congested switch port and load balanced traffic across two links."),
    ],
    "API": [
        ("API gateway returning 502 Bad Gateway errors intermittently",
         "Increased upstream timeout settings and restarted the gateway pods."),
        ("Authentication API rate limit exceeded during peak traffic",
         "Raised rate limit thresholds and enabled request queueing."),
        ("Third-party payment API returning inconsistent responses",
         "Added circuit breaker and fallback logic for the third-party integration."),
        ("REST API response time degraded after latest deployment",
         "Rolled back the deployment that introduced an inefficient serialization step."),
        ("API endpoint returning 500 errors for bulk upload requests",
         "Fixed a null pointer bug in the bulk upload validation logic."),
        ("Search API timing out for large result sets",
         "Added pagination and result caching to reduce query load."),
    ],
    "Server": [
        ("Server CPU usage sustained above 95 percent for 20 minutes",
         "Identified a runaway background process and restarted the service."),
        ("Server memory usage steadily increasing, suspected memory leak",
         "Deployed patched build fixing a memory leak in the caching layer."),
        ("Web server unresponsive, health checks failing",
         "Restarted the web server process and increased worker thread count."),
        ("Disk space on application server reached critical threshold",
         "Cleared old log files and expanded the disk volume."),
        ("Server load average unusually high during off-peak hours",
         "Found a misconfigured cron job running every minute instead of hourly."),
        ("Application server crashed with out of memory error",
         "Increased server memory allocation and tuned garbage collection settings."),
    ],
    "Application": [
        ("Application throwing null reference exceptions on user login",
         "Deployed hotfix correcting a null check in the login validation code."),
        ("Background job queue backed up, jobs not processing",
         "Restarted the worker service and scaled up the number of workers."),
        ("Application error rate spiked after feature flag rollout",
         "Rolled back the feature flag pending further testing."),
        ("Users reporting slow page load times on the dashboard",
         "Optimized a slow database query powering the dashboard widgets."),
        ("Scheduled report generation job failing silently",
         "Added proper error logging and fixed a timezone conversion bug."),
        ("Application session handling causing users to be logged out randomly",
         "Fixed a session token expiry bug introduced in the last release."),
    ],
}


def random_timestamp(days_back=90):
    """Return a random timestamp within the last `days_back` days."""
    start = datetime.now() - timedelta(days=days_back)
    random_seconds = random.randint(0, days_back * 24 * 60 * 60)
    return start + timedelta(seconds=random_seconds)


def severity_for(is_anomaly, error_count):
    """Pick a severity that roughly matches how bad the incident looks."""
    if is_anomaly and error_count > 15:
        return random.choices(SEVERITIES, weights=[0.05, 0.15, 0.35, 0.45])[0]
    elif is_anomaly:
        return random.choices(SEVERITIES, weights=[0.10, 0.30, 0.40, 0.20])[0]
    else:
        return random.choices(SEVERITIES, weights=[0.55, 0.30, 0.12, 0.03])[0]


# Each category has a slightly different "typical signature" in the metrics,
# just like real incident categories do in practice (e.g. Network incidents
# tend to show up as latency spikes, Server incidents as CPU/memory spikes).
# This gives the Random Forest classifier real signal to learn from, instead
# of categories that are statistically independent of the metrics.
CATEGORY_METRIC_BIAS = {
    "Database":    {"cpu": 10, "memory": 5,  "response": 400, "errors": 2, "latency": 0},
    "Network":     {"cpu": 0,  "memory": 0,  "response": 100, "errors": 1, "latency": 180},
    "API":         {"cpu": 5,  "memory": 5,  "response": 250, "errors": 3, "latency": 40},
    "Server":      {"cpu": 20, "memory": 20, "response": 100, "errors": 1, "latency": 0},
    "Application": {"cpu": 5,  "memory": 10, "response": 150, "errors": 2, "latency": 20},
}


def generate_row(incident_id, is_anomaly):
    category = random.choice(CATEGORIES)
    error_message, resolution = random.choice(INCIDENT_TEMPLATES[category])
    bias = CATEGORY_METRIC_BIAS[category]

    if is_anomaly:
        # Abnormal metrics: high resource usage, high latency, more errors
        cpu_usage = round(np.clip(np.random.normal(80 + bias["cpu"] * 0.3, 8), 60, 100), 2)
        memory_usage = round(np.clip(np.random.normal(82 + bias["memory"] * 0.3, 7), 60, 100), 2)
        response_time = round(np.clip(np.random.normal(1800 + bias["response"] * 2, 500), 800, 6000), 2)
        error_count = int(np.clip(np.random.normal(15 + bias["errors"] * 2, 7), 5, 60))
        network_latency = round(np.clip(np.random.normal(250 + bias["latency"] * 1.5, 100), 100, 1000), 2)
    else:
        # Normal metrics: healthy ranges, still shifted per category
        cpu_usage = round(np.clip(np.random.normal(35 + bias["cpu"], 10), 5, 78), 2)
        memory_usage = round(np.clip(np.random.normal(38 + bias["memory"], 10), 10, 78), 2)
        response_time = round(np.clip(np.random.normal(150 + bias["response"], 70), 50, 600), 2)
        error_count = int(np.clip(np.random.normal(1 + bias["errors"] * 0.5, 1.3), 0, 6))
        network_latency = round(np.clip(np.random.normal(35 + bias["latency"], 20), 5, 150), 2)

    severity = severity_for(is_anomaly, error_count)

    return {
        "incident_id": f"INC{incident_id:05d}",
        "timestamp": random_timestamp(),
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "response_time": response_time,
        "error_count": error_count,
        "network_latency": network_latency,
        "error_message": error_message,
        "category": category,
        "severity": severity,
        "resolution": resolution,
    }


def generate_dataset(num_records=NUM_RECORDS, anomaly_ratio=0.18):
    rows = []
    num_anomalies = int(num_records * anomaly_ratio)
    anomaly_flags = [True] * num_anomalies + [False] * (num_records - num_anomalies)
    random.shuffle(anomaly_flags)

    for i, is_anomaly in enumerate(anomaly_flags, start=1):
        rows.append(generate_row(i, is_anomaly))

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_dataset()
    output_path = os.path.join("data", "incidents.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Generated {len(dataset)} incident records -> {output_path}")
    print(dataset["category"].value_counts())
    print(dataset["severity"].value_counts())
