"""
app.py
------
AI Incident Copilot - Streamlit dashboard.

Run with:  streamlit run app.py
"""

import os
import subprocess
import hashlib
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from utils.preprocessing import load_and_clean_data, NUMERIC_COLUMNS
from utils.anomaly_detection import train_anomaly_model, explain_anomaly
from utils.classification import train_classifier, predict_category
from utils.similarity import build_embeddings, find_most_similar
from utils.recommendation import generate_recommendation

load_dotenv()  # loads GROQ_API_KEY from .env if present

# ---------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Incident Copilot",
    page_icon="🛠️",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0; }
    .subtitle { font-size: 1.05rem; color: #4B5563; margin-top: 0; }
    div[data-testid="stMetric"] {
        background-color: #F0F5FF;
        border: 1px solid #C7D9FF;
        border-radius: 10px;
        padding: 10px;
    }
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "incidents.csv")
UPLOADED_DATA_PATH = os.path.join("data", "uploaded_incidents.csv")

REQUIRED_COLUMNS = [
    "incident_id",
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "response_time",
    "error_count",
    "network_latency",
    "error_message",
    "category",
    "severity",
    "resolution",
]


# ---------------------------------------------------------------------
# DATASET UPLOAD
# ---------------------------------------------------------------------
st.sidebar.markdown("### 📂 Incident Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV incident dataset",
    type=["csv"],
    help="Upload a CSV containing the incident columns used by this project.",
)

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    upload_key = hashlib.md5(uploaded_bytes).hexdigest()

    os.makedirs("data", exist_ok=True)
    with open(UPLOADED_DATA_PATH, "wb") as f:
        f.write(uploaded_bytes)

    try:
        uploaded_preview = pd.read_csv(UPLOADED_DATA_PATH)
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in uploaded_preview.columns
        ]

        if missing_columns:
            st.sidebar.error("Uploaded CSV is missing required columns:")
            st.sidebar.write(", ".join(missing_columns))
            st.sidebar.info(
                "Required columns: " + ", ".join(REQUIRED_COLUMNS)
            )
            st.stop()

        ACTIVE_DATA_PATH = UPLOADED_DATA_PATH
        ACTIVE_DATA_KEY = upload_key
        st.sidebar.success(
            f"Uploaded: {uploaded_file.name} "
            f"({len(uploaded_preview)} rows)"
        )
    except Exception as e:
        st.sidebar.error(f"Could not read the uploaded CSV: {e}")
        st.stop()
else:
    ACTIVE_DATA_PATH = DATA_PATH
    ACTIVE_DATA_KEY = str(
        os.path.getmtime(DATA_PATH)
        if os.path.exists(DATA_PATH)
        else "default"
    )
    st.sidebar.caption("Using the default dataset: data/incidents.csv")


# ---------------------------------------------------------------------
# DATA + MODEL LOADING (cached so it only runs once per session)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_data(data_path, data_key):
    if not os.path.exists(data_path):
        if data_path == DATA_PATH:
            subprocess.run(["python", "generate_dataset.py"], check=True)
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")
    return load_and_clean_data(data_path)


@st.cache_resource(show_spinner=False)
def get_anomaly_model(df):
    return train_anomaly_model(df)


@st.cache_resource(show_spinner=False)
def get_classifier(df):
    return train_classifier(df)


@st.cache_resource(show_spinner=False)
def get_embeddings(df):
    return build_embeddings(df)


with st.spinner("Loading data and training models..."):
    raw_df = get_clean_data(ACTIVE_DATA_PATH, ACTIVE_DATA_KEY)
    anomaly_model, anomaly_scaler, df = get_anomaly_model(raw_df)
    clf_model, label_encoder, clf_metrics = get_classifier(df)
    embeddings = get_embeddings(df)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown('<p class="main-title">🛠️ AI Incident Copilot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Intelligent IT Incident Detection & Resolution Assistant</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔍 Incident Analyzer", "🧭 Similar Incidents", "🤖 AI Recommendation"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset size: {len(df)} incidents")
st.sidebar.caption(f"Classifier accuracy: {clf_metrics['accuracy']*100:.1f}%")

# Keep the currently selected incident in session state so it's shared
# across the Analyzer / Similar Incidents / AI Recommendation pages.
if "selected_incident_id" not in st.session_state:
    st.session_state.selected_incident_id = df["incident_id"].iloc[0]


# =======================================================================
# PAGE 1 — DASHBOARD
# =======================================================================
if page == "📊 Dashboard":
    total_incidents = len(df)
    anomalies = (df["anomaly_label"] == "Anomaly").sum()
    critical = (df["severity"] == "Critical").sum()
    resolved = df["resolution"].notna().sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("Anomalies Detected", int(anomalies))
    col3.metric("Critical Incidents", int(critical))
    col4.metric("Resolved Incidents", int(resolved))

    st.markdown("### Overview Charts")
    c1, c2 = st.columns(2)

    with c1:
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig1 = px.pie(cat_counts, names="category", values="count",
                       title="Incident Category Distribution",
                       color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig2 = px.bar(sev_counts, x="severity", y="count",
                       title="Severity Distribution",
                       color="severity",
                       color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        trend_df = df.copy()
        trend_df["date"] = trend_df["timestamp"].dt.date
        trend = trend_df.groupby("date").size().reset_index(name="incident_count")
        fig3 = px.line(trend, x="date", y="incident_count",
                        title="Incident Trend Over Time",
                        markers=True)
        fig3.update_traces(line_color="#1E3A8A")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        anomaly_counts = df["anomaly_label"].value_counts().reset_index()
        anomaly_counts.columns = ["status", "count"]
        fig4 = px.bar(anomaly_counts, x="status", y="count",
                       title="Normal vs Anomaly Count",
                       color="status",
                       color_discrete_map={"Normal": "#93C5FD", "Anomaly": "#1E3A8A"})
        st.plotly_chart(fig4, use_container_width=True)


# =======================================================================
# PAGE 2 — INCIDENT ANALYZER
# =======================================================================
elif page == "🔍 Incident Analyzer":
    st.markdown("### Select an Incident")
    selected_id = st.selectbox(
        "Incident ID",
        df["incident_id"].tolist(),
        index=df["incident_id"].tolist().index(st.session_state.selected_incident_id),
    )
    st.session_state.selected_incident_id = selected_id

    row = df[df["incident_id"] == selected_id].iloc[0]
    row_index = df[df["incident_id"] == selected_id].index[0]

    st.markdown("#### Incident Details")
    with st.container():
        d1, d2, d3 = st.columns(3)
        d1.write(f"**Incident ID:** {row['incident_id']}")
        d1.write(f"**Timestamp:** {row['timestamp']}")
        d2.write(f"**CPU Usage:** {row['cpu_usage']}%")
        d2.write(f"**Memory Usage:** {row['memory_usage']}%")
        d3.write(f"**Response Time:** {row['response_time']} ms")
        d3.write(f"**Network Latency:** {row['network_latency']} ms")
        st.write(f"**Error Count:** {row['error_count']}")
        st.info(f"**Error Message:** {row['error_message']}")

    st.markdown("#### ML Analysis")

    feature_row = row[NUMERIC_COLUMNS].values.astype(float)
    predicted_category, confidence, prob_dict = predict_category(
        clf_model, label_encoder, feature_row
    )

    a1, a2, a3 = st.columns(3)
    anomaly_color = "🔴" if row["anomaly_label"] == "Anomaly" else "🟢"
    a1.metric("Anomaly Status", f"{anomaly_color} {row['anomaly_label']}")
    a2.metric("Anomaly Score", f"{row['anomaly_score']:.3f}")
    a3.metric("Predicted Category", predicted_category)

    st.write(f"**Classification Confidence:** {confidence*100:.1f}%")
    prob_df = pd.DataFrame(prob_dict.items(), columns=["Category", "Probability"])
    prob_df["Probability"] = (prob_df["Probability"] * 100).round(1)
    st.bar_chart(prob_df.set_index("Category"))

    if row["anomaly_label"] == "Anomaly":
        st.markdown("**Why this incident looks abnormal:**")
        for reason in explain_anomaly(row, df):
            st.write(f"- {reason}")
    else:
        st.success("This incident's metrics are within normal ranges.")

    with st.expander("Show model evaluation metrics (Random Forest classifier)"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{clf_metrics['accuracy']*100:.1f}%")
        m2.metric("Precision", f"{clf_metrics['precision']*100:.1f}%")
        m3.metric("Recall", f"{clf_metrics['recall']*100:.1f}%")
        m4.metric("F1-score", f"{clf_metrics['f1_score']*100:.1f}%")
        st.caption("Metrics computed on a held-out 20% test split.")


# =======================================================================
# PAGE 3 — SIMILAR INCIDENTS
# =======================================================================
elif page == "🧭 Similar Incidents":
    st.markdown("### Select an Incident")
    selected_id = st.selectbox(
        "Incident ID",
        df["incident_id"].tolist(),
        index=df["incident_id"].tolist().index(st.session_state.selected_incident_id),
        key="similar_page_select",
    )
    st.session_state.selected_incident_id = selected_id

    row_index = df[df["incident_id"] == selected_id].index[0]
    current_row = df.iloc[row_index]

    st.info(f"**Current Incident Error Message:** {current_row['error_message']}")

    with st.spinner("Searching historical incidents using TF-IDF and cosine similarity..."):
        results = find_most_similar(df, embeddings, row_index, top_n=1)

    st.markdown("### Most Similar Historical Incident")
    if results:
        match = results[0]
        match_row = match["row"]
        similarity = match["similarity_percent"]

        st.markdown(f"""
        <div class="card">
        <b>Similarity:</b> {similarity}%<br>
        <b>Incident ID:</b> {match_row['incident_id']}<br>
        <b>Error Message:</b> {match_row['error_message']}<br>
        <b>Category:</b> {match_row['category']}<br>
        <b>Severity:</b> {match_row['severity']}<br>
        <b>Previous Resolution:</b> {match_row['resolution']}
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(similarity), 100))
    else:
        st.warning("No similar incident found.")


# =======================================================================
# PAGE 4 — AI RECOMMENDATION
# =======================================================================
elif page == "🤖 AI Recommendation":
    st.markdown("### Select an Incident")
    selected_id = st.selectbox(
        "Incident ID",
        df["incident_id"].tolist(),
        index=df["incident_id"].tolist().index(st.session_state.selected_incident_id),
        key="ai_page_select",
    )
    st.session_state.selected_incident_id = selected_id

    row_index = df[df["incident_id"] == selected_id].index[0]
    row = df.iloc[row_index]

    feature_row = row[NUMERIC_COLUMNS].values.astype(float)
    predicted_category, confidence, _ = predict_category(clf_model, label_encoder, feature_row)

    similar = find_most_similar(df, embeddings, row_index, top_n=1)[0]
    similar_row = similar["row"]

    if not os.environ.get("GROQ_API_KEY"):
        st.warning(
            "No GROQ_API_KEY found in environment. The app will use a rule-based "
            "fallback recommendation instead of calling the Groq LLM."
        )

    if st.button("🔎 Generate AI Recommendation", type="primary"):
        with st.spinner("Analyzing incident and generating recommendation..."):
            result = generate_recommendation(
                incident=row,
                predicted_category=predicted_category,
                anomaly_status=row["anomaly_label"],
                anomaly_score=row["anomaly_score"],
                similar_incident=similar_row["error_message"],
                similar_resolution=similar_row["resolution"],
                similarity_percent=similar["similarity_percent"],
            )

        st.markdown("### AI Incident Analysis")

        if result.get("source") == "groq":
            st.markdown(f'<div class="card">{result["raw_text"]}</div>', unsafe_allow_html=True)
        else:
            if result.get("error_note"):
                st.caption(f"⚠️ {result['error_note']}")
            st.markdown("**Probable Issue**")
            st.write(result["probable_issue"])

            st.markdown("**Supporting Evidence**")
            for item in result["supporting_evidence"]:
                st.write(f"- {item}")

            st.markdown("**Recommended Investigation Steps**")
            for item in result["investigation_steps"]:
                st.write(f"- {item}")

            st.markdown("**Suggested Resolution**")
            for item in result["suggested_resolution"]:
                st.write(f"- {item}")

        st.caption(
            "This is an AI-assisted decision-support suggestion, not a confirmed diagnosis. "
            "Always verify with logs and monitoring before acting."
        )
    else:
        st.caption("Click the button above to generate a recommendation for this incident.")
        
