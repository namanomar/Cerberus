from __future__ import annotations

import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yaml")
MAX_ALERTS = 200  # rolling window of alerts to keep in memory

# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "alerts" not in st.session_state:
    st.session_state.alerts: list[dict] = []
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True  # auto-generate transactions when API unavailable


# ──────────────────────────────────────────────────────────────────────────────
# API client helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=2.0)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def score_transaction(tx: dict) -> Optional[dict]:
    try:
        r = httpx.post(f"{API_URL}/predict", json=tx, timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Demo data generator (when API is offline)
# ──────────────────────────────────────────────────────────────────────────────

def generate_demo_transaction() -> dict:
    """Synthesise a realistic-looking transaction for demo purposes."""
    is_fraud = random.random() < 0.04
    amount = (
        random.uniform(1000, 8000) if is_fraud else random.uniform(5, 500)
    )
    score = (
        random.uniform(0.65, 0.99) if is_fraud else random.uniform(0.01, 0.35)
    )
    return {
        "transaction_id": f"TX{random.randint(100000, 999999)}",
        "fraud_score": round(score, 4),
        "risk_level": (
            "CRITICAL" if score > 0.85
            else "HIGH" if score > 0.60
            else "MEDIUM" if score > 0.30
            else "LOW"
        ),
        "component_scores": {
            "lgbm": round(random.uniform(score * 0.8, min(score * 1.2, 1.0)), 4),
            "graph": round(random.uniform(0, 1), 4),
            "anomaly": round(random.uniform(0, 1), 4),
        },
        "reasons": (
            ["Unusually high amount", "Device used by 8+ accounts", "Anomalous pattern"]
            if is_fraud else ["Normal transaction pattern"]
        ),
        "top_shap_features": [
            {"feature": "TransactionAmt", "value": amount, "shap_impact": 0.12 if is_fraud else -0.03},
            {"feature": "device_user_count", "value": random.randint(1, 15), "shap_impact": 0.08 if is_fraud else -0.01},
        ],
        "latency_ms": round(random.uniform(8, 50), 2),
        "timestamp": time.time(),
        # Extra for display
        "_amount": round(amount, 2),
        "_device": f"Device_{random.randint(1, 50)}",
        "_card": f"Card_{random.randint(1000, 9999)}",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/IEEE_logo.svg/300px-IEEE_logo.svg.png", width=80)
    st.title("🚨 Fraud Monitor")
    st.markdown("---")

    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ API Offline — Demo Mode")
        st.session_state.demo_mode = True

    st.markdown("---")
    st.subheader("Settings")
    refresh_rate = st.slider("Refresh interval (s)", 1, 10, 3)
    risk_filter = st.multiselect(
        "Show risk levels",
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default=["MEDIUM", "HIGH", "CRITICAL"],
    )
    min_score = st.slider("Min fraud score", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []
        st.session_state.total_processed = 0

    st.markdown("---")
    st.caption(f"API: `{API_URL}`")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# ──────────────────────────────────────────────────────────────────────────────
# Fetch new transactions every refresh
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.demo_mode:
    new_alerts = [generate_demo_transaction() for _ in range(random.randint(1, 5))]
else:
    new_alerts = []

st.session_state.alerts.extend(new_alerts)
st.session_state.total_processed += len(new_alerts)

# Keep rolling window
if len(st.session_state.alerts) > MAX_ALERTS:
    st.session_state.alerts = st.session_state.alerts[-MAX_ALERTS:]

# Build DataFrame
alerts_df = pd.DataFrame(st.session_state.alerts)
if not alerts_df.empty:
    alerts_df["fraud_score"] = alerts_df["fraud_score"].astype(float)
    alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"], unit="s")

    # Apply filters
    alerts_df = alerts_df[
        (alerts_df["risk_level"].isin(risk_filter))
        & (alerts_df["fraud_score"] >= min_score)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Header KPI Row
# ──────────────────────────────────────────────────────────────────────────────

st.title("🚨 Graph-Enhanced Fraud Detection — Live Dashboard")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

total = st.session_state.total_processed or 1
n_high = len(alerts_df[alerts_df["risk_level"].isin(["HIGH", "CRITICAL"])]) if not alerts_df.empty else 0
n_critical = len(alerts_df[alerts_df["risk_level"] == "CRITICAL"]) if not alerts_df.empty else 0
avg_score = alerts_df["fraud_score"].mean() if not alerts_df.empty else 0.0
fraud_rate = n_high / max(total, 1)

with col1:
    st.metric("Total Processed", f"{total:,}")
with col2:
    st.metric("High/Critical Alerts", n_high, delta=len(new_alerts))
with col3:
    st.metric("Critical Alerts", n_critical)
with col4:
    st.metric("Avg Fraud Score", f"{avg_score:.3f}")
with col5:
    st.metric("Alert Rate", f"{fraud_rate:.2%}")

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Charts Row 1
# ──────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Fraud Score Timeline")
    if not alerts_df.empty:
        timeline = alerts_df.sort_values("timestamp")
        color_map = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
        fig = px.scatter(
            timeline,
            x="timestamp",
            y="fraud_score",
            color="risk_level",
            color_discrete_map=color_map,
            size="fraud_score",
            size_max=12,
            hover_data=["transaction_id", "fraud_score", "risk_level"],
            title="",
            height=300,
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Fraud Score",
            yaxis_range=[0, 1],
            legend_title="Risk Level",
            margin=dict(l=20, r=20, t=10, b=20),
        )
        fig.add_hline(y=0.60, line_dash="dash", line_color="orange", annotation_text="HIGH threshold")
        fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="CRITICAL threshold")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Waiting for transactions …")

with col_right:
    st.subheader("🎯 Risk Distribution")
    if not alerts_df.empty:
        risk_counts = alerts_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
        fig = px.pie(
            risk_counts,
            values="count",
            names="risk_level",
            color="risk_level",
            color_discrete_map=colors,
            height=300,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")


# ──────────────────────────────────────────────────────────────────────────────
# Charts Row 2
# ──────────────────────────────────────────────────────────────────────────────

col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("🖥️ Model Component Scores")
    if not alerts_df.empty and "component_scores" in alerts_df.columns:
        try:
            comp_df = pd.json_normalize(alerts_df["component_scores"].tolist())
            comp_df.columns = ["LightGBM", "Graph", "Anomaly"]
            comp_melted = comp_df.mean().reset_index()
            comp_melted.columns = ["Model", "Average Score"]
            fig = px.bar(
                comp_melted,
                x="Model",
                y="Average Score",
                color="Model",
                color_discrete_sequence=["#3498db", "#e74c3c", "#f39c12"],
                height=300,
                text="Average Score",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(
                yaxis_range=[0, 1],
                showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Component scores unavailable")

with col_r2:
    st.subheader("💡 Top Suspicious Devices")
    if not alerts_df.empty and "_device" in alerts_df.columns:
        high_risk_df = alerts_df[alerts_df["risk_level"].isin(["HIGH", "CRITICAL"])]
        if not high_risk_df.empty:
            device_counts = (
                high_risk_df["_device"]
                .value_counts()
                .head(8)
                .reset_index()
            )
            device_counts.columns = ["Device", "Alert Count"]
            fig = px.bar(
                device_counts,
                x="Alert Count",
                y="Device",
                orientation="h",
                color="Alert Count",
                color_continuous_scale="Reds",
                height=300,
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No high-risk alerts yet")


# ──────────────────────────────────────────────────────────────────────────────
# Live Alert Table
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔔 Live Fraud Alerts")

if not alerts_df.empty:
    display_df = alerts_df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")

    # Colour-code risk levels
    def risk_badge(level: str) -> str:
        badges = {
            "CRITICAL": "🟣 CRITICAL",
            "HIGH": "🔴 HIGH",
            "MEDIUM": "🟠 MEDIUM",
            "LOW": "🟢 LOW",
        }
        return badges.get(level, level)

    display_df["risk_level"] = display_df["risk_level"].apply(risk_badge)

    cols_to_show = [c for c in ["timestamp", "transaction_id", "fraud_score", "risk_level", "reasons"]
                    if c in display_df.columns]

    # Show most recent 20 alerts
    st.dataframe(
        display_df[cols_to_show]
        .sort_values("timestamp", ascending=False)
        .head(20)
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
else:
    st.info("No alerts matching current filters.")


# ──────────────────────────────────────────────────────────────────────────────
# Manual Prediction Panel
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🔍 Manual Transaction Scoring")

with st.expander("Enter transaction details"):
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        amount = st.number_input("Amount ($)", min_value=0.01, value=500.0, step=10.0)
        product = st.selectbox("Product Code", ["W", "H", "C", "S", "R"])
    with m_col2:
        device_type = st.selectbox("Device Type", ["desktop", "mobile", "Unknown"])
        device_info = st.text_input("Device Info", "Chrome 80")
    with m_col3:
        card1 = st.number_input("Card ID", min_value=1, value=12345, step=1)
        email = st.text_input("Email Domain", "gmail.com")

    if st.button("🔍 Score This Transaction", type="primary"):
        tx_payload = {
            "TransactionAmt": amount,
            "ProductCD": product,
            "DeviceType": device_type,
            "DeviceInfo": device_info,
            "card1": int(card1),
            "P_emaildomain": email,
            "TransactionDT": 86400.0,
        }

        if api_healthy:
            result = score_transaction(tx_payload)
        else:
            result = generate_demo_transaction()
            result["fraud_score"] = round(random.uniform(0.01, 0.99), 4)
            if result["fraud_score"] > 0.85:
                result["risk_level"] = "CRITICAL"
            elif result["fraud_score"] > 0.60:
                result["risk_level"] = "HIGH"
            elif result["fraud_score"] > 0.30:
                result["risk_level"] = "MEDIUM"
            else:
                result["risk_level"] = "LOW"

        if result:
            score = result.get("fraud_score", 0)
            level = result.get("risk_level", "UNKNOWN")

            level_colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "purple"}
            color = level_colors.get(level, "gray")

            st.markdown(f"""
            <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                <h3 style="color: {color};">🚨 Risk Level: {level}</h3>
                <h4>Fraud Score: {score:.4f}</h4>
            </div>
            """, unsafe_allow_html=True)

            r_col1, r_col2, r_col3 = st.columns(3)
            comp = result.get("component_scores", {})
            with r_col1:
                st.metric("LightGBM", f"{comp.get('lgbm', 0):.4f}")
            with r_col2:
                st.metric("Graph", f"{comp.get('graph', 0):.4f}")
            with r_col3:
                st.metric("Anomaly", f"{comp.get('anomaly', 0):.4f}")

            reasons = result.get("reasons", [])
            if reasons:
                st.markdown("**Reasons:**")
                for r in reasons:
                    st.markdown(f"  • {r}")
        else:
            st.error("Scoring failed. Check API status.")


# ──────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ──────────────────────────────────────────────────────────────────────────────

time.sleep(refresh_rate)
st.rerun()
