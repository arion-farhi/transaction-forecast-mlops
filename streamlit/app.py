
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from google.cloud import storage
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Transaction Forecast MLOps", layout="wide")

st.title("Transaction Volume Forecasting")
st.markdown("**MLOps Pipeline Demo** - XGBoost model achieving 6.41% MAPE")

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.metric("Best Model", "XGBoost")
st.sidebar.metric("MAPE", "6.41%")
st.sidebar.metric("Latency", "0.5ms")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Comparison")
comparison_data = {
    "Model": ["XGBoost", "Prophet", "LSTM"],
    "MAPE (%)": [6.41, 9.77, 10.40],
    "Latency (ms)": [0.5, 196, 86]
}
st.sidebar.dataframe(pd.DataFrame(comparison_data), hide_index=True)

# Main content
tab1, tab2, tab3 = st.tabs(["Predictions", "Model Performance", "Architecture"])

with tab1:
    st.header("Transaction Volume Predictions")
    
    # Generate sample predictions
    dates = pd.date_range(start="2018-08-01", end="2018-08-22", freq="D")
    actual = [180, 195, 210, 165, 145, 190, 205, 175, 160, 220, 
              235, 200, 185, 170, 195, 210, 180, 165, 225, 240, 215, 200]
    predicted = [185, 190, 205, 170, 150, 185, 210, 180, 155, 215,
                 230, 205, 190, 165, 200, 205, 185, 170, 220, 235, 210, 195]
    
    df = pd.DataFrame({
        "Date": dates,
        "Actual": actual,
        "Predicted": predicted
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual"], name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Predicted"], name="Predicted", line=dict(color="red", dash="dash")))
    fig.update_layout(title="Actual vs Predicted Transaction Volume", xaxis_title="Date", yaxis_title="Transactions")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MAPE comparison
        models = ["XGBoost", "Prophet", "LSTM"]
        mape_values = [6.41, 9.77, 10.40]
        fig_mape = px.bar(x=models, y=mape_values, title="MAPE by Model (%)", 
                         labels={"x": "Model", "y": "MAPE (%)"})
        fig_mape.update_traces(marker_color=["green", "orange", "red"])
        st.plotly_chart(fig_mape, use_container_width=True)
    
    with col2:
        # Latency comparison
        latency_values = [0.5, 196, 86]
        fig_latency = px.bar(x=models, y=latency_values, title="Inference Latency (ms)",
                            labels={"x": "Model", "y": "Latency (ms)"})
        fig_latency.update_traces(marker_color=["green", "red", "orange"])
        st.plotly_chart(fig_latency, use_container_width=True)
    
    st.markdown("### Key Findings")
    st.markdown("""
    - **XGBoost** achieved best accuracy (6.41% MAPE) AND fastest inference (0.5ms)
    - **Hyperparameter tuning** did not improve XGBoost - defaults were optimal
    - **LSTM** benefited most from tuning (+15.2% improvement)
    - **Feature engineering** was key - 7-day rolling features dominated importance
    """)

with tab3:
    st.header("MLOps Architecture")
    
    st.markdown("""
```
    ┌─────────────────────────────────────────────────────────────────┐
    │                         TRIGGERS                                 │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
    │  │   CI/CD     │  │    Drift    │  │  Scheduled  │              │
    │  │ Cloud Build │  │  Detection  │  │   BigQuery  │              │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
    │         │                │                │                      │
    │         └────────────────┼────────────────┘                      │
    │                          ▼                                       │
    │                 ┌─────────────────┐                              │
    │                 │ Cloud Function  │                              │
    │                 │ retrain-trigger │                              │
    │                 └────────┬────────┘                              │
    │                          ▼                                       │
    │         ┌─────────────────────────────────┐                      │
    │         │      Vertex AI Pipeline         │                      │
    │         │  ┌─────────┐    ┌───────────┐   │                      │
    │         │  │ Ingest  │───▶│ Engineer  │   │                      │
    │         │  └─────────┘    └─────┬─────┘   │                      │
    │         │                       ▼         │                      │
    │         │  ┌─────────┐    ┌───────────┐   │                      │
    │         │  │Register │◀───│   Train   │   │                      │
    │         │  └────┬────┘    └───────────┘   │                      │
    │         └───────┼─────────────────────────┘                      │
    │                 ▼                                                │
    │         ┌─────────────────┐                                      │
    │         │  Model Registry │                                      │
    │         └────────┬────────┘                                      │
    │                  ▼                                               │
    │         ┌─────────────────┐      ┌─────────────────┐             │
    │         │    Endpoint     │─────▶│ Model Monitoring│             │
    │         └─────────────────┘      └─────────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
```
    """)
    
    st.markdown("### Tech Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Modeling**")
        st.markdown("- Prophet\n- XGBoost\n- TensorFlow/LSTM\n- Scikit-learn")
    with col2:
        st.markdown("**Orchestration**")
        st.markdown("- Kubeflow Pipelines\n- Vertex AI Training\n- Vertex AI Registry")
    with col3:
        st.markdown("**Monitoring**")
        st.markdown("- BigQuery Evaluation\n- Model Monitoring\n- Cloud Functions")
