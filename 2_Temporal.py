# streamlit_app/pages/2_Temporal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="⏰ Temporal Patterns", layout="wide")
st.title("⏰ Temporal Crime Patterns")

# -----------------------------
# Load processed data
# -----------------------------
data_path = os.path.join("data", "processed", "crime_processed_50k.csv")

if not os.path.exists(data_path):
    st.error(f"File not found: {data_path}. Run the pipeline first!")
else:
    df = pd.read_csv(data_path)
    st.success(f"Dataset loaded: {df.shape[0]} records")

    # -----------------------------
    # Sidebar filters
    # -----------------------------
    st.sidebar.header("Filters")
    
    # Crime type filter
    crime_types = st.sidebar.multiselect(
        "Select Crime Types",
        options=df["Primary Type"].unique(),
        default=df["Primary Type"].unique()
    )

    # Temporal cluster filter
    temporal_clusters = st.sidebar.multiselect(
        "Select Temporal Clusters",
        options=df["Temporal_Cluster"].unique(),
        default=df["Temporal_Cluster"].unique()
    )

    # Apply filters
    df_filtered = df[
        (df["Primary Type"].isin(crime_types)) &
        (df["Temporal_Cluster"].isin(temporal_clusters))
    ]
    st.markdown(f"**Filtered records:** {df_filtered.shape[0]}")

    # -----------------------------
    # Hourly Crime Distribution
    # -----------------------------
    st.subheader("🕒 Crimes by Hour")
    fig_hour = px.histogram(
        df_filtered,
        x="Hour",
        color="Temporal_Cluster",
        nbins=24,
        labels={"Hour": "Hour of Day"},
        title="Crime Distribution by Hour"
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # -----------------------------
    # Crimes by Day of Week
    # -----------------------------
    st.subheader("📅 Crimes by Day of Week")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig_day = px.histogram(
        df_filtered,
        x="Day_of_Week",
        color="Temporal_Cluster",
        category_orders={"Day_of_Week": day_order},
        title="Crime Distribution by Day of Week"
    )
    st.plotly_chart(fig_day, use_container_width=True)

    # -----------------------------
    # Crimes by Month / Season
    # -----------------------------
    st.subheader("🌦️ Crimes by Month and Season")
    fig_month = px.histogram(
        df_filtered,
        x="Month",
        color="Season",
        nbins=12,
        title="Monthly Crime Distribution by Season"
    )
    st.plotly_chart(fig_month, use_container_width=True)

