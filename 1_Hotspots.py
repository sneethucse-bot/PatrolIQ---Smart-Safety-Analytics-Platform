# 1_Hotspots.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="📍 Crime Hotspots",
    layout="wide"
)
st.title("📍 PatrolIQ - Chicago Crime Hotspots")

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

    # Geographic cluster filter
    geo_clusters = st.sidebar.multiselect(
        "Select Geographic Clusters",
        options=df["Geo_Cluster"].unique(),
        default=df["Geo_Cluster"].unique()
    )

    # Apply filters
    df_filtered = df[
        (df["Primary Type"].isin(crime_types)) &
        (df["Geo_Cluster"].isin(geo_clusters))
    ]

    st.markdown(f"**Filtered records:** {df_filtered.shape[0]}")

    # -----------------------------
    # Crime Hotspots Map
    # -----------------------------
    st.subheader("📍 Chicago Crime Hotspots")
    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="Latitude",
        lon="Longitude",
        color="Geo_Cluster",
        hover_data=["Primary Type", "Date", "Crime_Severity_Score"],
        zoom=10,
        height=600,
        color_continuous_scale="Turbo"
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

    # -----------------------------
    # Cluster Summary
    # -----------------------------
    st.subheader("📊 Geographic Cluster Summary")
    cluster_summary = df_filtered.groupby("Geo_Cluster").agg(
        total_crimes=("Primary Type", "count"),
        most_common_crime=("Primary Type", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
        avg_severity=("Crime_Severity_Score", "mean")
    ).reset_index()
    st.dataframe(cluster_summary)
