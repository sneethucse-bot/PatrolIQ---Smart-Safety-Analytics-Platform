# 3_Dimensionality.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="📊 PCA & UMAP", layout="wide")
st.title("📊 Dimensionality Reduction Visualization")

# Load processed dataset
data_path = os.path.join("data", "processed", "crime_processed_50k.csv")

if not os.path.exists(data_path):
    st.error(f"File not found: {data_path}. Run the pipeline first!")
else:
    df = pd.read_csv(data_path)
    st.success(f"Dataset loaded: {df.shape[0]} records")

    # Sidebar filter for crime types
    st.sidebar.header("Filters")
    crime_types = st.sidebar.multiselect(
        "Select Crime Types",
        options=df["Primary Type"].unique(),
        default=df["Primary Type"].unique()
    )
    df_filtered = df[df["Primary Type"].isin(crime_types)]
    st.markdown(f"**Filtered records:** {df_filtered.shape[0]}")

    # PCA 2D Scatter
    st.subheader("📈 PCA 2D Scatter Plot")
    if all(col in df_filtered.columns for col in ["PCA1", "PCA2"]):
        fig_pca = px.scatter(
            df_filtered,
            x="PCA1",
            y="PCA2",
            color="Geo_Cluster" if "Geo_Cluster" in df_filtered.columns else None,
            hover_data=["Primary Type", "Hour", "Temporal_Cluster"] if "Temporal_Cluster" in df_filtered.columns else ["Primary Type", "Hour"],
            title="PCA 2D Visualization of Crime Patterns",
            height=600
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("PCA columns not found in the dataset. Run the pipeline to generate PCA.")

    # UMAP 2D Scatter
    st.subheader("🔍 UMAP 2D Scatter Plot")
    if all(col in df_filtered.columns for col in ["UMAP1", "UMAP2"]):
        fig_umap = px.scatter(
            df_filtered,
            x="UMAP1",
            y="UMAP2",
            color="Geo_Cluster" if "Geo_Cluster" in df_filtered.columns else None,
            hover_data=["Primary Type", "Hour", "Temporal_Cluster"] if "Temporal_Cluster" in df_filtered.columns else ["Primary Type", "Hour"],
            title="UMAP 2D Visualization of Crime Patterns",
            height=600
        )
        st.plotly_chart(fig_umap, use_container_width=True)
    else:
        st.warning("UMAP columns not found in the dataset. Run the pipeline to generate UMAP.")


