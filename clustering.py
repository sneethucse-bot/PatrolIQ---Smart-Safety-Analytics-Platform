import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def run_geographic_clustering(df, silhouette_sample_size=10000):
    X = df[["Lat_norm", "Lon_norm", "Crime_Severity_Score"]].values

    kmeans = MiniBatchKMeans(n_clusters=6, batch_size=10000, random_state=42)
    labels = kmeans.fit_predict(X)

    # Silhouette score on a subset safely using integer positions
    if len(X) > silhouette_sample_size:
        sample_idx = np.random.choice(len(X), size=silhouette_sample_size, replace=False)
        X_sample = X[sample_idx]
        labels_sample = labels[sample_idx]
    else:
        X_sample = X
        labels_sample = labels

    score = silhouette_score(X_sample, labels_sample)
    print(f"Geographic KMeans silhouette score: {score:.2f}")

    return pd.Series(labels, index=df.index)


def run_temporal_clustering(df, silhouette_sample_size=10000):
    X = df[["Hour", "Month", "Is_Weekend"]].values

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=5000, random_state=42)
    labels = kmeans.fit_predict(X)

    # Silhouette score on subset safely using integer positions
    if len(X) > silhouette_sample_size:
        sample_idx = np.random.choice(len(X), size=silhouette_sample_size, replace=False)
        X_sample = X[sample_idx]
        labels_sample = labels[sample_idx]
    else:
        X_sample = X
        labels_sample = labels

    score = silhouette_score(X_sample, labels_sample)
    print(f"Temporal KMeans silhouette score: {score:.2f}")

    return pd.Series(labels, index=df.index)