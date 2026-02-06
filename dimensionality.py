from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap


def apply_pca(df, features=None, n_components=2):
    """
    Apply PCA to selected numeric features.
    Returns transformed array and explained variance %
    """
    if features is None:
        features = ["Latitude", "Longitude", "Hour", "Month", "Crime_Severity_Score"]

    X = df[features].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_.sum() * 100
    return X_pca, explained_var


def apply_umap(df, features=None, n_components=2, random_state=42):
    """
    Apply UMAP to selected numeric features.
    Returns transformed array.
    """
    if features is None:
        features = ["Latitude", "Longitude", "Hour", "Month", "Crime_Severity_Score"]

    X = df[features].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state
    )

    X_umap = reducer.fit_transform(X_scaled)
    return X_umap