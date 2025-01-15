from sklearn.cluster import KMeans
from model_results import save_model_results

def train_kmeans(X):
    model = KMeans(n_clusters=3)
    model.fit(X)

    cluster_centers = model.cluster_centers_
    labels = model.labels_
    inertia = model.inertia_

    results = {
        'cluster_centers': cluster_centers.tolist(),
        'labels': labels.tolist(),
        'inertia': inertia,
        'model': 'kmeans'
    }

    save_model_results(results, 'kmeans')
    return results
