from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from model_results import save_model_results

def train_hierarchical_clustering(X):
    model = AgglomerativeClustering(n_clusters=3)
    labels = model.fit_predict(X)
    linked = linkage(X, method='ward')

    results = {
        'labels': labels.tolist(),
        'linkage_matrix': linked.tolist(),
        'model': 'hierarchical_clustering'
    }

    save_model_results(results, 'hierarchical_clustering')
    return results
