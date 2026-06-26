import numpy as np
import pandas as pd
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_documents(n_clusters=15, sample_for_silhouette=5000):
    os.makedirs("data/clustering", exist_ok=True)
    model_path = "data/clustering/kmeans_model.joblib"

    data = np.load("data/index/embeddings/doc_embeddings.npz", allow_pickle=True)
    embeddings = data["embeddings"]
    doc_ids = data["doc_ids"]

    if os.path.exists(model_path):
        print("Clustering already done, loading cache.")
        kmeans = joblib.load(model_path)
        labels = kmeans.labels_
    else:
        print(f"Running KMeans with k={n_clusters} on {embeddings.shape[0]} documents...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        joblib.dump(kmeans, model_path, compress=3)

    # تقييم Silhouette على عينة (لتسريع الحساب، 522K كبير جداً لحسابه كامل)
    idx_sample = np.random.choice(len(embeddings), size=min(sample_for_silhouette, len(embeddings)), replace=False)
    score = silhouette_score(embeddings[idx_sample], labels[idx_sample])
    print(f"Silhouette Score (sampled): {score:.4f}")

    # حفظ نتائج الـ clusters
    result_df = pd.DataFrame({"doc_id": doc_ids, "cluster": labels})
    result_df.to_csv("data/clustering/doc_clusters.csv", index=False)

    # رسم بياني: تقليل الأبعاد بـ PCA لعرض الـ clusters بصرياً
    print("Generating visualization (PCA 2D)...")
    pca = PCA(n_components=2, random_state=42)
    coords_sample = pca.fit_transform(embeddings[idx_sample])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords_sample[:, 0], coords_sample[:, 1], c=labels[idx_sample], cmap="tab20", s=5, alpha=0.6)
    plt.title(f"Document Clusters (k={n_clusters}) - Silhouette: {score:.3f}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster")
    plt.savefig("data/clustering/clusters_visualization.png", dpi=150)
    print("Saved clusters_visualization.png")

    return result_df, score

if __name__ == "__main__":
    cluster_documents(n_clusters=15)