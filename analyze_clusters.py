import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score, adjusted_rand_score
import umap
import matplotlib.pyplot as plt
import os

# Cargar datos
features = np.load('data/feat_dinov2_voc-pascal.npy')

# Cargar etiquetas
with open('VocPascal/val_voc.txt', 'r') as f:
    labels = [line.split('\t')[1].strip() for line in f.readlines()]



dims = [16, 32, 64, 128, 256]
k_values = [20, 50]
results = {}

# Clustering en espacio original
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    ri = rand_score(labels, clusters)
    ari = adjusted_rand_score(labels, clusters)
    
    results[f'original_k{k}'] = {'RI': ri, 'ARI': ari}

# Seleccionar puntos aleatorios para visualización
np.random.seed(42)
idx = np.random.choice(len(features), 100, replace=False)

# Convertir etiquetas de texto a números
unique_labels = np.unique(labels)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = [label_to_int[label] for label in labels]

# PCA
for dim in dims:
    pca = PCA(n_components=dim)
    features_pca = pca.fit_transform(features[idx])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    scatter = plt.scatter(
        features_pca[:, 0], features_pca[:, 1], 
        c=[numeric_labels[i] for i in idx], cmap='tab20'
    )
    plt.title('PCA 2D')
    
    # UMAP
    reducer = umap.UMAP(n_components=2)
    features_umap = reducer.fit_transform(features[idx])
    
    plt.subplot(122)
    scatter = plt.scatter(
        features_umap[:, 0], features_umap[:, 1],
        c=[numeric_labels[i] for i in idx], cmap='tab20'
    )
    plt.title('UMAP 2D')
    
    plt.tight_layout()
    plt.savefig('scatter_plots.png')

if __name__ == '__main__':
    print_results_table()
    create_scatter_plots()