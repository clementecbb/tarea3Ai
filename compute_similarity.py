import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os

# Configuración
DATASET = 'voc-pascal'
MODELS = ['resnet34', 'clip', 'dinov2']  # Modelos a comparar
data_dir = 'VocPascal'
image_dir = os.path.join(data_dir, 'JPEGImages')
val_file = os.path.join(data_dir, 'val_voc.txt')

# Función para calcular precisión media (mAP)
def compute_map(sim, labels):
    n = len(labels)
    mAPs = []
    
    for i in range(n):
        query_label = labels[i]
        # Crear vector de relevancia (1 si es de la misma clase, 0 si no)
        y_true = np.array([1 if labels[j] == query_label and j != i else 0 for j in range(n)])
        # Obtener puntuaciones de similitud
        y_score = sim[i]
        
        # Calcular AP para esta consulta
        sorted_indices = np.argsort(-y_score)
        y_true_sorted = y_true[sorted_indices]
        
        precisions = []
        num_relevant = 0
        for k, relevant in enumerate(y_true_sorted):
            if relevant:
                num_relevant += 1
                precisions.append(num_relevant / (k + 1))
        
        if precisions:
            mAPs.append(np.mean(precisions))
    
    return np.mean(mAPs)

if __name__ == '__main__':
    # Cargar datos
    with open(val_file, "r+") as file: 
        files = [f.split('\t') for f in file]
    
    # Extraer etiquetas (clases) de las imágenes
    labels = [f[1].strip() for f in files]
    
    # Comparar modelos
    results = {}
    
    for model in MODELS:
        # Cargar características
        feat_file = os.path.join('data', f'feat_{model}_{DATASET}.npy')
        
        if not os.path.exists(feat_file):
            print(f"Archivo de características para {model} no encontrado. Ejecute compute_features.py primero.")
            continue
            
        feats = np.load(feat_file)
        
        # Normalizar características
        norm2 = np.linalg.norm(feats, ord=2, axis=1, keepdims=True)
        feats_n = feats / norm2
        
        # Calcular matriz de similitud
        sim = feats_n @ np.transpose(feats_n)
        
        # Calcular mAP
        map_score = compute_map(sim, labels)
        results[model] = map_score
        print(f"Modelo {model}: mAP = {map_score:.4f}")
        
        # Visualizar resultados para una consulta aleatoria
        query = np.random.randint(sim.shape[0])
        k = 5  # Número de resultados a mostrar
        sim_idx = np.argsort(-sim[query])
        best_idx = sim_idx[:k+1]  # +1 porque el primero será la propia imagen
        
        print(f"\nConsulta: {files[query][0]} (Clase: {labels[query]})")
        print("Similitudes:", sim[query, best_idx])
        
        # Visualizar imágenes
        fig, ax = plt.subplots(1, k+1, figsize=(15, 3))
        for i, idx in enumerate(best_idx):
            filename = os.path.join(image_dir, files[idx][0] + '.jpg')
            im = io.imread(filename)
            im = transform.resize(im, (64, 64))
            ax[i].imshow(im)
            ax[i].set_title(f"{labels[idx]}\nSim: {sim[query, idx]:.2f}")
            ax[i].set_axis_off()
        
        # Resaltar la imagen de consulta
        ax[0].patch.set(lw=3, ec='b')
        plt.suptitle(f"Resultados de similitud para {model}")
        plt.tight_layout()
        plt.savefig(f"similarity_{model}.png")
        
    # Comparar resultados
    if results:
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.ylabel('mAP')
        plt.title('Comparación de modelos')
        plt.ylim(0, 1)
        for model, score in results.items():
            plt.text(model, score + 0.02, f"{score:.4f}", ha='center')
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.show()