import os
import subprocess

# Modelos a comparar
models = ['resnet34', 'clip', 'dinov2']
dataset = 'voc-pascal'

# Crear directorio para datos si no existe
os.makedirs('data', exist_ok=True)

# Ejecutar extracción de características para cada modelo
for model in models:
    print(f"\n\n===== Extrayendo características con {model} =====\n")
    
    # Modificar el archivo compute_features.py para usar el modelo actual
    with open('compute_features.py', 'r') as file:
        content = file.read()
    
    # Reemplazar la configuración del modelo
    content = content.replace("MODEL = 'resnet34'", f"MODEL = '{model}'")
    content = content.replace("DATASET = 'simple1k'", f"DATASET = '{dataset}'")
    
    # Guardar los cambios
    with open('compute_features.py', 'w') as file:
        file.write(content)
    
    # Ejecutar el script
    subprocess.run(['python', 'compute_features.py'])

# Ejecutar comparación de similitud
print("\n\n===== Comparando modelos =====\n")
subprocess.run(['python', 'compute_similarity.py'])

print("\n\n===== Comparación completada =====\n")
print("Los resultados se han guardado en los archivos 'similarity_*.png' y 'model_comparison.png'")