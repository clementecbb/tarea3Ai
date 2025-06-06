import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import clip

# Configuración
DATASET = 'voc-pascal'  # Opciones: 'simple1k', 'voc-pascal'
MODEL = 'dinov2'  # Opciones: 'resnet34', 'clip', 'dinov2'
data_dir = 'VocPascal'  # Directorio de datos
image_dir = os.path.join(data_dir, 'JPEGImages')
list_of_images = os.path.join(data_dir, 'val_voc.txt')

if __name__ == '__main__':
    # Lectura de datos
    with open(list_of_images, "r+") as file: 
        files = [f.split('\t') for f in file]
        
    # Verificar GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocesamiento de imágenes (para ResNet y DINOv2)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
        ])
    
    # Cargar el modelo según la configuración
    model = None
    dim = 0
    
    if MODEL == 'resnet18':
        model = models.resnet18(pretrained=True).to(device)
        model.fc = torch.nn.Identity()
        dim = 512
    elif MODEL == 'resnet34':
        model = models.resnet34(pretrained=True).to(device)
        model.fc = torch.nn.Identity()
        dim = 512
    elif MODEL == 'clip':
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.encode_image
        dim = 512
    elif MODEL == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        dim = 384

    # Crear directorio para guardar características si no existe
    os.makedirs('data', exist_ok=True)
    
    # Extracción de características
    with torch.no_grad():        
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype=np.float32)        
        for i, file in enumerate(files):
            # Para VOC Pascal, el nombre del archivo está en la primera columna
            filename = os.path.join(image_dir, file[0] + '.jpg')
            image = Image.open(filename).convert('RGB')
            
            # Preprocesar según el modelo
            if MODEL == 'clip':
                image_tensor = preprocess(image).unsqueeze(0).to(device)
            else:
                image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # Extraer características
            features[i,:] = model(image_tensor).cpu().numpy()[0,:]
            
            if i % 100 == 0:
                print(f'{i}/{n_images}')
                
        # Guardar características
        feat_file = os.path.join('data', f'feat_{MODEL}_{DATASET}.npy')
        np.save(feat_file, features)
        print('Características guardadas correctamente')